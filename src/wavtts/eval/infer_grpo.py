from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from wavtts.eval.infer_xpred_flow import load_token_model
from wavtts.model.grpo_dataset import PromptTargetJsonl, sample_prefix_length
from wavtts.model.grpo_flow import rollout_sde
from wavtts.model.speaker_embedding import SpeakerEmbeddingExtractor
from wavtts.train.train_grpo import (
    DEFAULT_CFG,
    DEFAULT_CKPT,
    autocast_context,
    dtype_from_name,
    speaker_model_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for WavTTS GRPO checkpoints.")
    parser.add_argument("--model-cfg", default=DEFAULT_CFG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument(
        "--compute-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Autocast dtype for rollout/ODE forward passes; model parameters remain in model-dtype.",
    )
    parser.add_argument("--sampler", choices=["ode", "sde"], default="ode")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--cfg-strength", type=float, default=1.0)
    parser.add_argument("--timestep-mapping", choices=["uniform", "power", "sway_sampling"], default="power")
    parser.add_argument("--timestep-power", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--token-rate", type=float, default=25.0)
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Normalize low-RMS prompt audio like the standard token-to-wave inference path.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--data-mode", choices=["paired", "self_prefix"], default="paired")
    parser.add_argument("--prompt-fraction-min", type=float, default=0.0)
    parser.add_argument("--prompt-fraction-max", type=float, default=0.3)
    parser.add_argument("--prefix-seed", type=int, default=666)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fix generation randomness per global row index for paired checkpoint comparisons.",
    )
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of inference workers sharing the JSONL rows.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index in [0, num-shards).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    if not 0.0 <= args.prompt_fraction_min <= args.prompt_fraction_max < 1.0:
        raise ValueError("Prompt fractions must satisfy 0 <= min <= max < 1.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_dtype = dtype_from_name(args.compute_dtype)
    model_cfg = OmegaConf.load(args.model_cfg)
    model = load_token_model(
        model_cfg,
        args.ckpt,
        args.device,
        args.model_dtype,
    ).eval()
    dataset = PromptTargetJsonl(
        args.jsonl,
        sample_rate=int(model.target_sample_rate),
        frame_len=model.wav_frame_len,
        token_rate=args.token_rate,
        max_rows=args.max_rows,
        data_mode=args.data_mode,
    )
    speaker_extractor = SpeakerEmbeddingExtractor(
        model_path=speaker_model_path(args.model_cfg),
        sample_rate=int(model.target_sample_rate),
    )

    meta_zh = []
    meta_en = []
    shard_suffix = "" if args.num_shards == 1 else f".shard{args.shard_index:05d}-of-{args.num_shards:05d}"
    manifest_path = output_dir / f"manifest{shard_suffix}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        row_indices = range(args.shard_index, len(dataset), args.num_shards)
        skipped = 0
        for index in tqdm(row_indices, desc=f"infer[{args.shard_index}/{args.num_shards}]"):
            task = dataset[index]
            if args.data_mode == "self_prefix":
                total_len = int(task["duration"])
                prompt_len, prompt_fraction = sample_prefix_length(
                    total_len,
                    frame_len=model.wav_frame_len,
                    min_fraction=args.prompt_fraction_min,
                    max_fraction=args.prompt_fraction_max,
                    seed=args.prefix_seed + index * 1_000_003,
                )
                raw_prompt = task["audio"]
                tokens = task["speech_token"].unsqueeze(0).to(args.device)
                prompt_wav_path = task["audio_path"]
                target_text = ""
                language = task.get("language") or args.language
            else:
                prompt_len = int(task["prompt_duration"])
                total_len = prompt_len + int(task["target_duration"])
                prompt_fraction = prompt_len / total_len
                raw_prompt = task["prompt_wav"]
                tokens = torch.cat([task["prompt_token"], task["target_token"]]).unsqueeze(0).to(args.device)
                prompt_wav_path = task["prompt_wav_path"]
                target_text = task["target_text"]
                language = task.get("language") or (
                    "zh" if any("\u4e00" <= char <= "\u9fff" for char in target_text) else "en"
                )
            safe_utt = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task["utt"])).strip("._") or f"sample_{index:06d}"
            wav_path = output_dir / f"{safe_utt}.wav"
            generation_seed = None if args.seed is None else (args.seed + index * 1_000_003) % (2**63 - 1)
            if wav_path.is_file() and wav_path.stat().st_size > 44:
                skipped += 1
            else:
                if generation_seed is not None:
                    torch.manual_seed(generation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(generation_seed)
                prompt_rms = torch.sqrt(torch.mean(raw_prompt.square())).item()
                prompt_gain = (
                    args.target_rms / max(prompt_rms, 1e-8)
                    if prompt_rms < args.target_rms
                    else 1.0
                )
                prompt = (raw_prompt * prompt_gain).unsqueeze(0).to(args.device)
                prompt_lens = torch.tensor([prompt_len], device=args.device)
                durations = torch.tensor([total_len], device=args.device)
                with torch.no_grad(), autocast_context(args.device, compute_dtype):
                    speaker = speaker_extractor.extract(raw_prompt, device=args.device).unsqueeze(0)
                    if args.sampler == "sde":
                        rollout = rollout_sde(
                            model,
                            cond=prompt,
                            speech_tokens=tokens,
                            prompt_lens=prompt_lens,
                            durations=durations,
                            spk_embedding=speaker,
                            steps=args.steps,
                            noise_scale=args.noise_scale,
                            cfg_strength=args.cfg_strength,
                            timestep_mapping=args.timestep_mapping,
                            timestep_power=args.timestep_power,
                            sway_sampling_coef=args.sway_sampling_coef,
                            shift=args.shift,
                        )
                        generated = rollout.final
                    else:
                        generated, _ = model.sample(
                            cond=prompt,
                            text=None,
                            speech_tokens=tokens,
                            spk_embedding=speaker,
                            lens=prompt_lens,
                            duration=durations,
                            steps=args.steps,
                            cfg_strength=args.cfg_strength,
                            timestep_mapping=args.timestep_mapping,
                            timestep_power=args.timestep_power,
                            sway_sampling_coef=args.sway_sampling_coef,
                            shift=args.shift,
                            use_epss=args.timestep_mapping == "sway_sampling",
                        )

                audio = generated[0, prompt_len:total_len].float().cpu() / prompt_gain
                torchaudio.save(str(wav_path), audio.unsqueeze(0), int(model.target_sample_rate))

            record = {
                "utt": task["utt"],
                "prompt_wav": prompt_wav_path,
                "target_text": target_text,
                "gen_wav": str(wav_path),
                "generation_seed": generation_seed,
                "sampler": args.sampler,
                "steps": args.steps,
                "cfg_strength": args.cfg_strength,
                "prompt_fraction": prompt_fraction,
                "prompt_duration": prompt_len,
                "total_duration": total_len,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            line = "|".join(
                [
                    safe_utt,
                    task.get("prompt_text", ""),
                    prompt_wav_path,
                    target_text,
                ]
            )
            if language == "zh":
                meta_zh.append(line)
            else:
                meta_en.append(line)

    for name, lines in (("meta_zh", meta_zh), ("meta_en", meta_en)):
        meta_path = output_dir / f"{name}{shard_suffix}.lst"
        if lines:
            meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            meta_path.unlink(missing_ok=True)
    print(f"Wrote inference outputs to {output_dir}; skipped_existing={skipped}")


if __name__ == "__main__":
    main()
