from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from wavtts.model.grpo_dataset import PromptTargetJsonl
from wavtts.model.grpo_flow import rollout_sde
from wavtts.model.speaker_embedding import SpeakerEmbeddingExtractor
from wavtts.train.train_grpo import (
    DEFAULT_CFG,
    DEFAULT_CKPT,
    dtype_from_name,
    load_model,
    speaker_model_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for WavTTS GRPO checkpoints.")
    parser.add_argument("--model-cfg", default=DEFAULT_CFG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--sampler", choices=["ode", "sde"], default="ode")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--cfg-strength", type=float, default=1.0)
    parser.add_argument("--timestep-mapping", choices=["uniform", "power", "sway_sampling"], default="power")
    parser.add_argument("--timestep-power", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--token-rate", type=float, default=25.0)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = dtype_from_name(args.model_dtype)
    model = load_model(args.ckpt, args.model_cfg, args.device, model_dtype).eval()
    dataset = PromptTargetJsonl(
        args.jsonl,
        sample_rate=int(model.target_sample_rate),
        frame_len=model.wav_frame_len,
        token_rate=args.token_rate,
        max_rows=args.max_rows,
    )
    speaker_extractor = SpeakerEmbeddingExtractor(
        model_path=speaker_model_path(args.model_cfg),
        sample_rate=int(model.target_sample_rate),
    )

    meta_zh = []
    meta_en = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in tqdm(range(len(dataset)), desc="infer"):
            task = dataset[index]
            prompt = task["prompt_wav"].unsqueeze(0).to(args.device)
            tokens = torch.cat([task["prompt_token"], task["target_token"]]).unsqueeze(0).to(args.device)
            prompt_lens = torch.tensor([task["prompt_duration"]], device=args.device)
            durations = torch.tensor(
                [task["prompt_duration"] + task["target_duration"]],
                device=args.device,
            )
            with torch.no_grad():
                speaker = speaker_extractor.extract(task["prompt_wav"], device=args.device).unsqueeze(0)
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

            prompt_len = int(task["prompt_duration"])
            total_len = int(durations[0].item())
            audio = generated[0, prompt_len:total_len].float().cpu()
            safe_utt = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task["utt"])).strip("._") or f"sample_{index:06d}"
            wav_path = output_dir / f"{safe_utt}.wav"
            torchaudio.save(str(wav_path), audio.unsqueeze(0), int(model.target_sample_rate))

            target_text = task["target_text"]
            record = {
                "utt": task["utt"],
                "prompt_wav": task["prompt_wav_path"],
                "target_text": target_text,
                "gen_wav": str(wav_path),
                "sampler": args.sampler,
                "steps": args.steps,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            line = "|".join(
                [
                    safe_utt,
                    task.get("prompt_text", ""),
                    task["prompt_wav_path"],
                    target_text,
                ]
            )
            if any("\u4e00" <= char <= "\u9fff" for char in target_text):
                meta_zh.append(line)
            else:
                meta_en.append(line)

    if meta_zh:
        (output_dir / "meta_zh.lst").write_text("\n".join(meta_zh) + "\n", encoding="utf-8")
    if meta_en:
        (output_dir / "meta_en.lst").write_text("\n".join(meta_en) + "\n", encoding="utf-8")
    print(f"Wrote inference outputs to {output_dir}")


if __name__ == "__main__":
    main()
