#!/usr/bin/env python3
"""Compare the normal WavTTS ODE path with the GRPO rollout path.

This diagnostic deliberately uses the same initial Gaussian noise for both
samplers. With ``--noise-scale 0``, any material output difference identifies
a rollout/inference-path mismatch rather than SDE exploration noise.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torchaudio

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
    parser = argparse.ArgumentParser(
        description="Compare CFM.sample ODE against GRPO rollout on one exact input."
    )
    parser.add_argument("--model-cfg", default=DEFAULT_CFG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Persistent model/weight dtype.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help="Autocast dtype; defaults to --model-dtype.",
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg-strength", type=float, default=1.5)
    parser.add_argument("--timestep-mapping", choices=["uniform", "power", "sway_sampling"], default="power")
    parser.add_argument("--timestep-power", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=None)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--token-rate", type=float, default=25.0)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def signal_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().flatten()
    candidate = candidate.float().flatten()
    difference = candidate - reference
    mse = difference.square().mean()
    signal_power = reference.square().mean().clamp_min(1e-12)
    correlation = torch.corrcoef(torch.stack((reference, candidate)))[0, 1]
    return {
        "mse": float(mse),
        "max_abs": float(difference.abs().max()),
        "snr_db": float(10.0 * torch.log10(signal_power / mse.clamp_min(1e-20))),
        "correlation": float(correlation),
    }


def main():
    args = parse_args()
    if args.row < 0:
        raise ValueError("--row must be non-negative")

    model_dtype = dtype_from_name(args.model_dtype)
    compute_dtype_name = args.compute_dtype or args.model_dtype
    compute_dtype = dtype_from_name(compute_dtype_name)
    model = load_model(args.ckpt, args.model_cfg, args.device, model_dtype).eval()
    dataset = PromptTargetJsonl(
        args.jsonl,
        sample_rate=int(model.target_sample_rate),
        frame_len=model.wav_frame_len,
        token_rate=args.token_rate,
        max_rows=args.row + 1,
    )
    task = dataset[args.row]
    prompt_rms = torch.sqrt(torch.mean(task["prompt_wav"].square())).item()
    prompt_gain = args.target_rms / max(prompt_rms, 1e-8) if prompt_rms < args.target_rms else 1.0
    prompt = (task["prompt_wav"] * prompt_gain).unsqueeze(0).to(args.device)
    tokens = torch.cat([task["prompt_token"], task["target_token"]]).unsqueeze(0).to(args.device)
    prompt_lens = torch.tensor([task["prompt_duration"]], device=args.device)
    durations = torch.tensor(
        [task["prompt_duration"] + task["target_duration"]],
        device=args.device,
    )
    extractor = SpeakerEmbeddingExtractor(
        model_path=speaker_model_path(args.model_cfg),
        sample_rate=int(model.target_sample_rate),
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    initial_noise = torch.randn(
        (1, int(durations.item())),
        device=args.device,
        dtype=torch.float32,
    )
    original_randn = torch.randn

    def fixed_initial_noise(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], torch.Size) else tuple(size)
        if shape == (int(durations.item()),):
            return initial_noise[0].to(device=kwargs["device"])
        if shape == tuple(initial_noise.shape):
            return initial_noise.to(device=kwargs["device"])
        return original_randn(*size, **kwargs)

    autocast = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if str(args.device).startswith("cuda") and compute_dtype != torch.float32
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        speaker = extractor.extract(task["prompt_wav"], device=args.device).unsqueeze(0)

        # CFM.sample and rollout_sde allocate initial noise with different
        # requested dtypes. Pin them to one FP32 tensor for an exact path test.
        with patch("torch.randn", side_effect=fixed_initial_noise):
            ode, _ = model.sample(
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
                seed=args.seed,
            )
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

    start = int(prompt_lens.item())
    end = int(durations.item())
    ode_audio = ode[0, start:end].float().cpu() / prompt_gain
    rollout_audio = rollout.final[0, start:end].float().cpu() / prompt_gain
    metrics = signal_metrics(ode_audio, rollout_audio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_dir / "ode.wav"), ode_audio.unsqueeze(0), int(model.target_sample_rate))
    torchaudio.save(
        str(output_dir / f"rollout_noise{args.noise_scale:g}.wav"),
        rollout_audio.unsqueeze(0),
        int(model.target_sample_rate),
    )
    print(
        "DIAGNOSTIC "
        f"utt={task['utt']} model_dtype={args.model_dtype} "
        f"compute_dtype={compute_dtype_name} steps={args.steps} "
        f"noise_scale={args.noise_scale:g} seed={args.seed} "
        + " ".join(f"{key}={value:.8g}" for key, value in metrics.items()),
        flush=True,
    )
    print(f"Wrote {output_dir / 'ode.wav'}", flush=True)
    print(f"Wrote {output_dir / f'rollout_noise{args.noise_scale:g}.wav'}", flush=True)


if __name__ == "__main__":
    main()
