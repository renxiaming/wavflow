from __future__ import annotations

import argparse
import copy
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from wavtts.model import CFM
from wavtts.model.grpo_dataset import PromptTargetJsonl, sample_prefix_length
from wavtts.model.grpo_flow import (
    concatenate_rollouts,
    gaussian_mean_kl,
    replay_log_prob_step,
    rollout_log_probs,
    rollout_sde,
    slice_rollout,
)
from wavtts.model.lora import adapter_state_dict, inject_lora, lora_disabled, lora_modules
from wavtts.model.repa import WavLMTeacher
from wavtts.model.speaker_embedding import SpeakerEmbeddingExtractor
from wavtts.model.utils import get_tokenizer
from wavtts.rewards import SimWerReward


DEFAULT_CKPT = (
    "/home/A02_tmpdata1/xmren/code/grpo_flow/wavtts3.1_grpo/"
    "portable_wavtts_grpo/ckpts/model_80000.pt"
)
DEFAULT_CFG = str(
    Path(__file__).resolve().parents[1]
    / "configs"
    / "WavTTS_xpred_spkconcat_grpo.yaml"
)
DEFAULT_JSONL = (
    "/home/A02_tmpdata0/xmren/code/grpo_flow/wavtts3.1_grpo/"
    "data/grpo_seed/zh/seedtest_zh_cosyvoice2_tokens.jsonl"
)
DEFAULT_REWARD_ROOT = (
    "/home/A02_tmpdata0/xmren/code/grpo_flow/wavtts3.1_grpo/"
    "portable_wavtts_grpo/third_party/sim_wer_test"
)


def build_parser():
    parser = argparse.ArgumentParser(description="Fixed-variance flow GRPO for WavTTS.")
    parser.add_argument("--model-cfg", default=DEFAULT_CFG)
    parser.add_argument("--init-ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--output-dir", default="./exp/nar_wav_tts_grpo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-device", default=None)
    parser.add_argument("--reward-device", default=None)
    parser.add_argument("--sim-device", default=None)
    parser.add_argument("--wer-device", default=None)
    parser.add_argument("--mos-device", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--data-mode", choices=["paired", "self_prefix"], default="paired")
    parser.add_argument("--prompt-fraction-min", type=float, default=0.0)
    parser.add_argument("--prompt-fraction-max", type=float, default=0.3)
    parser.add_argument(
        "--batch-size-frame",
        "--batch-size-frames",
        dest="batch_size_frames",
        type=int,
        default=0,
        help="Maximum waveform frames per GPU rollout/replay microbatch; 0 disables splitting.",
    )
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--save-per-updates", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--prompts-per-update",
        type=int,
        default=1,
        help="Number of distinct prompts rolled out before one PPO update.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument(
        "--train-timestep-fraction",
        type=float,
        default=1.0,
        help="Fraction of rollout timesteps replayed for each PPO update.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lr-scheduler", choices=["constant", "linear"], default="constant")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--adv-clip-max",
        type=float,
        default=5.0,
        help="Clip group-relative advantages before the PPO objective; <=0 disables clipping.",
    )
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--rollout-steps-min", type=int, default=0,
        help="Randomize rollout steps in [min, steps]; 0 keeps a fixed step count.",
    )
    parser.add_argument("--sde-window-size", type=int, default=0)
    parser.add_argument("--sde-window-start-min", type=int, default=0)
    parser.add_argument("--sde-window-start-max", type=int, default=0)
    cfg_group = parser.add_mutually_exclusive_group()
    cfg_group.add_argument(
        "--use-cfg",
        dest="use_cfg",
        action="store_true",
        help="Use CFG during rollout and replay (official original Flow-GRPO config default).",
    )
    cfg_group.add_argument(
        "--no-cfg",
        dest="use_cfg",
        action="store_false",
        help="Disable classifier-free guidance; cfg_strength is forced to 1.0.",
    )
    parser.set_defaults(use_cfg=True)
    parser.add_argument("--cfg-strength", type=float, default=1.0)
    parser.add_argument("--timestep-mapping", choices=["uniform", "power", "sway_sampling"], default="uniform")
    parser.add_argument("--timestep-power", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=None)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--token-rate", type=float, default=25.0)
    parser.add_argument("--wer-weight", type=float, default=0.5)
    parser.add_argument("--sim-weight", type=float, default=0.5)
    parser.add_argument("--mos-weight", type=float, default=0.0)
    parser.add_argument("--reward-std-eps", type=float, default=1e-4)
    parser.add_argument(
        "--sft-anchor-coef", type=float, default=0.0,
        help="Weight of prompt-waveform x-pred/v-loss + Mel/REPA/STFT preservation loss.",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Normalize low-RMS prompt audio to this level, matching inference.",
    )
    parser.add_argument("--reward-language", choices=["auto", "zh", "en"], default="auto")
    parser.add_argument("--reward-backend", choices=["server", "resident", "script", "direct"], default="resident")
    parser.add_argument("--reward-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--reward-server-host", default="127.0.0.1")
    parser.add_argument("--reward-server-port", type=int, default=29631)
    parser.add_argument("--reward-work-dir", default=None)
    parser.add_argument("--keep-reward-work", action="store_true")
    parser.add_argument("--sim-python", default=None)
    parser.add_argument("--wer-python", default=None)
    parser.add_argument("--mos-python", default=None)
    parser.add_argument(
        "--model-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="fp32",
        help="Persistent parameter dtype. GRPO WavTTS requires fp32.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Autocast dtype for rollout/replay forward passes.",
    )
    parser.add_argument("--seed", type=int, default=666)
    rollout_group = parser.add_mutually_exclusive_group()
    rollout_group.add_argument(
        "--keep-rollouts",
        dest="keep_rollouts",
        action="store_true",
        help="Keep generated rollout wavs and append per-wav reward records (default).",
    )
    rollout_group.add_argument(
        "--discard-rollouts",
        dest="keep_rollouts",
        action="store_false",
        help="Delete rollout wavs after scoring; JSONL records are still written.",
    )
    parser.set_defaults(keep_rollouts=True)
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--use-lora",
        dest="use_lora",
        action="store_true",
        help="Train LoRA adapters instead of all model parameters.",
    )
    lora_group.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Train all model parameters instead of LoRA adapters.",
    )
    parser.set_defaults(use_lora=False)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-targets",
        default="attn.to_q,attn.to_k,attn.to_v,attn.to_out.0",
        help="Comma-separated Linear suffixes. Set --lora-rank 0 for full fine-tuning.",
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--no-reference",
        dest="no_reference",
        action="store_true",
        help="Do not create a frozen reference model (the default GRPO configuration).",
    )
    reference_group.add_argument(
        "--with-reference",
        dest="no_reference",
        action="store_false",
        help="Create one additional frozen reference model per training rank.",
    )
    parser.set_defaults(no_reference=False)
    return parser


def parse_args():
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    pre_args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(pre_args.model_cfg)
    grpo_cfg = getattr(cfg, "grpo", {}) or {}
    if grpo_cfg:
        parser.set_defaults(
            **{
                key: value
                for key, value in OmegaConf.to_container(grpo_cfg, resolve=True).items()
                if key in defaults
            }
        )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def normalize_and_fuse_rewards(
    components: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each objective over the complete rollout batch, then fuse it."""
    if components.ndim != 2 or components.shape[1] != weights.numel():
        raise ValueError("Reward components must be [N, num_weights].")
    means = components.mean(dim=0)
    stds = components.std(dim=0, unbiased=False)
    active = (stds > float(eps)) & (weights != 0)
    normalized = torch.where(
        active,
        (components - means) / stds.clamp_min(float(eps)),
        torch.zeros_like(components),
    )
    return (normalized * weights).sum(dim=1), means, stds


def speaker_model_path(config_path: str) -> str | None:
    cfg = OmegaConf.load(config_path)
    speaker_cfg = getattr(cfg.model, "speaker_embedding", {}) or {}
    configured_path = speaker_cfg.get("model_path") if hasattr(speaker_cfg, "get") else None
    return str(configured_path) if configured_path else None


def autocast_context(device: str, dtype: torch.dtype):
    if not str(device).startswith("cuda") or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def distributed_context(requested_device: str) -> tuple[str, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif requested_device == "cuda":
        device = "cuda:0"
    else:
        device = requested_device
    return device, rank, local_rank, world_size, rank == 0


def average_gradients(model: torch.nn.Module, world_size: int):
    if world_size <= 1:
        return
    bucket_bytes = 64 * 1024 * 1024
    bucket: list[tuple[torch.Tensor, torch.Tensor]] = []
    bucket_dtype = None
    bucket_size_bytes = 0

    def flush():
        nonlocal bucket, bucket_dtype, bucket_size_bytes
        if not bucket:
            return
        flat = torch.cat([gradient.reshape(-1) for _, gradient in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for _, gradient in bucket:
            next_offset = offset + gradient.numel()
            gradient.copy_(flat[offset:next_offset].view_as(gradient))
            offset = next_offset
        bucket = []
        bucket_dtype = None
        bucket_size_bytes = 0

    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_dtype = (gradient.device, gradient.dtype)
        gradient_bytes = gradient.numel() * gradient.element_size()
        if (
            bucket
            and (gradient_dtype != bucket_dtype or bucket_size_bytes + gradient_bytes > bucket_bytes)
        ):
            flush()
        bucket.append((parameter, gradient))
        bucket_dtype = gradient_dtype
        bucket_size_bytes += gradient_bytes
    flush()


def gather_equal_tensors(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def local_group_size(global_group_size: int, world_size: int) -> int:
    if global_group_size < 2:
        raise ValueError("--group-size must be >= 2 for group-relative advantages.")
    if global_group_size < world_size:
        raise ValueError(
            f"--group-size ({global_group_size}) must be >= world size ({world_size})."
        )
    if global_group_size % world_size != 0:
        raise ValueError(
            f"--group-size ({global_group_size}) must be divisible by world size ({world_size}) "
            "so every rank has the same local batch."
        )
    return global_group_size // world_size


def rollout_microbatch_size(
    *,
    local_samples: int,
    duration: int,
    frame_len: int,
    frame_budget: int,
) -> int:
    if frame_budget <= 0:
        return local_samples
    frames_per_sample = max(1, (duration + frame_len - 1) // frame_len)
    return max(1, min(local_samples, frame_budget // frames_per_sample))


def model_from_config(config_path: str, device: str, dtype: torch.dtype) -> CFM:
    cfg = OmegaConf.load(config_path)
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model_arc = OmegaConf.to_container(cfg.model.arch, resolve=True)
    waveform_kwargs = OmegaConf.to_container(cfg.model.waveform, resolve=True)
    cfm_kwargs = OmegaConf.to_container(cfg.model.cfm, resolve=True)

    # Teacher construction belongs to the training entry point, not CFM.
    cfm_kwargs.pop("repa_teacher_model", None)
    cfm_kwargs.pop("repa_teacher_layer", None)

    tokenizer = str(cfg.model.tokenizer)
    tokenizer_path = cfg.model.tokenizer_path if tokenizer == "custom" else cfg.datasets.name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            wav_frame_len=int(waveform_kwargs["wav_frame_len"]),
        ),
        waveform_kwargs=waveform_kwargs,
        vocab_char_map=vocab_char_map,
        **cfm_kwargs,
    )
    return model.to(device=device, dtype=dtype)


def load_ema_state(path: str) -> dict[str, torch.Tensor]:
    # The SFT checkpoint also contains optimizer states and is about 11 GB.
    # mmap keeps those unused storages out of resident memory while loading EMA.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("ema_model_state_dict") or checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(f"{path} has neither ema_model_state_dict nor model_state_dict.")
    return {
        key.replace("ema_model.", ""): value
        for key, value in state.items()
        if key not in {"initted", "update", "step"}
    }


def load_model(path: str, config_path: str, device: str, dtype: torch.dtype) -> CFM:
    model = model_from_config(config_path, device, dtype)
    state = load_ema_state(path)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        bad_missing = [
            key
            for key in incompatible.missing_keys
            if "repa_alignment_head" not in key
        ]
        if bad_missing:
            raise RuntimeError(f"Checkpoint is missing model weights: {bad_missing[:20]}")
    if incompatible.unexpected_keys:
        print(
            "Ignoring checkpoint-only optional keys:",
            incompatible.unexpected_keys[:8],
            flush=True,
        )
    del state
    return model


def load_adapter_checkpoint(path: str) -> dict | None:
    if os.path.getsize(path) > 2 * 1024**3:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "adapter_state_dict" not in checkpoint or "model_state_dict" in checkpoint:
        return None
    return checkpoint


def save_checkpoint(
    path: Path,
    *,
    model: CFM,
    optimizer: torch.optim.Optimizer,
    update: int,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter_state = adapter_state_dict(model)
    checkpoint = {
        "adapter_state_dict": adapter_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "grpo_args": vars(args),
        "base_model_path": getattr(args, "base_init_ckpt", args.init_ckpt),
    }
    if not adapter_state:
        checkpoint["model_state_dict"] = model.state_dict()
    torch.save(checkpoint, path)


def save_rollout_audio(
    rollout,
    *,
    task: dict,
    output_dir: Path,
    update: int,
    sample_rate: int,
    keep: bool,
    output_gain: float = 1.0,
    rank: int = 0,
) -> list[str]:
    rollout_dir = output_dir / "rollouts" / f"update_{update:08d}" / f"rank_{rank:04d}"
    if keep:
        rollout_dir.mkdir(parents=True, exist_ok=True)
    else:
        rollout_dir.mkdir(parents=True, exist_ok=True)

    prompt_len = int(task["prompt_duration"])
    paths = []
    safe_utt = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task["utt"])).strip("._") or "sample"
    for sample_idx, duration in enumerate(rollout.durations.tolist()):
        audio = rollout.final[sample_idx, prompt_len:int(duration)].float().cpu()
        audio = audio * float(output_gain)
        if audio.numel() == 0:
            audio = torch.zeros(1)
        path = rollout_dir / f"{safe_utt}__rank{rank}__sample{sample_idx}.wav"
        torchaudio.save(str(path), audio.unsqueeze(0), sample_rate)
        paths.append(str(path))
    if not keep:
        # Reward models need the files during this update.  The caller removes
        # the directory after scoring to keep long RL runs from filling disk.
        return paths
    return paths


def save_rollout_metrics(
    *,
    generated_paths: list[str],
    reward_result: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    global_rewards: torch.Tensor,
    global_wers: torch.Tensor,
    global_sims: torch.Tensor,
    global_mos: torch.Tensor,
    reward_component_stds: torch.Tensor,
    task: dict,
    output_dir: Path,
    update: int,
    rank: int,
    world_size: int,
):
    """Save one JSONL record per wav in the complete multi-rank rollout group."""
    local_count = len(generated_paths)
    metric_values = {
        key: reward_result[key].detach().float().cpu().tolist()
        for key in ("reward", "wer", "sim", "mos", "wer_reward", "sim_reward", "mos_reward")
    }
    advantage_values = advantages.detach().float().cpu().tolist()
    group_mean = float(global_rewards.mean().item())
    group_std = float(global_rewards.std(unbiased=False).item())
    group_wer_mean = float(global_wers.mean().item())
    group_wer_std = float(global_wers.std(unbiased=False).item())
    group_sim_mean = float(global_sims.mean().item())
    group_sim_std = float(global_sims.std(unbiased=False).item())
    group_mos_mean = float(global_mos.mean().item())
    group_mos_std = float(global_mos.std(unbiased=False).item())
    local_records = []
    for sample_index, wav_path in enumerate(generated_paths):
        local_records.append(
            {
                "record_id": f"update-{update:08d}:{task['utt']}:{rank}:{sample_index}",
                "update": update,
                "utt": task["utt"],
                "rank": rank,
                "local_sample_index": sample_index,
                "group_sample_index": rank * local_count + sample_index,
                "wav_path": str(Path(wav_path).resolve()),
                "prompt_wav_path": str(Path(task["prompt_wav_path"]).resolve()),
                "prompt_text": task.get("prompt_text", ""),
                "target_text": task.get("target_text", ""),
                "language": task.get("language"),
                "prompt_duration": int(task["prompt_duration"]),
                "total_duration": int(task["total_duration"]),
                "prompt_fraction": float(task["prompt_fraction"]),
                "reward": float(metric_values["reward"][sample_index]),
                "wer": float(metric_values["wer"][sample_index]),
                "sim": float(metric_values["sim"][sample_index]),
                "mos": float(metric_values["mos"][sample_index]),
                "wer_reward": float(metric_values["wer_reward"][sample_index]),
                "sim_reward": float(metric_values["sim_reward"][sample_index]),
                "mos_reward": float(metric_values["mos_reward"][sample_index]),
                "advantage": float(advantage_values[sample_index]),
                "group_reward_mean": group_mean,
                "group_reward_std": group_std,
                "group_wer_mean": group_wer_mean,
                "group_wer_std": group_wer_std,
                "group_sim_mean": group_sim_mean,
                "group_sim_std": group_sim_std,
                "group_mos_mean": group_mos_mean,
                "group_mos_std": group_mos_std,
                "batch_wer_reward_std": float(reward_component_stds[0].item()),
                "batch_sim_reward_std": float(reward_component_stds[1].item()),
                "batch_mos_reward_std": float(reward_component_stds[2].item()),
            }
        )

    if world_size > 1:
        gathered_records: list[list[dict] | None] = [None] * world_size
        dist.all_gather_object(gathered_records, local_records)
        group_records = [
            record
            for rank_records in gathered_records
            if rank_records is not None
            for record in rank_records
        ]
    else:
        group_records = local_records

    if rank != 0:
        return
    group_records.sort(key=lambda record: record["group_sample_index"])
    sim_values = [float(record["sim"]) for record in group_records]
    sim_min = min(sim_values)
    sim_max = max(sim_values)
    for record in group_records:
        # These are group-level bounds repeated on each line for easy filtering
        # without having to regroup the aggregate JSONL first.
        record["sim_min"] = sim_min
        record["sim_max"] = sim_max

    # Keep one append-only index for the whole run. Each line corresponds to
    # one generated wav and can be filtered by update, SIM, WER, or reward.
    aggregate_path = output_dir / "rollouts" / "rollout_metrics.jsonl"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("a", encoding="utf-8") as handle:
        for record in group_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    # The switches are authoritative so that a config cannot accidentally run
    # a different training regime just because a numeric value was left over.
    if not args.use_cfg:
        args.cfg_strength = 1.0
    if not args.use_lora:
        args.lora_rank = 0
    elif args.lora_rank <= 0:
        raise ValueError("use_lora=true requires --lora-rank > 0")
    if args.no_reference:
        args.kl_coef = 0.0
    if args.target_rms <= 0:
        raise ValueError(f"--target-rms must be > 0, got {args.target_rms}")
    if args.noise_scale <= 0:
        raise ValueError(
            "--noise-scale must be > 0 for GRPO training because the policy "
            "log-probability is Gaussian. Use the normal inference script "
            "or CFM.sample() for a deterministic ODE quality check."
        )
    if not 0.0 <= args.prompt_fraction_min <= args.prompt_fraction_max < 1.0:
        raise ValueError("Prompt fractions must satisfy 0 <= min <= max < 1.")
    if args.batch_size_frames < 0:
        raise ValueError("--batch-size-frames must be >= 0.")
    if args.prompts_per_update < 1:
        raise ValueError("--prompts-per-update must be >= 1.")
    if args.rollout_steps_min < 0 or args.rollout_steps_min > args.steps:
        raise ValueError("--rollout-steps-min must be 0 or <= --steps.")
    if args.sde_window_size < 0:
        raise ValueError("--sde-window-size must be >= 0.")
    if args.sde_window_start_min < 0 or args.sde_window_start_max < args.sde_window_start_min:
        raise ValueError("Invalid SDE window start range.")
    if args.reward_std_eps <= 0:
        raise ValueError("--reward-std-eps must be > 0.")
    if args.sft_anchor_coef < 0:
        raise ValueError("--sft-anchor-coef must be >= 0.")
    if not 0.0 < args.train_timestep_fraction <= 1.0:
        raise ValueError("--train-timestep-fraction must be in (0, 1].")
    device, rank, local_rank, world_size, is_main = distributed_context(args.device)
    local_samples = local_group_size(args.group_size, world_size)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    reference_device = args.reference_device or device
    reward_device = args.reward_device or reference_device
    if world_size > 1 and args.reward_backend == "resident":
        # In local-resident mode every DDP rank owns exactly one policy,
        # one SIM worker, and one WER worker on its own local GPU.
        reward_device = device
        sim_device = device
        wer_device = device
        mos_device = device
    else:
        sim_device = args.sim_device
        wer_device = args.wer_device
        mos_device = args.mos_device
    model_dtype = dtype_from_name(args.model_dtype)
    compute_dtype = dtype_from_name(args.compute_dtype)
    if model_dtype != torch.float32:
        raise ValueError(
            "GRPO WavTTS requires --model-dtype fp32 so the SFT base and LoRA "
            "parameters are not permanently quantized. Use --compute-dtype bf16 "
            "for mixed-precision forward passes."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Loading policy from {args.init_ckpt}", flush=True)
    resume_checkpoint = load_adapter_checkpoint(args.init_ckpt)
    base_init_ckpt = args.init_ckpt
    start_update = 0
    if resume_checkpoint is not None:
        saved_args = resume_checkpoint.get("grpo_args", {})
        saved_model_dtype = saved_args.get("model_dtype")
        if saved_model_dtype not in (None, "fp32"):
            raise ValueError(
                f"Refusing to resume a {saved_model_dtype} GRPO checkpoint. "
                "Restart from the original FP32 SFT checkpoint."
            )
        base_init_ckpt = (
            resume_checkpoint.get("base_model_path")
            or saved_args.get("base_init_ckpt")
            or saved_args.get("init_ckpt")
        )
        if not base_init_ckpt or str(base_init_ckpt) == str(args.init_ckpt):
            cfg = OmegaConf.load(args.model_cfg)
            base_init_ckpt = OmegaConf.select(cfg, "grpo.init_ckpt", default=DEFAULT_CKPT)
        if not base_init_ckpt or str(base_init_ckpt) == str(args.init_ckpt):
            raise ValueError(
                f"Adapter checkpoint {args.init_ckpt} does not identify its base SFT checkpoint."
            )
        for key in (
            "use_cfg",
            "use_lora",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_targets",
        ):
            if key in saved_args:
                setattr(args, key, saved_args[key])
        if not args.use_cfg:
            args.cfg_strength = 1.0
        if not args.use_lora:
            args.lora_rank = 0
        start_update = int(resume_checkpoint.get("update", 0))
        if is_main:
            mode = "LoRA adapter" if args.use_lora else "full-parameter model"
            print(
                f"Resuming {mode} at update={start_update} with base={base_init_ckpt}",
                flush=True,
            )
    args.base_init_ckpt = str(base_init_ckpt)
    policy = load_model(str(base_init_ckpt), args.model_cfg, device, model_dtype)
    if args.lora_rank > 0:
        target_suffixes = tuple(
            item.strip() for item in str(args.lora_targets).split(",") if item.strip()
        )
        lora_stats = inject_lora(
            policy,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_suffixes=target_suffixes,
        )
        if is_main:
            print(
                f"LoRA enabled rank={args.lora_rank} alpha={args.lora_alpha:g} "
                f"modules={lora_stats.modules} trainable={lora_stats.trainable_parameters:,} "
                f"total={lora_stats.total_parameters:,}",
                flush=True,
            )
        if resume_checkpoint is not None:
            policy.load_state_dict(resume_checkpoint["adapter_state_dict"], strict=False)
    policy.train()
    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters. Use --lora-rank 0 or check --lora-targets.")
    using_lora = any(True for _ in lora_modules(policy))

    reference = None
    if args.kl_coef > 0.0 and not using_lora:
        if reference_device != device:
            raise NotImplementedError(
                "The first GRPO implementation requires reference-device == device. "
                "Use --no-reference for a single-model smoke test."
            )
        reference = copy.deepcopy(policy).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    if resume_checkpoint is not None and "optimizer_state_dict" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=(
            (lambda step: max(0.0, 1.0 - step / max(1, args.max_updates)))
            if args.lr_scheduler == "linear"
            else (lambda step: 1.0)
        ),
        last_epoch=start_update - 1,
    )
    dataset = PromptTargetJsonl(
        args.jsonl,
        sample_rate=int(policy.target_sample_rate),
        frame_len=policy.wav_frame_len,
        token_rate=args.token_rate,
        max_rows=args.max_rows,
        data_mode=args.data_mode,
    )
    speaker_extractor = SpeakerEmbeddingExtractor(
        model_path=speaker_model_path(args.model_cfg),
        sample_rate=int(policy.target_sample_rate),
    )
    repa_teacher = None
    cfm_cfg = OmegaConf.load(args.model_cfg).model.cfm
    if args.sft_anchor_coef > 0 and bool(cfm_cfg.get("use_repa_loss", False)):
        repa_teacher = WavLMTeacher(
            model_name=str(cfm_cfg.get("repa_teacher_model", "microsoft/wavlm-base-plus")),
            layer=cfm_cfg.get("repa_teacher_layer"),
            sample_rate=int(policy.target_sample_rate),
        ).to(device).eval()
    rewarder = SimWerReward(
        device=reward_device,
        sim_device=sim_device,
        wer_device=wer_device,
        mos_device=mos_device,
        wer_weight=args.wer_weight,
        sim_weight=args.sim_weight,
        mos_weight=args.mos_weight,
        language=args.reward_language,
        backend=args.reward_backend,
        sim_root=args.reward_root,
        work_dir=args.reward_work_dir,
        keep_work_dir=args.keep_reward_work,
        sim_python=args.sim_python,
        wer_python=args.wer_python,
        mos_python=args.mos_python,
        server_host=args.reward_server_host,
        server_port=args.reward_server_port,
    )

    print(
        f"GRPO rank={rank}/{world_size} policy_device={device} "
        f"reward_device={reward_device} sim_device={rewarder.sim_device} "
        f"wer_device={rewarder.wer_device} mos_device={rewarder.mos_device}",
        flush=True,
    )
    if is_main:
        print(
            f"GRPO rows={len(dataset)} world={world_size} group={args.group_size} "
            f"prompts_per_update={args.prompts_per_update} "
            f"local_group={local_samples} steps=[{args.rollout_steps_min or args.steps},{args.steps}] "
            f"sde_window={args.sde_window_size} noise={args.noise_scale} "
            f"model_dtype={args.model_dtype} compute_dtype={args.compute_dtype} "
            f"cfg={args.cfg_strength} mapping={args.timestep_mapping} "
            f"power={args.timestep_power} shift={args.shift} "
            f"target_rms={args.target_rms} kl={args.kl_coef} "
            f"reward_backend={args.reward_backend} data_mode={args.data_mode} "
            f"prefix=[{args.prompt_fraction_min},{args.prompt_fraction_max}] "
            f"batch_frames={args.batch_size_frames} "
            f"train_timestep_fraction={args.train_timestep_fraction} "
            f"reward_weights=({args.wer_weight},{args.sim_weight},{args.mos_weight}) "
            f"sft_anchor={args.sft_anchor_coef}",
            flush=True,
        )

    shuffle_epoch = None
    shuffled_indices: list[int] = []

    def run_multi_prompt_update(update: int, data_indices: list[int]) -> dict:
        """Collect several prompt groups, then perform one accumulated PPO update."""
        records = []
        for data_index in data_indices:
            raw_task = dataset[data_index]
            if args.data_mode == "self_prefix":
                total_duration = int(raw_task["duration"])
                prefix_len, sampled_fraction = sample_prefix_length(
                    total_duration,
                    frame_len=policy.wav_frame_len,
                    min_fraction=args.prompt_fraction_min,
                    max_fraction=args.prompt_fraction_max,
                    seed=args.seed + update * 1_000_003 + data_index,
                )
                raw_condition = raw_task["audio"]
                sequence_tokens = raw_task["speech_token"]
                reference_audio_path = raw_task["audio_path"]
                task = {
                    **raw_task,
                    "prompt_wav_path": reference_audio_path,
                    "prompt_wav": raw_condition,
                    "prompt_duration": prefix_len,
                    "target_duration": total_duration - prefix_len,
                    "total_duration": total_duration,
                    "prompt_fraction": sampled_fraction,
                }
            else:
                prefix_len = int(raw_task["prompt_duration"])
                total_duration = prefix_len + int(raw_task["target_duration"])
                raw_condition = raw_task["prompt_wav"]
                sequence_tokens = torch.cat([raw_task["prompt_token"], raw_task["target_token"]])
                reference_audio_path = raw_task["prompt_wav_path"]
                task = {
                    **raw_task,
                    "total_duration": total_duration,
                    "prompt_fraction": prefix_len / total_duration,
                }

            anchor_audio = raw_condition.to(device)
            anchor_tokens = (
                raw_task["speech_token"]
                if args.data_mode == "self_prefix"
                else raw_task["prompt_token"]
            ).to(device)
            schedule_generator = torch.Generator(device="cpu")
            schedule_generator.manual_seed(args.seed + update * 1_000_003 + data_index * 97)
            rollout_steps_min = args.rollout_steps_min or args.steps
            rollout_steps = int(torch.randint(
                rollout_steps_min,
                args.steps + 1,
                (1,),
                generator=schedule_generator,
            ).item())
            window_size = min(args.sde_window_size, rollout_steps)
            window_start = None
            if window_size > 0:
                max_start = min(args.sde_window_start_max, rollout_steps - window_size)
                min_start = min(args.sde_window_start_min, max_start)
                window_start = int(torch.randint(
                    min_start,
                    max_start + 1,
                    (1,),
                    generator=schedule_generator,
                ).item())

            prompt_rms = torch.sqrt(torch.mean(raw_condition.square())).item()
            prompt_gain = (
                args.target_rms / max(prompt_rms, 1e-8)
                if prompt_rms < args.target_rms
                else 1.0
            )
            condition_audio = (raw_condition * prompt_gain).to(device)
            sequence_tokens = sequence_tokens.to(device)
            with torch.no_grad():
                speaker_base = speaker_extractor.extract(raw_condition, device=device)
            speaker = speaker_base.unsqueeze(0).repeat(local_samples, 1)
            microbatch_size = rollout_microbatch_size(
                local_samples=local_samples,
                duration=total_duration,
                frame_len=policy.wav_frame_len,
                frame_budget=args.batch_size_frames,
            )
            if local_samples >= 3 and microbatch_size < 3:
                required_frames = 3 * max(
                    1, (total_duration + policy.wav_frame_len - 1) // policy.wav_frame_len
                )
                raise ValueError(
                    "Rollout microbatch is smaller than 3 samples. "
                    f"Increase --batch-size-frames to at least {required_frames} "
                    f"(current {args.batch_size_frames})."
                )

            policy.eval()
            with torch.no_grad(), autocast_context(device, compute_dtype):
                rollout_parts = []
                for start in range(0, local_samples, microbatch_size):
                    end = min(start + microbatch_size, local_samples)
                    count = end - start
                    rollout_parts.append(
                        rollout_sde(
                            policy,
                            cond=condition_audio.unsqueeze(0).repeat(count, 1),
                            speech_tokens=sequence_tokens.unsqueeze(0).repeat(count, 1),
                            prompt_lens=torch.full((count,), prefix_len, device=device, dtype=torch.long),
                            durations=torch.full((count,), total_duration, device=device, dtype=torch.long),
                            spk_embedding=speaker[start:end],
                            steps=rollout_steps,
                            noise_scale=args.noise_scale,
                            cfg_strength=args.cfg_strength,
                            timestep_mapping=args.timestep_mapping,
                            timestep_power=args.timestep_power,
                            sway_sampling_coef=args.sway_sampling_coef,
                            shift=args.shift,
                            sde_window_start=window_start,
                            sde_window_size=window_size,
                        )
                    )
                rollout = concatenate_rollouts(rollout_parts)
                old_log_probs = rollout_log_probs(rollout)

            generated_paths = save_rollout_audio(
                rollout,
                task=task,
                output_dir=output_dir,
                update=update,
                sample_rate=int(policy.target_sample_rate),
                keep=args.keep_rollouts,
                output_gain=1.0 / prompt_gain,
                rank=rank,
            )
            reward_target_text = (
                task.get("prompt_text", "")
                if args.data_mode == "self_prefix"
                else task.get("target_text", "")
            )
            reward_result = rewarder.score(
                generated_paths=generated_paths,
                prompt_paths=[reference_audio_path] * local_samples,
                target_texts=[reward_target_text] * local_samples,
                languages=[task.get("language")] * local_samples,
            )
            local_metrics = torch.stack(
                [
                    reward_result["wer_reward"].to(device),
                    reward_result["sim_reward"].to(device),
                    reward_result["mos_reward"].to(device),
                    reward_result["wer"].to(device),
                    reward_result["sim"].to(device),
                    reward_result["mos"].to(device),
                ],
                dim=1,
            )
            records.append(
                {
                    "data_index": data_index,
                    "task": task,
                    "rollout": rollout,
                    "old_log_probs": old_log_probs,
                    "speaker": speaker,
                    "microbatch_size": microbatch_size,
                    "generated_paths": generated_paths,
                    "reward_result": reward_result,
                    "local_metrics": local_metrics,
                    "anchor_audio": anchor_audio,
                    "anchor_tokens": anchor_tokens,
                    "anchor_speaker": speaker_base,
                    "rollout_steps": rollout_steps,
                    "sde_window_start": window_start,
                }
            )

        local_metrics = torch.cat([record["local_metrics"] for record in records], dim=0)
        gathered_metrics = gather_equal_tensors(local_metrics, world_size)
        group_metrics = gathered_metrics.view(world_size, len(records), local_samples, 6)
        group_metrics = group_metrics.permute(1, 0, 2, 3).contiguous()
        reward_components = group_metrics[..., :3].reshape(-1, 3)
        reward_weights = torch.tensor(
            [args.wer_weight, args.sim_weight, args.mos_weight],
            device=device,
            dtype=torch.float32,
        )
        fused_rewards, reward_component_means, reward_component_stds = normalize_and_fuse_rewards(
            reward_components,
            reward_weights,
            args.reward_std_eps,
        )
        fused_by_group = fused_rewards.view(len(records), world_size, local_samples)
        for group_index, record in enumerate(records):
            group_global = group_metrics[group_index]
            global_rewards = fused_by_group[group_index].reshape(-1)
            global_wers, global_sims, global_mos = group_global[..., 3:].reshape(-1, 3).unbind(dim=1)
            rewards = fused_by_group[group_index, rank]
            record["reward_result"]["reward"] = rewards.detach().cpu()
            advantages = (rewards - global_rewards.mean()) / global_rewards.std(unbiased=False).clamp_min(1e-4)
            if args.adv_clip_max > 0:
                advantages = advantages.clamp(-args.adv_clip_max, args.adv_clip_max)
            record["advantages"] = advantages
            record["global_rewards"] = global_rewards
            record["global_wers"] = global_wers
            record["global_sims"] = global_sims
            record["global_mos"] = global_mos
            save_rollout_metrics(
                generated_paths=record["generated_paths"],
                reward_result=record["reward_result"],
                advantages=advantages,
                global_rewards=global_rewards,
                global_wers=global_wers,
                global_sims=global_sims,
                global_mos=global_mos,
                reward_component_stds=reward_component_stds,
                task=record["task"],
                output_dir=output_dir,
                update=update,
                rank=rank,
                world_size=world_size,
            )

        policy.train()
        last_loss = torch.zeros((), device=device)
        last_policy_loss = torch.zeros((), device=device)
        last_kl = torch.zeros((), device=device)
        last_ratio_mean = torch.ones((), device=device)
        last_ratio_std = torch.zeros((), device=device)
        last_clipfrac = torch.zeros((), device=device)
        last_approx_kl = torch.zeros((), device=device)
        last_grad_norm = torch.zeros((), device=device)
        last_anchor_loss = torch.zeros((), device=device)
        last_anchor_flow = torch.zeros((), device=device)
        last_anchor_mel = torch.zeros((), device=device)
        last_anchor_repa = torch.zeros((), device=device)
        last_anchor_stft = torch.zeros((), device=device)
        for record in records:
            active_steps = record["rollout"].train_step_indices
            keep_steps = max(1, round(len(active_steps) * args.train_timestep_fraction))
            record["train_step_indices"] = active_steps[:keep_steps]
        num_steps = max(len(record["train_step_indices"]) for record in records)
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            policy_loss_total = torch.zeros((), device=device)
            kl_loss_total = torch.zeros((), device=device)
            ratio_mean_total = torch.zeros((), device=device)
            ratio_sq_total = torch.zeros((), device=device)
            clipfrac_total = torch.zeros((), device=device)
            approx_kl_total = torch.zeros((), device=device)
            for step_position in range(num_steps):
                for record in records:
                    if step_position >= len(record["train_step_indices"]):
                        continue
                    step_idx = record["train_step_indices"][step_position]
                    microbatch_size = record["microbatch_size"]
                    local_count = local_samples
                    for start in range(0, local_count, microbatch_size):
                        end = min(start + microbatch_size, local_count)
                        batch_rollout = slice_rollout(record["rollout"], start, end)
                        batch_speaker = record["speaker"][start:end]
                        batch_advantages = record["advantages"][start:end]
                        batch_weight = (end - start) / (local_count * len(records))
                        reference_mean = None
                        if args.kl_coef > 0.0:
                            with torch.no_grad(), autocast_context(device, compute_dtype):
                                if using_lora:
                                    with lora_disabled(policy):
                                        _, reference_mean = replay_log_prob_step(
                                            policy, batch_rollout, step_idx,
                                            spk_embedding=batch_speaker,
                                            cfg_strength=args.cfg_strength, return_mean=True,
                                        )
                                else:
                                    _, reference_mean = replay_log_prob_step(
                                        reference, batch_rollout, step_idx,
                                        spk_embedding=batch_speaker,
                                        cfg_strength=args.cfg_strength, return_mean=True,
                                    )
                        with autocast_context(device, compute_dtype):
                            new_log_prob, new_mean = replay_log_prob_step(
                                policy, batch_rollout, step_idx,
                                spk_embedding=batch_speaker,
                                cfg_strength=args.cfg_strength,
                                return_mean=args.kl_coef > 0.0,
                            )
                            old_log_prob = record["old_log_probs"][step_idx, start:end]
                            ratio = torch.exp((new_log_prob - old_log_prob).clamp(-20.0, 20.0))
                            clipped_ratio = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)
                            policy_loss = -torch.minimum(
                                ratio * batch_advantages,
                                clipped_ratio * batch_advantages,
                            ).mean()
                            kl_loss = torch.zeros((), device=device)
                            if args.kl_coef > 0.0:
                                kl_loss = gaussian_mean_kl(
                                    new_mean.unsqueeze(0), reference_mean.unsqueeze(0),
                                    batch_rollout.stds[step_idx : step_idx + 1],
                                    batch_rollout.action_mask,
                                ).mean()
                            loss = policy_loss + args.kl_coef * kl_loss
                        (loss * batch_weight / num_steps).backward()
                        loss_total += loss.detach() * batch_weight
                        policy_loss_total += policy_loss.detach() * batch_weight
                        kl_loss_total += kl_loss.detach() * batch_weight
                        ratio_mean_total += ratio.detach().mean() * batch_weight
                        ratio_sq_total += ratio.detach().square().mean() * batch_weight
                        clipfrac_total += (torch.abs(ratio.detach() - 1.0) > args.clip_eps).float().mean() * batch_weight
                        approx_kl_total += 0.5 * (new_log_prob.detach() - old_log_prob).square().mean() * batch_weight
            anchor_loss_total = torch.zeros((), device=device)
            anchor_parts = {
                "flow_loss": torch.zeros((), device=device),
                "aux_mel_loss": torch.zeros((), device=device),
                "repa_loss": torch.zeros((), device=device),
                "stft_loss": torch.zeros((), device=device),
            }
            if args.sft_anchor_coef > 0:
                for record in records:
                    anchor_audio = record["anchor_audio"].unsqueeze(0)
                    anchor_lens = torch.tensor(
                        [anchor_audio.shape[1]], device=device, dtype=torch.long
                    )
                    teacher_features = teacher_mask = None
                    if repa_teacher is not None:
                        with torch.no_grad():
                            teacher_features, teacher_mask = repa_teacher(anchor_audio, anchor_lens)
                    with autocast_context(device, compute_dtype):
                        anchor_loss, _, _, loss_parts = policy(
                            anchor_audio,
                            None,
                            speech_tokens=record["anchor_tokens"].unsqueeze(0),
                            spk_embedding=record["anchor_speaker"].unsqueeze(0),
                            lens=anchor_lens,
                            repa_teacher_features=teacher_features,
                            repa_teacher_mask=teacher_mask,
                            global_update=update,
                        )
                    (args.sft_anchor_coef * anchor_loss / len(records)).backward()
                    anchor_loss_total += anchor_loss.detach() / len(records)
                    for key in anchor_parts:
                        anchor_parts[key] += loss_parts[key].detach() / len(records)
            average_gradients(policy, world_size)
            last_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm).detach()
            optimizer.step()
            scheduler.step()
            last_loss = loss_total / num_steps + args.sft_anchor_coef * anchor_loss_total
            last_policy_loss = policy_loss_total / num_steps
            last_kl = kl_loss_total / num_steps
            last_ratio_mean = ratio_mean_total / num_steps
            last_ratio_std = (ratio_sq_total / num_steps - last_ratio_mean.square()).clamp_min(0.0).sqrt()
            last_clipfrac = clipfrac_total / num_steps
            last_approx_kl = approx_kl_total / num_steps
            last_anchor_loss = anchor_loss_total
            last_anchor_flow = anchor_parts["flow_loss"]
            last_anchor_mel = anchor_parts["aux_mel_loss"]
            last_anchor_repa = anchor_parts["repa_loss"]
            last_anchor_stft = anchor_parts["stft_loss"]

        if not args.keep_rollouts:
            for record in records:
                for path in record["generated_paths"]:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
        all_raw_metrics = group_metrics[..., 3:].reshape(-1, 3)
        return {
            "data_indices": data_indices,
            "utts": [record["task"]["utt"] for record in records],
            "prompt_fraction": [float(record["task"]["prompt_fraction"]) for record in records],
            "prompt_duration": [int(record["task"]["prompt_duration"]) for record in records],
            "total_duration": [int(record["task"]["total_duration"]) for record in records],
            "microbatch_sizes": [record["microbatch_size"] for record in records],
            "rollout_steps": [record["rollout_steps"] for record in records],
            "sde_window_starts": [record["sde_window_start"] for record in records],
            "loss": last_loss,
            "policy_loss": last_policy_loss,
            "kl": last_kl,
            "ratio_mean": last_ratio_mean,
            "ratio_std": last_ratio_std,
            "clipfrac": last_clipfrac,
            "approx_kl": last_approx_kl,
            "grad_norm": last_grad_norm,
            "learning_rate": torch.tensor(optimizer.param_groups[0]["lr"], device=device),
            "anchor_loss": last_anchor_loss,
            "anchor_flow_loss": last_anchor_flow,
            "anchor_mel_loss": last_anchor_mel,
            "anchor_repa_loss": last_anchor_repa,
            "anchor_stft_loss": last_anchor_stft,
            "reward": fused_rewards.mean(),
            "wer": all_raw_metrics[:, 0].mean(),
            "sim": all_raw_metrics[:, 1].mean(),
            "mos": all_raw_metrics[:, 2].mean(),
            "reward_std": fused_rewards.std(unbiased=False),
            "wer_std": all_raw_metrics[:, 0].std(unbiased=False),
            "sim_std": all_raw_metrics[:, 1].std(unbiased=False),
            "mos_std": all_raw_metrics[:, 2].std(unbiased=False),
            "wer_reward_batch_std": reward_component_stds[0],
            "sim_reward_batch_std": reward_component_stds[1],
            "mos_reward_batch_std": reward_component_stds[2],
        }

    try:
        update_iter = tqdm(
            range(start_update + 1, args.max_updates + 1),
            desc="GRPO",
            disable=not is_main,
        )
        for update in update_iter:
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            if args.prompts_per_update >= 1:
                batch_start = (update - 1) * args.prompts_per_update
                batch_indices = []
                for absolute_position in range(
                    batch_start, batch_start + args.prompts_per_update
                ):
                    data_epoch, data_position = divmod(absolute_position, len(dataset))
                    if data_epoch != shuffle_epoch:
                        order_generator = torch.Generator(device="cpu")
                        order_generator.manual_seed(args.seed + data_epoch)
                        shuffled_indices = torch.randperm(
                            len(dataset), generator=order_generator
                        ).tolist()
                        shuffle_epoch = data_epoch
                    batch_indices.append(shuffled_indices[data_position])
                batch_log = run_multi_prompt_update(update, batch_indices)
                if str(device).startswith("cuda"):
                    peak_memory = torch.tensor(
                        [
                            torch.cuda.max_memory_allocated(device),
                            torch.cuda.max_memory_reserved(device),
                        ],
                        device=device,
                        dtype=torch.float64,
                    )
                    if world_size > 1:
                        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
                    peak_allocated_gib = float(peak_memory[0].item() / 1024**3)
                    peak_reserved_gib = float(peak_memory[1].item() / 1024**3)
                else:
                    peak_allocated_gib = 0.0
                    peak_reserved_gib = 0.0
                log_values = {
                    "update": update,
                    "rank": rank,
                    "data_indices": batch_log["data_indices"],
                    "utts": batch_log["utts"],
                    "prompt_fraction": batch_log["prompt_fraction"],
                    "prompt_duration": batch_log["prompt_duration"],
                    "total_duration": batch_log["total_duration"],
                    "microbatch_sizes": batch_log["microbatch_sizes"],
                    "rollout_steps": batch_log["rollout_steps"],
                    "sde_window_starts": batch_log["sde_window_starts"],
                    "prompts_per_update": args.prompts_per_update,
                    "loss": float(batch_log["loss"].item()),
                    "policy_loss": float(batch_log["policy_loss"].item()),
                    "kl": float(batch_log["kl"].item()),
                    "reward": float(batch_log["reward"].item()),
                    "wer": float(batch_log["wer"].item()),
                    "sim": float(batch_log["sim"].item()),
                    "mos": float(batch_log["mos"].item()),
                    "reward_std": float(batch_log["reward_std"].item()),
                    "wer_std": float(batch_log["wer_std"].item()),
                    "sim_std": float(batch_log["sim_std"].item()),
                    "mos_std": float(batch_log["mos_std"].item()),
                    "wer_reward_batch_std": float(batch_log["wer_reward_batch_std"].item()),
                    "sim_reward_batch_std": float(batch_log["sim_reward_batch_std"].item()),
                    "mos_reward_batch_std": float(batch_log["mos_reward_batch_std"].item()),
                    "anchor_loss": float(batch_log["anchor_loss"].item()),
                    "anchor_flow_loss": float(batch_log["anchor_flow_loss"].item()),
                    "anchor_mel_loss": float(batch_log["anchor_mel_loss"].item()),
                    "anchor_repa_loss": float(batch_log["anchor_repa_loss"].item()),
                    "anchor_stft_loss": float(batch_log["anchor_stft_loss"].item()),
                    "ratio_mean": float(batch_log["ratio_mean"].item()),
                    "ratio_std": float(batch_log["ratio_std"].item()),
                    "clipfrac": float(batch_log["clipfrac"].item()),
                    "approx_kl": float(batch_log["approx_kl"].item()),
                    "grad_norm": float(batch_log["grad_norm"].item()),
                    "learning_rate": float(batch_log["learning_rate"].item()),
                    "peak_allocated_gib": peak_allocated_gib,
                    "peak_reserved_gib": peak_reserved_gib,
                }
                if is_main:
                    print(json.dumps(log_values, ensure_ascii=False), flush=True)
                if is_main and update % args.save_per_updates == 0:
                    save_checkpoint(
                        output_dir / f"model_{update}.pt",
                        model=policy,
                        optimizer=optimizer,
                        update=update,
                        args=args,
                    )
                    save_checkpoint(
                        output_dir / "model_last.pt",
                        model=policy,
                        optimizer=optimizer,
                        update=update,
                        args=args,
                    )
                continue
            data_epoch, data_position = divmod(update - 1, len(dataset))
            if data_epoch != shuffle_epoch:
                order_generator = torch.Generator(device="cpu")
                order_generator.manual_seed(args.seed + data_epoch)
                shuffled_indices = torch.randperm(
                    len(dataset), generator=order_generator
                ).tolist()
                shuffle_epoch = data_epoch
            data_index = shuffled_indices[data_position]
            task = dataset[data_index]

            if args.data_mode == "self_prefix":
                total_duration = int(task["duration"])
                prefix_len, sampled_fraction = sample_prefix_length(
                    total_duration,
                    frame_len=policy.wav_frame_len,
                    min_fraction=args.prompt_fraction_min,
                    max_fraction=args.prompt_fraction_max,
                    seed=args.seed + update * 1_000_003 + data_index,
                )
                raw_condition = task["audio"]
                sequence_tokens = task["speech_token"]
                reference_audio_path = task["audio_path"]
                task = {
                    **task,
                    "prompt_wav_path": reference_audio_path,
                    "prompt_wav": raw_condition,
                    "prompt_duration": prefix_len,
                    "target_duration": total_duration - prefix_len,
                    "total_duration": total_duration,
                    "prompt_fraction": sampled_fraction,
                }
            else:
                prefix_len = int(task["prompt_duration"])
                total_duration = prefix_len + int(task["target_duration"])
                raw_condition = task["prompt_wav"]
                sequence_tokens = torch.cat([task["prompt_token"], task["target_token"]])
                reference_audio_path = task["prompt_wav_path"]
                task = {
                    **task,
                    "total_duration": total_duration,
                    "prompt_fraction": prefix_len / total_duration,
                }

            prompt_rms = torch.sqrt(torch.mean(raw_condition.square())).item()
            prompt_gain = (
                args.target_rms / max(prompt_rms, 1e-8)
                if prompt_rms < args.target_rms
                else 1.0
            )
            condition_audio = (raw_condition * prompt_gain).to(device)
            sequence_tokens = sequence_tokens.to(device)
            with torch.no_grad():
                speaker_base = speaker_extractor.extract(raw_condition, device=device)
            speaker = speaker_base.unsqueeze(0).repeat(local_samples, 1)
            microbatch_size = rollout_microbatch_size(
                local_samples=local_samples,
                duration=total_duration,
                frame_len=policy.wav_frame_len,
                frame_budget=args.batch_size_frames,
            )
            if local_samples >= 3 and microbatch_size < 3:
                required_frames = 3 * max(
                    1, (total_duration + policy.wav_frame_len - 1) // policy.wav_frame_len
                )
                raise ValueError(
                    "Rollout microbatch is smaller than 3 samples. "
                    f"Increase --batch-size-frames to at least {required_frames} "
                    f"(current {args.batch_size_frames})."
                )

            policy.eval()
            with torch.no_grad(), autocast_context(device, compute_dtype):
                rollout_parts = []
                for start in range(0, local_samples, microbatch_size):
                    end = min(start + microbatch_size, local_samples)
                    count = end - start
                    rollout_parts.append(
                        rollout_sde(
                            policy,
                            cond=condition_audio.unsqueeze(0).repeat(count, 1),
                            speech_tokens=sequence_tokens.unsqueeze(0).repeat(count, 1),
                            prompt_lens=torch.full(
                                (count,), prefix_len, device=device, dtype=torch.long
                            ),
                            durations=torch.full(
                                (count,), total_duration, device=device, dtype=torch.long
                            ),
                            spk_embedding=speaker[start:end],
                            steps=args.steps,
                            noise_scale=args.noise_scale,
                            cfg_strength=args.cfg_strength,
                            timestep_mapping=args.timestep_mapping,
                            timestep_power=args.timestep_power,
                            sway_sampling_coef=args.sway_sampling_coef,
                            shift=args.shift,
                        )
                    )
                rollout = concatenate_rollouts(rollout_parts)
                # The old policy log-probability must use the exact transition
                # means that produced each sampled action. Re-running an
                # approximate replay path here breaks the on-policy ratio.
                old_log_probs = rollout_log_probs(rollout)

            generated_paths = save_rollout_audio(
                rollout,
                task=task,
                output_dir=output_dir,
                update=update,
                sample_rate=int(policy.target_sample_rate),
                keep=args.keep_rollouts,
                output_gain=1.0 / prompt_gain,
                rank=rank,
            )
            # self_prefix predicts the latter part of prompt_wav, so use the
            # prompt transcript as the WER reference for every generated wav.
            reward_target_text = (
                task.get("prompt_text", "")
                if args.data_mode == "self_prefix"
                else task.get("target_text", "")
            )
            reward_result = rewarder.score(
                generated_paths=generated_paths,
                prompt_paths=[reference_audio_path] * local_samples,
                target_texts=[reward_target_text] * local_samples,
                languages=[task.get("language")] * local_samples,
            )
            rewards = reward_result["reward"].to(device)
            local_metrics = torch.stack(
                [
                    rewards,
                    reward_result["wer"].to(device),
                    reward_result["sim"].to(device),
                ],
                dim=1,
            )
            global_rewards, global_wers, global_sims = gather_equal_tensors(
                local_metrics, world_size
            ).unbind(dim=1)
            advantages = (
                (rewards - global_rewards.mean())
                / global_rewards.std(unbiased=False).clamp_min(1e-4)
            )
            if args.adv_clip_max > 0:
                advantages = advantages.clamp(-args.adv_clip_max, args.adv_clip_max)
            save_rollout_metrics(
                generated_paths=generated_paths,
                reward_result=reward_result,
                advantages=advantages,
                global_rewards=global_rewards,
                global_wers=global_wers,
                global_sims=global_sims,
                task=task,
                output_dir=output_dir,
                update=update,
                rank=rank,
                world_size=world_size,
            )

            policy.train()
            last_loss = None
            last_policy_loss = torch.zeros((), device=device)
            last_kl = torch.zeros((), device=device)
            last_ratio_mean = torch.ones((), device=device)
            last_ratio_std = torch.zeros((), device=device)
            last_clipfrac = torch.zeros((), device=device)
            last_approx_kl = torch.zeros((), device=device)
            last_grad_norm = torch.zeros((), device=device)
            for _ in range(args.ppo_epochs):
                optimizer.zero_grad(set_to_none=True)
                loss_total = torch.zeros((), device=device)
                policy_loss_total = torch.zeros((), device=device)
                kl_loss_total = torch.zeros((), device=device)
                ratio_mean_total = torch.zeros((), device=device)
                ratio_sq_total = torch.zeros((), device=device)
                clipfrac_total = torch.zeros((), device=device)
                approx_kl_total = torch.zeros((), device=device)
                num_steps = max(
                    1,
                    int(len(rollout.times) * args.train_timestep_fraction),
                )
                for step_idx in range(num_steps):
                    for start in range(0, local_samples, microbatch_size):
                        end = min(start + microbatch_size, local_samples)
                        batch_rollout = slice_rollout(rollout, start, end)
                        batch_speaker = speaker[start:end]
                        batch_advantages = advantages[start:end]
                        batch_weight = (end - start) / local_samples
                        reference_mean = None
                        if args.kl_coef > 0.0:
                            with torch.no_grad(), autocast_context(device, compute_dtype):
                                if using_lora:
                                    with lora_disabled(policy):
                                        _, reference_mean = replay_log_prob_step(
                                            policy,
                                            batch_rollout,
                                            step_idx,
                                            spk_embedding=batch_speaker,
                                            cfg_strength=args.cfg_strength,
                                            return_mean=True,
                                        )
                                else:
                                    _, reference_mean = replay_log_prob_step(
                                        reference,
                                        batch_rollout,
                                        step_idx,
                                        spk_embedding=batch_speaker,
                                        cfg_strength=args.cfg_strength,
                                        return_mean=True,
                                    )
                        with autocast_context(device, compute_dtype):
                            new_log_prob, new_mean = replay_log_prob_step(
                                policy,
                                batch_rollout,
                                step_idx,
                                spk_embedding=batch_speaker,
                                cfg_strength=args.cfg_strength,
                                return_mean=args.kl_coef > 0.0,
                            )
                            old_log_prob = old_log_probs[step_idx, start:end]
                            ratio = torch.exp((new_log_prob - old_log_prob).clamp(-20.0, 20.0))
                            clipped_ratio = ratio.clamp(
                                1.0 - args.clip_eps,
                                1.0 + args.clip_eps,
                            )
                            policy_loss = -torch.minimum(
                                ratio * batch_advantages,
                                clipped_ratio * batch_advantages,
                            ).mean()
                            kl_loss = torch.zeros((), device=device)
                            if args.kl_coef > 0.0:
                                kl_loss = gaussian_mean_kl(
                                    new_mean.unsqueeze(0),
                                    reference_mean.unsqueeze(0),
                                    batch_rollout.stds[step_idx : step_idx + 1],
                                    batch_rollout.action_mask,
                                ).mean()
                            loss = policy_loss + args.kl_coef * kl_loss
                        (loss * batch_weight / num_steps).backward()
                        metric_weight = batch_weight
                        loss_total += loss.detach() * metric_weight
                        policy_loss_total += policy_loss.detach() * metric_weight
                        kl_loss_total += kl_loss.detach() * metric_weight
                        ratio_mean_total += ratio.detach().mean() * metric_weight
                        ratio_sq_total += ratio.detach().square().mean() * metric_weight
                        clipfrac_total += (
                            (torch.abs(ratio.detach() - 1.0) > args.clip_eps)
                            .float()
                            .mean()
                            * metric_weight
                        )
                        approx_kl_total += (
                            0.5
                            * (new_log_prob.detach() - old_log_prob).square().mean()
                            * metric_weight
                        )
                average_gradients(policy, world_size)
                last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.max_grad_norm,
                ).detach()
                optimizer.step()
                last_loss = loss_total / num_steps
                last_policy_loss = policy_loss_total / num_steps
                last_kl = kl_loss_total / num_steps
                last_ratio_mean = ratio_mean_total / num_steps
                last_ratio_std = (
                    (ratio_sq_total / num_steps - last_ratio_mean.square()).clamp_min(0.0).sqrt()
                )
                last_clipfrac = clipfrac_total / num_steps
                last_approx_kl = approx_kl_total / num_steps

            if not args.keep_rollouts:
                for path in generated_paths:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

            peak_allocated_gib = 0.0
            peak_reserved_gib = 0.0
            if str(device).startswith("cuda"):
                peak_memory = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(device),
                        torch.cuda.max_memory_reserved(device),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if world_size > 1:
                    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
                peak_allocated_gib = float(peak_memory[0].item() / 1024**3)
                peak_reserved_gib = float(peak_memory[1].item() / 1024**3)

            log_values = {
                "update": update,
                "rank": rank,
                "data_index": data_index,
                "utt": task["utt"],
                "prompt_fraction": float(task["prompt_fraction"]),
                "prompt_duration": int(task["prompt_duration"]),
                "total_duration": int(task["total_duration"]),
                "microbatch_size": microbatch_size,
                "loss": float(last_loss.item()),
                "policy_loss": float(last_policy_loss.item()),
                "kl": float(last_kl.item()),
                "reward": float(global_rewards.mean().item()),
                "wer": float(global_wers.mean().item()),
                "sim": float(global_sims.mean().item()),
                "reward_std": float(global_rewards.std(unbiased=False).item()),
                "wer_std": float(global_wers.std(unbiased=False).item()),
                "sim_std": float(global_sims.std(unbiased=False).item()),
                "ratio_mean": float(last_ratio_mean.item()),
                "ratio_std": float(last_ratio_std.item()),
                "clipfrac": float(last_clipfrac.item()),
                "approx_kl": float(last_approx_kl.item()),
                "grad_norm": float(last_grad_norm.item()),
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
            }
            if world_size > 1:
                stats = torch.tensor(
                    [
                        log_values["loss"],
                        log_values["policy_loss"],
                        log_values["kl"],
                        log_values["reward"],
                        log_values["wer"],
                        log_values["sim"],
                        log_values["reward_std"],
                        log_values["wer_std"],
                        log_values["sim_std"],
                        log_values["ratio_mean"],
                        log_values["ratio_std"],
                        log_values["clipfrac"],
                        log_values["approx_kl"],
                        log_values["grad_norm"],
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats.div_(world_size)
                log_values.update(
                    {
                        "loss": float(stats[0].item()),
                        "policy_loss": float(stats[1].item()),
                        "kl": float(stats[2].item()),
                        "reward": float(stats[3].item()),
                        "wer": float(stats[4].item()),
                        "sim": float(stats[5].item()),
                        "reward_std": float(stats[6].item()),
                        "wer_std": float(stats[7].item()),
                        "sim_std": float(stats[8].item()),
                        "ratio_mean": float(stats[9].item()),
                        "ratio_std": float(stats[10].item()),
                        "clipfrac": float(stats[11].item()),
                        "approx_kl": float(stats[12].item()),
                        "grad_norm": float(stats[13].item()),
                    }
                )
            if is_main:
                print(
                    json.dumps(log_values, ensure_ascii=False),
                    flush=True,
                )

            if is_main and update % args.save_per_updates == 0:
                save_checkpoint(
                    output_dir / f"model_{update}.pt",
                    model=policy,
                    optimizer=optimizer,
                    update=update,
                    args=args,
                )
                save_checkpoint(
                    output_dir / "model_last.pt",
                    model=policy,
                    optimizer=optimizer,
                    update=update,
                    args=args,
                )
    finally:
        rewarder.close()

    if is_main:
        save_checkpoint(
            output_dir / "model_last.pt",
            model=policy,
            optimizer=optimizer,
            update=args.max_updates,
            args=args,
        )
    if world_size > 1 and dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
