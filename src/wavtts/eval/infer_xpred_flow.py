import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio
from accelerate import Accelerator
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from wavtts.model import CFM
from wavtts.model.lora import inject_lora
from wavtts.model.speaker_embedding import SpeakerEmbeddingExtractor
from wavtts.model.utils import get_tokenizer


def parse_ckpt_steps(raw_steps: list[str] | None) -> list[str] | None:
    if raw_steps is None:
        return None

    steps = []
    for raw_step in raw_steps:
        for step in raw_step.replace("，", ",").split(","):
            step = step.strip()
            if step:
                steps.append(step)

    if not steps:
        raise ValueError("--ckpt-steps was provided, but no valid steps were parsed.")

    return steps


def parse_list_values(raw_values: list[str] | None, cast, option_name: str) -> list:
    if raw_values is None:
        return []

    values = []
    for raw_value in raw_values:
        normalized = str(raw_value).replace("，", ",").replace(",", " ")
        for value in normalized.split():
            value = value.strip()
            if value:
                try:
                    values.append(cast(value))
                except ValueError as exc:
                    raise ValueError(f"Invalid value for {option_name}: {value!r}") from exc

    if not values:
        raise ValueError(f"{option_name} was provided, but no valid values were parsed.")

    return values


def tag_num(value) -> str:
    text = str(value)
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text[:-2] if text.endswith("p0") else text


def case_name(cfg_strength, shift, timestep_power, nfe_step) -> str:
    return (
        f"cfg{tag_num(cfg_strength)}_"
        f"shift{tag_num(shift)}_"
        f"power{tag_num(timestep_power)}_"
        f"nfe{nfe_step}"
    )


def resolve_ckpt_paths(path: str, ckpt_steps: list[str] | None = None) -> list[tuple[str, str]]:
    if path.startswith(("hf://", "http://", "https://")):
        if ckpt_steps:
            raise ValueError("--ckpt-steps is only supported for local checkpoint directories.")
        ckpt_path = str(cached_path(path))
        return [(Path(ckpt_path).stem, ckpt_path)]

    p = Path(path)
    if ckpt_steps:
        if not p.is_dir():
            raise ValueError("--ckpt must be a checkpoint directory when --ckpt-steps is provided.")

        ckpt_paths = []
        seen = set()
        for step in ckpt_steps:
            step_lower = step.lower()
            if step_lower in {"last", "latest"}:
                label = "last"
                ckpt = p / "model_last.pt"
            else:
                if not step.isdigit():
                    raise ValueError(f"Checkpoint step must be numeric or 'last', got {step!r}.")
                label = str(int(step))
                ckpt = p / f"model_{label}.pt"

            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint for step {step!r} not found: {ckpt}")
            if str(ckpt) in seen:
                continue
            seen.add(str(ckpt))
            ckpt_paths.append((label, str(ckpt)))

        return ckpt_paths

    if p.is_dir():
        last = p / "model_last.pt"
        if last.exists():
            return [("last", str(last))]
        numbered = sorted(
            p.glob("model_*.pt"),
            key=lambda x: int("".join(ch for ch in x.stem if ch.isdigit()) or -1),
        )
        if numbered:
            return [(numbered[-1].stem.replace("model_", ""), str(numbered[-1]))]
        raise FileNotFoundError(f"No model_last.pt or model_*.pt found in {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return [(p.stem.replace("model_", ""), str(p))]


def infer_model_cfg(ckpt_path: str, model_cfg: str | None):
    if model_cfg:
        return model_cfg
    ckpt = Path(ckpt_path)
    candidates = []
    if ckpt.parent.name == "ckpts":
        candidates.append(ckpt.parent.parent / ".hydra" / "config.yaml")
    candidates.append(ckpt.parent / ".hydra" / "config.yaml")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise ValueError(
        "--model-cfg was not provided and no .hydra/config.yaml could be inferred from the checkpoint path."
    )


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if is_pair_row(obj):
                for key in ("prompt_text", "prompt_token", "target_token"):
                    if key not in obj:
                        raise KeyError(f"Missing key {key!r} in {path}")
                if "prompt_wavpath" not in obj and "prompt_wav" not in obj:
                    raise KeyError(
                        f"Missing 'prompt_wavpath'/'prompt_wav' in {path}"
                    )
            else:
                for key in ("utt", "text", "wavpath", "token"):
                    if key not in obj:
                        raise KeyError(f"Missing key {key!r} in {path}")
            rows.append(obj)
    return rows


def shard_rows(rows: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards}), got {shard_index}.")
    if num_shards == 1:
        return rows
    return rows[shard_index::num_shards]


def is_pair_row(row: dict) -> bool:
    return (
        "prompt_wavpath" in row
        or "prompt_wav" in row
        or "prompt_token" in row
        or "target_token" in row
    )


def row_utt(row: dict) -> str:
    if "utt" in row:
        return row["utt"]
    prompt_utt = row.get("prompt_utt", "prompt")
    target_utt = row.get("target_utt", "target")
    return f"{prompt_utt}__to__{target_utt}"


def row_text(row: dict) -> str:
    return row.get("target_text", row.get("text", ""))


def row_prompt_wavpath(row: dict) -> str:
    if "prompt_wavpath" in row:
        value = row["prompt_wavpath"]
    elif "prompt_wav" in row:
        value = row["prompt_wav"]
    else:
        value = row["wavpath"]

    path = Path(str(value)).expanduser()
    if path.exists():
        return str(path)
    # The training/evaluation JSONL can be generated on another mount
    # (for example /home/work_nfs20), while the data is mirrored under this
    # project. Resolve the shared `data/...` suffix locally when possible.
    if path.is_absolute() and "data" in path.parts:
        data_index = path.parts.index("data")
        project_root = Path(__file__).resolve().parents[3]
        local_path = project_root.joinpath(*path.parts[data_index:])
        if local_path.exists():
            return str(local_path)
    return str(path)


def make_batches(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def align_to_frame_len(num_samples: int, wav_frame_len: int) -> int:
    return ((int(num_samples) + wav_frame_len - 1) // wav_frame_len) * wav_frame_len


def duration_from_token_count(token_count: int | torch.Tensor, token_rate: float, sample_rate: int, wav_frame_len: int) -> int:
    if isinstance(token_count, torch.Tensor):
        token_count = int(token_count.item())
    duration = round(float(token_count) / float(token_rate) * sample_rate)
    return align_to_frame_len(duration, wav_frame_len)


def prompt_history_lengths(
    *,
    requested_seconds: float,
    token_count: int,
    available_samples: int,
    token_rate: float,
    sample_rate: int,
    wav_frame_len: int,
) -> tuple[int, int]:
    """Return aligned (prompt_samples, prompt_tokens) for Cosy-style history conditioning."""
    if requested_seconds <= 0 or token_count <= 1:
        return 0, 0

    requested_tokens = max(1, int(round(requested_seconds * token_rate)))
    prompt_tokens = min(requested_tokens, token_count - 1)

    available_samples = (int(available_samples) // wav_frame_len) * wav_frame_len
    prompt_samples = duration_from_token_count(prompt_tokens, token_rate, sample_rate, wav_frame_len)
    prompt_samples = min(prompt_samples, available_samples)
    if prompt_samples <= 0:
        return 0, 0

    # Keep prompt wav and prompt token on the same time axis. This mirrors CosyVoice2's
    # prompt_feat / prompt_token pair, adapted to direct token-to-waveform modeling.
    prompt_tokens = min(prompt_tokens, max(0, int(round(prompt_samples / sample_rate * token_rate))))
    if prompt_tokens <= 0:
        return 0, 0
    prompt_samples = duration_from_token_count(prompt_tokens, token_rate, sample_rate, wav_frame_len)
    prompt_samples = min(prompt_samples, available_samples)
    return prompt_samples, prompt_tokens


def full_prompt_lengths(
    *,
    token_count: int,
    available_samples: int,
    token_rate: float,
    sample_rate: int,
    wav_frame_len: int,
) -> tuple[int, int]:
    if token_count <= 0:
        return 0, 0

    available_samples = (int(available_samples) // wav_frame_len) * wav_frame_len
    prompt_samples = min(
        duration_from_token_count(token_count, token_rate, sample_rate, wav_frame_len),
        available_samples,
    )
    if prompt_samples <= 0:
        return 0, 0
    prompt_tokens = min(token_count, max(1, int(round(prompt_samples / sample_rate * token_rate))))
    prompt_samples = duration_from_token_count(prompt_tokens, token_rate, sample_rate, wav_frame_len)
    prompt_samples = min(prompt_samples, available_samples)
    return prompt_samples, prompt_tokens


def load_token_model(model_cfg, ckpt_path, device, load_dtype_name):
    model_cls = get_class(f"wavtts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    waveform_kwargs = model_cfg.model.waveform
    cfm_kwargs = getattr(model_cfg.model, "cfm", {}) or {}
    if not isinstance(cfm_kwargs, dict):
        cfm_kwargs = OmegaConf.to_container(cfm_kwargs, resolve=True)
    cfm_kwargs.pop("repa_teacher_model", None)
    cfm_kwargs.pop("repa_teacher_layer", None)

    tokenizer = model_cfg.model.tokenizer
    if tokenizer == "custom":
        tokenizer_path = model_cfg.model.tokenizer_path
    else:
        tokenizer_path = model_cfg.datasets.name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            wav_frame_len=waveform_kwargs.wav_frame_len,
        ),
        waveform_kwargs=waveform_kwargs,
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
        **cfm_kwargs,
    ).to(device)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[load_dtype_name]
    model = model.to(dtype)
    checkpoint = load_checkpoint(ckpt_path)

    if "adapter_state_dict" in checkpoint:
        return load_grpo_checkpoint(model, checkpoint, ckpt_path, device)
    return load_sft_checkpoint(model, checkpoint, ckpt_path, device)


def load_checkpoint(ckpt_path: str) -> dict:
    try:
        return torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def _load_model_state(model, model_state_dict, ckpt_path: str):
    incompatible = model.load_state_dict(model_state_dict, strict=False)
    ignorable_unexpected_prefixes = (
        "repa_alignment_head.",
        "aux_mel_loss.",
        "stft_loss.",
    )
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(ignorable_unexpected_prefixes)
    ]
    bad_missing = [
        key
        for key in incompatible.missing_keys
        if ".spk_" not in key
        and not key.startswith("spk_")
        and not key.startswith("ema_model.spk_")
        and not key.startswith("ema_model.transformer.spk_")
    ]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"Checkpoint is incompatible: unexpected={unexpected[:20]}, "
            f"missing={bad_missing[:20]}"
        )
    if incompatible.unexpected_keys:
        print(
            f"Ignoring checkpoint-only optional keys from {ckpt_path}: "
            f"{incompatible.unexpected_keys[:8]}",
            flush=True,
        )
    return model


def _state_dict_from_checkpoint(checkpoint: dict, ckpt_path: str) -> dict:
    state = checkpoint.get("ema_model_state_dict") or checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(
            f"Checkpoint {ckpt_path} contains neither ema_model_state_dict nor model_state_dict"
        )
    return {
        key.replace("ema_model.", ""): value
        for key, value in state.items()
        if key not in {"initted", "update", "step"}
    }


def load_sft_checkpoint(model, checkpoint: dict, ckpt_path: str, device):
    _load_model_state(model, _state_dict_from_checkpoint(checkpoint, ckpt_path), ckpt_path)
    del checkpoint
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model.to(device)


def _resolve_base_checkpoint(grpo_ckpt_path: str, base_model_path: str | None) -> str:
    if not base_model_path:
        raise KeyError(
            f"GRPO checkpoint {grpo_ckpt_path} does not contain base_model_path"
        )
    base_path = Path(str(base_model_path)).expanduser()
    if not base_path.exists():
        base_path = Path(grpo_ckpt_path).resolve().parent / base_path
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base checkpoint recorded in {grpo_ckpt_path} was not found: {base_path}"
        )
    return str(base_path)


def load_grpo_checkpoint(model, checkpoint: dict, ckpt_path: str, device):
    grpo_args = checkpoint.get("grpo_args", {}) or {}
    adapter_state = checkpoint.get("adapter_state_dict") or {}
    full_state = checkpoint.get("model_state_dict")

    if full_state is not None and not adapter_state:
        _load_model_state(model, full_state, ckpt_path)
        del checkpoint
        return model.to(device)

    base_path = _resolve_base_checkpoint(
        ckpt_path,
        checkpoint.get("base_model_path"),
    )
    base_checkpoint = load_checkpoint(base_path)
    _load_model_state(
        model,
        _state_dict_from_checkpoint(base_checkpoint, base_path),
        base_path,
    )
    del base_checkpoint

    rank = int(grpo_args.get("lora_rank", 0) or 0)
    if rank <= 0:
        raise RuntimeError(
            f"GRPO checkpoint {ckpt_path} has adapter weights but lora_rank={rank}"
        )
    targets = tuple(
        item.strip()
        for item in str(
            grpo_args.get(
                "lora_targets",
                "attn.to_q,attn.to_k,attn.to_v,attn.to_out.0",
            )
        ).split(",")
        if item.strip()
    )
    stats = inject_lora(
        model,
        rank=rank,
        alpha=float(grpo_args.get("lora_alpha", 64.0)),
        dropout=float(grpo_args.get("lora_dropout", 0.0)),
        target_suffixes=targets,
    )
    incompatible = model.load_state_dict(adapter_state, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys if ".lora_" not in key
    ]
    missing = [
        key for key in incompatible.missing_keys if ".lora_" in key
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"LoRA checkpoint is incompatible: unexpected={unexpected[:20]}, "
            f"missing={missing[:20]}"
        )
    alpha = float(grpo_args.get("lora_alpha", 64.0))
    print(
        f"Loaded GRPO LoRA checkpoint rank={rank} alpha={alpha:g} "
        f"modules={stats.modules}: {ckpt_path}",
        flush=True,
    )
    del checkpoint
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model.to(device)


def load_ema_checkpoint(model, ckpt_path, device, dtype):
    # Kept as a compatibility helper for callers outside this script.
    model = model.to(dtype)
    checkpoint = load_checkpoint(ckpt_path)
    if "adapter_state_dict" in checkpoint:
        return load_grpo_checkpoint(model, checkpoint, ckpt_path, device)
    return load_sft_checkpoint(model, checkpoint, ckpt_path, device)


def infer_experiment_name(ckpt_path: str) -> str:
    ckpt = Path(ckpt_path)
    if ckpt.parent.name == "ckpts":
        return ckpt.parent.parent.name
    return ckpt.parent.name


def default_output_base(jsonl: str) -> Path:
    return Path(jsonl).parent / "infer"


def get_output_dir(base_output_dir: Path, ckpt_path: str, ckpt_label: str, use_ckpt_hierarchy: bool) -> Path:
    if use_ckpt_hierarchy:
        return base_output_dir / infer_experiment_name(ckpt_path) / ckpt_label
    return base_output_dir


def run_inference_for_ckpt(
    *,
    accelerator,
    rows,
    model_cfg,
    ckpt_path,
    ckpt_label,
    output_dir,
    device,
    args,
    target_sample_rate,
    wav_frame_len,
    autocast_dtype,
):
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "refs").mkdir(parents=True, exist_ok=True)
        print(f"Running checkpoint {ckpt_label}: {ckpt_path}")
        print(f"Output directory: {output_dir}")
    accelerator.wait_for_everyone()

    model = load_token_model(model_cfg, ckpt_path, device, args.load_dtype)
    model.eval()
    speaker_extractor = None
    if bool(getattr(model_cfg.model.arch, "use_spk_embedding", False)):
        speaker_cfg = getattr(model_cfg.model, "speaker_embedding", {}) or {}
        speaker_model_path = args.spk_embedding_model or speaker_cfg.get("model_path")
        speaker_extractor = SpeakerEmbeddingExtractor(
            model_path=speaker_model_path,
            sample_rate=target_sample_rate,
        )

    with accelerator.split_between_processes(rows) as local_rows:
        for batch_rows in tqdm(list(make_batches(local_rows, args.batch_size)), disable=not accelerator.is_local_main_process):
            utts = [row_utt(row) for row in batch_rows]
            ref_audios = []
            ref_rms_values = []
            for row in batch_rows:
                ref_audio, ref_sr = torchaudio.load(row_prompt_wavpath(row))
                if ref_sr != target_sample_rate:
                    ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, target_sample_rate)
                if ref_audio.shape[0] > 1:
                    ref_audio = ref_audio.mean(dim=0, keepdim=True)
                ref_audio = ref_audio.to(torch.float32)
                ref_audios.append(ref_audio)
                ref_rms_values.append(torch.sqrt(torch.mean(torch.square(ref_audio))).item())

            full_token_tensors = [
                torch.tensor(row["token"], dtype=torch.long, device=device).reshape(-1)
                for row in batch_rows
                if not is_pair_row(row)
            ]

            if any(is_pair_row(row) for row in batch_rows):
                if not all(is_pair_row(row) for row in batch_rows):
                    raise ValueError("Cannot mix normal rows and prompt/target pair rows in one batch.")
                prompt_tensors = []
                prompt_lengths = []
                target_durations = []
                speech_token_sequences = []
                for row, ref_audio, ref_rms in zip(batch_rows, ref_audios, ref_rms_values):
                    prompt_tokens_full = torch.tensor(row["prompt_token"], dtype=torch.long, device=device).reshape(-1)
                    target_tokens = torch.tensor(row["target_token"], dtype=torch.long, device=device).reshape(-1)
                    prompt_len, prompt_token_len = full_prompt_lengths(
                        token_count=prompt_tokens_full.numel(),
                        available_samples=ref_audio.shape[-1],
                        token_rate=args.speech_token_rate,
                        sample_rate=target_sample_rate,
                        wav_frame_len=wav_frame_len,
                    )
                    if prompt_len <= 0 or prompt_token_len <= 0:
                        raise ValueError(f"Empty prompt wav/token for row {row_utt(row)}")
                    prompt_audio_in = ref_audio
                    if ref_rms < args.target_rms:
                        prompt_audio_in = prompt_audio_in * args.target_rms / max(ref_rms, 1e-8)
                    prompt = prompt_audio_in[0, :prompt_len]
                    prompt_tokens = prompt_tokens_full[:prompt_token_len]
                    speech_token_sequences.append(torch.cat([prompt_tokens, target_tokens], dim=0))
                    prompt_tensors.append(prompt.to(device=device))
                    prompt_lengths.append(prompt_len)
                    target_duration = duration_from_token_count(
                        target_tokens.numel(),
                        args.speech_token_rate,
                        target_sample_rate,
                        wav_frame_len,
                    )
                    target_duration = align_to_frame_len(round(target_duration * args.duration_scale), wav_frame_len)
                    target_durations.append(target_duration)

                cond = pad_sequence(prompt_tensors, batch_first=True, padding_value=0.0)
                lens = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
                durations = torch.tensor(
                    [prompt_len + target_duration for prompt_len, target_duration in zip(prompt_lengths, target_durations)],
                    dtype=torch.long,
                    device=device,
                )
                no_ref_audio = False
            else:
                speech_token_sequences = full_token_tensors

            if not any(is_pair_row(row) for row in batch_rows) and args.ref_prompt_seconds > 0:
                prompt_tensors = []
                prompt_lengths = []
                target_durations = []
                speech_tokens_with_history = []
                for ref_audio, ref_rms, full_tokens in zip(ref_audios, ref_rms_values, full_token_tensors):
                    prompt_len, prompt_token_len = prompt_history_lengths(
                        requested_seconds=args.ref_prompt_seconds,
                        token_count=full_tokens.numel(),
                        available_samples=ref_audio.shape[-1],
                        token_rate=args.speech_token_rate,
                        sample_rate=target_sample_rate,
                        wav_frame_len=wav_frame_len,
                    )
                    if prompt_len <= 0 or prompt_token_len <= 0:
                        prompt = torch.zeros((0,), dtype=torch.float32)
                        prompt_len = 0
                        prompt_token_len = 0
                    else:
                        prompt_audio_in = ref_audio
                        if ref_rms < args.target_rms:
                            prompt_audio_in = prompt_audio_in * args.target_rms / max(ref_rms, 1e-8)
                        prompt = prompt_audio_in[0, :prompt_len]

                    prompt_tokens = full_tokens[:prompt_token_len]
                    if args.ref_prompt_target_mode == "continuation":
                        target_tokens = full_tokens[prompt_token_len:]
                    elif args.ref_prompt_target_mode == "full":
                        target_tokens = full_tokens
                    else:
                        raise ValueError(f"Unknown ref prompt target mode: {args.ref_prompt_target_mode}")

                    # WavTTS has one token-conditioning tensor, so CosyVoice2's
                    # prompt_token + target_token split is represented explicitly here
                    # as history tokens followed by target tokens.
                    speech_tokens_with_history.append(torch.cat([prompt_tokens, target_tokens], dim=0))
                    prompt_tensors.append(prompt.to(device=device))
                    prompt_lengths.append(prompt_len)
                    target_duration = duration_from_token_count(
                        target_tokens.numel(),
                        args.speech_token_rate,
                        target_sample_rate,
                        wav_frame_len,
                    )
                    target_duration = align_to_frame_len(round(target_duration * args.duration_scale), wav_frame_len)
                    target_durations.append(target_duration)

                speech_token_sequences = speech_tokens_with_history
                cond = pad_sequence(prompt_tensors, batch_first=True, padding_value=0.0)
                lens = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
                durations = torch.tensor(
                    [prompt_len + target_duration for prompt_len, target_duration in zip(prompt_lengths, target_durations)],
                    dtype=torch.long,
                    device=device,
                )
                no_ref_audio = False
            elif not any(is_pair_row(row) for row in batch_rows):
                durations = torch.tensor(
                    [
                        align_to_frame_len(
                            round(
                                duration_from_token_count(
                                    t.numel(),
                                    args.speech_token_rate,
                                    target_sample_rate,
                                    wav_frame_len,
                                )
                                * args.duration_scale
                            ),
                            wav_frame_len,
                        )
                        for t in full_token_tensors
                    ],
                    dtype=torch.long,
                    device=device,
                )
                # No waveform prompt: lens=0 makes cond_mask empty, so the whole waveform is generated from tokens.
                cond = torch.zeros((len(batch_rows), 1), dtype=torch.float32, device=device)
                lens = torch.zeros((len(batch_rows),), dtype=torch.long, device=device)
                no_ref_audio = True
                prompt_lengths = [0] * len(batch_rows)

            speech_tokens = pad_sequence(speech_token_sequences, batch_first=True, padding_value=-1)

            spk_embedding = None
            if speaker_extractor is not None:
                spk_embedding = torch.stack(
                    [
                        speaker_extractor.extract_from_path(row_prompt_wavpath(row), device=device)
                        for row in batch_rows
                    ],
                    dim=0,
                )

            with torch.inference_mode():
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None and "cuda" in device):
                    generated, _ = model.sample(
                        cond=cond,
                        text=None,
                        speech_tokens=speech_tokens,
                        spk_embedding=spk_embedding,
                        lens=lens,
                        duration=durations,
                        steps=args.nfe_step,
                        cfg_strength=args.cfg_strength,
                        timestep_mapping=args.timestep_mapping,
                        timestep_power=args.timestep_power,
                        sway_sampling_coef=args.sway_sampling_coef,
                        shift=args.shift,
                        use_epss=args.timestep_mapping == "sway_sampling",
                        no_ref_audio=no_ref_audio,
                        seed=args.seed,
                    )

            for i, utt in enumerate(utts):
                gen = generated[i, : durations[i].item()].to(torch.float32).cpu()
                if (
                    prompt_lengths[i] > 0
                    and args.prepend_ref_prompt_output
                    and args.ref_prompt_target_mode == "continuation"
                ):
                    gen = torch.cat([ref_audios[i][0, : prompt_lengths[i]].cpu(), gen[prompt_lengths[i] :]], dim=0)
                elif prompt_lengths[i] > 0 and not args.keep_ref_prompt_output:
                    gen = gen[prompt_lengths[i] :]
                ref_audio = ref_audios[i]
                ref_rms = ref_rms_values[i]
                if ref_rms < args.target_rms:
                    gen = gen * ref_rms / args.target_rms

                torchaudio.save(str(output_dir / f"{utt}.wav"), gen.unsqueeze(0), target_sample_rate)
                ref_out = output_dir / "refs" / f"{utt}.wav"
                if not ref_out.exists():
                    torchaudio.save(str(ref_out), ref_audio.to(torch.float32), target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.num_shards > 1:
            manifest = output_dir / f"manifest.shard{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl"
        else:
            manifest = output_dir / "manifest.jsonl"
        with open(manifest, "w", encoding="utf-8") as f:
            for row in rows:
                obj = {
                    "utt": row_utt(row),
                    "text": row_text(row),
                    "ref_wavpath": row_prompt_wavpath(row),
                    "ref_prompt_target_mode": args.ref_prompt_target_mode,
                    "gen_wavpath": str(output_dir / f"{row_utt(row)}.wav"),
                    "token_length": len(row.get("target_token", row.get("token", []))),
                    "ckpt_path": ckpt_path,
                    "ckpt_step": ckpt_label,
                }
                if is_pair_row(row):
                    obj.update(
                        {
                            "prompt_utt": row.get("prompt_utt"),
                            "prompt_text": row.get("prompt_text"),
                            "prompt_token_length": len(row["prompt_token"]),
                            "target_utt": row.get("target_utt"),
                            "target_text": row.get("target_text"),
                            "target_wavpath": row.get("target_wavpath"),
                            "target_token_length": len(row["target_token"]),
                        }
                    )
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote generated wavs to {output_dir}")

    del model
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Token-to-wave inference from a JSONL test set.")
    parser.add_argument("--ckpt", "--ckptpath", required=True, help="Checkpoint file or ckpt directory.")
    parser.add_argument(
        "--ckpt-steps",
        nargs="*",
        default=None,
        help="One or more checkpoint steps under --ckpt. Accepts space- or comma-separated values, e.g. 180000 200000 or 180000,200000.",
    )
    parser.add_argument("--jsonl", required=True, help="JSONL with utt/text/wavpath/token.")
    parser.add_argument("--model-cfg", default=None, help="Training .hydra/config.yaml or WavTTS yaml.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory. With --ckpt-steps, outputs go to output-dir/experiment_name/step/.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--nfe-step",
        nargs="+",
        default=["50"],
        help="ODE steps. Accepts one or more space/comma-separated values, e.g. 20,30,40 or 20 30 40.",
    )
    parser.add_argument(
        "--cfg-strength",
        nargs="+",
        default=["1.5"],
        help="CFG strength. Accepts one or more space/comma-separated values.",
    )
    parser.add_argument("--timestep-mapping", default="power", choices=["uniform", "sway_sampling", "power"])
    parser.add_argument(
        "--timestep-power",
        nargs="+",
        default=["2.0"],
        help="Power exponent. Accepts one or more space/comma-separated values.",
    )
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument(
        "--shift",
        nargs="+",
        default=["3.0"],
        help="Timestep shift. Accepts one or more space/comma-separated values.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split JSONL rows into this many manual shards for multi-process inference.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Manual shard index for this process. Valid range: [0, num-shards).",
    )
    parser.add_argument("--load-dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--infer-dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--spk-embedding-model",
        default=None,
        help="CAM++ ONNX path. Overrides model.speaker_embedding.model_path.",
    )
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument(
        "--ref-prompt-seconds",
        type=float,
        default=0.0,
        help="Use the first N seconds of each row's wavpath as waveform reference prompt. 0 keeps token-only no-ref inference.",
    )
    parser.add_argument(
        "--speech-token-rate",
        type=float,
        default=25.0,
        help="Semantic speech token frame rate in Hz. Used to align prompt token history with prompt waveform.",
    )
    parser.add_argument(
        "--ref-prompt-target-mode",
        choices=["continuation", "full"],
        default="continuation",
        help=(
            "continuation: prompt is a prefix of the same utterance and target tokens are the remaining tokens. "
            "full: prompt is a separate reference and target tokens are the complete utterance."
        ),
    )
    parser.add_argument(
        "--keep-ref-prompt-output",
        action="store_true",
        help="Keep the prompt segment in saved audio when --ref-prompt-seconds is used.",
    )
    parser.add_argument(
        "--prepend-ref-prompt-output",
        action="store_true",
        help=(
            "For continuation mode, save original prompt wav + generated continuation as a full-length listening file. "
            "This keeps Cosy-style generation while avoiding a missing-prefix output."
        ),
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Scale duration inferred from token length. duration = len(token) / 25 * sample_rate * scale.",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    device = f"cuda:{accelerator.process_index}" if torch.cuda.is_available() else "cpu"

    nfe_steps = parse_list_values(args.nfe_step, int, "--nfe-step")
    if any(step < 4 for step in nfe_steps):
        print(
            "Warning: --nfe-step below 4 is a debug setting for WavTTS and "
            "will usually produce noisy/electronic or muffled audio. "
            "Use 20-50 steps for quality evaluation.",
            flush=True,
        )
    cfg_strengths = parse_list_values(args.cfg_strength, float, "--cfg-strength")
    timestep_powers = parse_list_values(args.timestep_power, float, "--timestep-power")
    shifts = parse_list_values(args.shift, float, "--shift")
    infer_cases = [
        SimpleNamespace(
            name=case_name(cfg_strength, shift, timestep_power, nfe_step),
            cfg_strength=cfg_strength,
            shift=shift,
            timestep_power=timestep_power,
            nfe_step=nfe_step,
        )
        for cfg_strength in cfg_strengths
        for shift in shifts
        for timestep_power in timestep_powers
        for nfe_step in nfe_steps
    ]
    multi_case = len(infer_cases) > 1

    ckpt_steps = parse_ckpt_steps(args.ckpt_steps)
    ckpt_specs = resolve_ckpt_paths(args.ckpt, ckpt_steps)
    first_ckpt_path = ckpt_specs[0][1]
    cfg_path = infer_model_cfg(first_ckpt_path, args.model_cfg)
    model_cfg = OmegaConf.load(cfg_path)

    target_sample_rate = int(model_cfg.model.waveform.target_sample_rate)
    wav_frame_len = int(model_cfg.model.waveform.wav_frame_len)
    if str(getattr(model_cfg.model.arch, "condition_type", "text")) != "speech_token":
        print("Warning: model.arch.condition_type is not speech_token in the loaded config.")

    rows = read_jsonl(args.jsonl)
    original_row_count = len(rows)
    rows = shard_rows(rows, args.num_shards, args.shard_index)
    if args.num_shards > 1 and accelerator.is_main_process:
        print(f"Manual shard {args.shard_index}/{args.num_shards}: {len(rows)} of {original_row_count} rows")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    autocast_dtype = dtype_map.get(args.infer_dtype)
    use_ckpt_hierarchy = ckpt_steps is not None or len(ckpt_specs) > 1
    output_base = Path(args.output_dir) if args.output_dir else default_output_base(args.jsonl)
    if not use_ckpt_hierarchy and args.output_dir is None:
        ckpt_label, ckpt_path = ckpt_specs[0]
        output_base = output_base / infer_experiment_name(ckpt_path) / ckpt_label

    if multi_case and accelerator.is_main_process:
        print(
            "Inference grid: "
            f"nfe={nfe_steps}, cfg={cfg_strengths}, shift={shifts}, power={timestep_powers} "
            f"({len(infer_cases)} configs)"
        )

    for case in infer_cases:
        case_args = SimpleNamespace(**vars(args))
        case_args.nfe_step = case.nfe_step
        case_args.cfg_strength = case.cfg_strength
        case_args.shift = case.shift
        case_args.timestep_power = case.timestep_power
        case_output_base = output_base / case.name if multi_case else output_base

        if multi_case and accelerator.is_main_process:
            print(
                f"======== {case.name}: cfg={case.cfg_strength}, shift={case.shift}, "
                f"mapping={args.timestep_mapping}, power={case.timestep_power}, nfe={case.nfe_step}"
            )

        for ckpt_label, ckpt_path in ckpt_specs:
            output_dir = get_output_dir(case_output_base, ckpt_path, ckpt_label, use_ckpt_hierarchy)
            run_inference_for_ckpt(
                accelerator=accelerator,
                rows=rows,
                model_cfg=model_cfg,
                ckpt_path=ckpt_path,
                ckpt_label=ckpt_label,
                output_dir=output_dir,
                device=device,
                args=case_args,
                target_sample_rate=target_sample_rate,
                wav_frame_len=wav_frame_len,
                autocast_dtype=autocast_dtype,
            )


if __name__ == "__main__":
    main()
