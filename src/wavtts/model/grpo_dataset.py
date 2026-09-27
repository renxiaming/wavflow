from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchaudio


def sample_prefix_length(
    duration: int,
    *,
    frame_len: int,
    min_fraction: float,
    max_fraction: float,
    seed: int,
) -> tuple[int, float]:
    """Sample a reproducible frame-aligned prefix and leave a non-empty suffix."""
    if not 0.0 <= min_fraction <= max_fraction < 1.0:
        raise ValueError(
            "Prefix fractions must satisfy 0 <= min <= max < 1, got "
            f"{min_fraction}, {max_fraction}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    unit = float(torch.rand((), generator=generator).item())
    fraction = min_fraction + unit * (max_fraction - min_fraction)
    prefix = int(duration * fraction)
    prefix = (prefix // frame_len) * frame_len
    prefix = min(prefix, duration - frame_len)
    return max(0, prefix), fraction


def _duration_from_tokens(
    token_count: int,
    *,
    token_rate: float,
    sample_rate: int,
    frame_len: int,
) -> int:
    samples = round(float(token_count) / token_rate * sample_rate)
    return max(frame_len, (samples // frame_len) * frame_len)


class PromptTargetJsonl:
    """Lazy JSONL reader used by GRPO and standalone inference.

    ``paired`` keeps the original prompt/target TTS behavior. ``self_prefix``
    treats ``prompt_wav`` and ``prompt_token`` as one complete utterance; the
    caller chooses a prefix from that utterance and predicts its suffix.
    """

    def __init__(
        self,
        jsonl_path: str,
        *,
        sample_rate: int = 16000,
        frame_len: int = 160,
        token_rate: float = 25.0,
        max_rows: int | None = None,
        data_mode: str = "paired",
    ):
        self.jsonl_path = str(Path(jsonl_path).expanduser())
        self.base_dir = Path(self.jsonl_path).parent
        self.sample_rate = int(sample_rate)
        self.frame_len = int(frame_len)
        self.token_rate = float(token_rate)
        if data_mode not in {"paired", "self_prefix"}:
            raise ValueError(f"data_mode must be paired or self_prefix, got {data_mode}")
        self.data_mode = data_mode
        self.rows: list[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
                    if max_rows is not None and len(self.rows) >= max_rows:
                        break
        if not self.rows:
            raise ValueError(f"No rows found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> str:
        path = Path(value).expanduser()
        if path.exists():
            return str(path)
        project_root = Path(__file__).resolve().parents[3]
        # Seed inference manifests may retain prompt paths from the machine
        # that produced the tokens. Resolve those prompts to this checkout by
        # language and basename.
        if "prompt-wavs" in path.parts:
            prompt_index = path.parts.index("prompt-wavs")
            language = path.parts[prompt_index - 1] if prompt_index > 0 else "zh"
            local_prompt = project_root / "data" / "grpo_seed" / "wav" / language / path.name
            if local_prompt.exists():
                return str(local_prompt)
        if path.is_absolute() and "data" in path.parts:
            data_index = path.parts.index("data")
            local_path = project_root.joinpath(*path.parts[data_index:])
            if local_path.exists():
                return str(local_path)
        return str(path if path.is_absolute() else self.base_dir / path)

    def _load_audio(self, path: str) -> torch.Tensor:
        audio, source_rate = torchaudio.load(self._path(path))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if source_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, self.sample_rate)
        return audio.squeeze(0).to(torch.float32)

    def _full_prompt_utterance(self, row: dict, index: int) -> dict:
        wav_key = row.get("prompt_wav") or row.get("prompt_wavpath")
        if not wav_key:
            raise KeyError("Each self-prefix GRPO row needs prompt_wav or prompt_wavpath.")
        tokens = row.get("prompt_token") or row.get("token")
        if not tokens:
            raise KeyError("Each self-prefix GRPO row needs prompt_token or token.")

        audio = self._load_audio(wav_key)
        audio_duration = (audio.numel() // self.frame_len) * self.frame_len
        token_count = min(
            len(tokens),
            max(1, int(audio_duration / self.sample_rate * self.token_rate)),
        )
        duration = min(
            audio_duration,
            _duration_from_tokens(
                token_count,
                token_rate=self.token_rate,
                sample_rate=self.sample_rate,
                frame_len=self.frame_len,
            ),
        )
        duration -= duration % self.frame_len
        if duration <= self.frame_len:
            raise ValueError(f"Self-prefix audio is too short for suffix prediction: {wav_key}")

        return {
            "utt": str(row.get("utt", index)),
            "audio_path": self._path(wav_key),
            "audio": audio[:duration],
            "speech_token": torch.tensor(tokens[:token_count], dtype=torch.long),
            "duration": int(duration),
            "prompt_text": str(row.get("prompt_text", "")),
            # In self-prefix mode the source is prompt_wav, so WER uses its
            # supplied prompt_text transcript. SIM still uses the full wav.
            "target_text": str(row.get("prompt_text") or ""),
            "language": row.get("language"),
        }

    def _paired_utterance(self, row: dict, index: int) -> dict:
        row = self.rows[index]
        prompt_wav_key = row.get("prompt_wav") or row.get("prompt_wavpath")
        if not prompt_wav_key:
            raise KeyError("Each GRPO row needs prompt_wav or prompt_wavpath.")
        prompt_tokens = row.get("prompt_token")
        target_tokens = row.get("gen_token") or row.get("target_token")
        if not prompt_tokens or not target_tokens:
            raise KeyError("Each GRPO row needs prompt_token and gen_token/target_token.")

        prompt_audio = self._load_audio(prompt_wav_key)
        prompt_token_count = len(prompt_tokens)
        prompt_duration = min(
            _duration_from_tokens(
                prompt_token_count,
                token_rate=self.token_rate,
                sample_rate=self.sample_rate,
                frame_len=self.frame_len,
            ),
            (prompt_audio.numel() // self.frame_len) * self.frame_len,
        )
        prompt_token_count = min(
            prompt_token_count,
            max(1, round(prompt_duration / self.sample_rate * self.token_rate)),
        )
        prompt_duration = _duration_from_tokens(
            prompt_token_count,
            token_rate=self.token_rate,
            sample_rate=self.sample_rate,
            frame_len=self.frame_len,
        )
        prompt_duration = min(prompt_duration, prompt_audio.numel())
        prompt_duration -= prompt_duration % self.frame_len
        if prompt_duration <= 0:
            raise ValueError(f"Prompt audio is empty after frame alignment: {prompt_wav_key}")

        target_wav_key = row.get("gen_wav") or row.get("target_wavpath") or row.get("target_wav")
        if target_wav_key and os.path.exists(self._path(target_wav_key)):
            target_audio = self._load_audio(target_wav_key)
            target_duration = (target_audio.numel() // self.frame_len) * self.frame_len
        else:
            target_duration = _duration_from_tokens(
                len(target_tokens),
                token_rate=self.token_rate,
                sample_rate=self.sample_rate,
                frame_len=self.frame_len,
            )
        if target_duration <= 0:
            raise ValueError(f"Target duration is empty for row {index}.")

        text = row.get("target_text") or row.get("text") or ""
        return {
            "utt": str(row.get("utt", index)),
            "prompt_wav_path": self._path(prompt_wav_key),
            "prompt_wav": prompt_audio[:prompt_duration],
            "prompt_token": torch.tensor(prompt_tokens[:prompt_token_count], dtype=torch.long),
            "target_token": torch.tensor(target_tokens, dtype=torch.long),
            "prompt_duration": int(prompt_duration),
            "target_duration": int(target_duration),
            "target_text": str(text),
            "target_wav_path": self._path(target_wav_key) if target_wav_key else None,
            "prompt_text": row.get("prompt_text", ""),
            "language": row.get("language"),
        }

    def get(self, index: int) -> dict:
        row = self.rows[index]
        if self.data_mode == "self_prefix":
            return self._full_prompt_utterance(row, index)
        return self._paired_utterance(row, index)

    def __getitem__(self, index: int) -> dict:
        return self.get(index)
