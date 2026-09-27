from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchaudio


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
    """Lazy prompt/target reader used by GRPO and standalone inference."""

    def __init__(
        self,
        jsonl_path: str,
        *,
        sample_rate: int = 16000,
        frame_len: int = 160,
        token_rate: float = 25.0,
        max_rows: int | None = None,
    ):
        self.jsonl_path = str(Path(jsonl_path).expanduser())
        self.base_dir = Path(self.jsonl_path).parent
        self.sample_rate = int(sample_rate)
        self.frame_len = int(frame_len)
        self.token_rate = float(token_rate)
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
        if path.is_absolute() and "data" in path.parts:
            data_index = path.parts.index("data")
            package_root = Path(__file__).resolve().parents[3]
            local_path = package_root.joinpath(*path.parts[data_index:])
            if local_path.exists():
                return str(local_path)
            repo_root = package_root.parent
            local_path = repo_root.joinpath(*path.parts[data_index:])
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

    def get(self, index: int) -> dict:
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

    def __getitem__(self, index: int) -> dict:
        return self.get(index)
