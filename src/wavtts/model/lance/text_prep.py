"""Shared text normalization for Emilia Lance training."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_utils_path = Path(__file__).resolve().parent.parent / "utils.py"
_utils_spec = importlib.util.spec_from_file_location("wavtts_model_utils", _utils_path)
_utils = importlib.util.module_from_spec(_utils_spec)
assert _utils_spec.loader is not None
_utils_spec.loader.exec_module(_utils)
convert_char_to_pinyin = _utils.convert_char_to_pinyin

_PUNCT_TRANS = str.maketrans({",": "，", "!": "！", "?": "？"})


def normalize_text(text: str) -> str:
    return text.translate(_PUNCT_TRANS)


def prepare_pinyin_text(text: str, polyphone: bool = True) -> list[str]:
    """Convert text to WavTTS pinyin tokens (content-based, not utt-prefix-based)."""
    return convert_char_to_pinyin([normalize_text(text)], polyphone=polyphone)[0]


def normalize_lang(lang: str | None) -> str:
    return (lang or "").strip().lower()


def utt_prefix_lang(utt: str) -> str:
    prefix = utt.split("_", 1)[0].lower()
    if prefix in {"zh", "en"}:
        return prefix
    return ""


def resolve_language(utt: str, lang: str | None = None) -> str:
    resolved = normalize_lang(lang)
    if resolved in {"zh", "en"}:
        return resolved
    return utt_prefix_lang(utt)


def language_allowed(lang: str, allowed_langs: set[str]) -> bool:
    if not allowed_langs:
        return True
    return normalize_lang(lang) in allowed_langs


def load_language_map(jsonl_path: str, utts: set[str]) -> dict[str, str]:
    """Load utt -> language for a subset of utts from emilia_merged.jsonl."""
    if not utts:
        return {}

    remaining = set(utts)
    lang_map: dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            obj = json.loads(line)
            utt = obj.get("id") or obj.get("utt")
            if utt not in remaining:
                continue
            lang_map[utt] = normalize_lang(obj.get("language"))
            remaining.remove(utt)

    if remaining:
        print(f"Warning: {len(remaining)} utts missing language in {jsonl_path}")

    return lang_map
