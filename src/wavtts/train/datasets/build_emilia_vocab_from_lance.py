"""Build pinyin vocab.txt from Emilia Lance (.li) + text sqlite/jsonl metadata."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "lance"))
from text_prep import (  # noqa: E402
    language_allowed,
    load_language_map,
    prepare_pinyin_text,
    resolve_language,
)


def _process_batch(items: list[tuple[str, str]], polyphone: bool) -> set[str]:
    vocab: set[str] = set()
    for _, raw_text in items:
        vocab.update(prepare_pinyin_text(raw_text, polyphone=polyphone))
    return vocab


def load_li_utts(li_path: str) -> list[str]:
    utts = []
    with open(li_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                utts.append(line.split("|", 1)[0])
    return utts


def _table_has_language(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('utt_text') WHERE name='language'"
    ).fetchone()
    return row is not None


def fetch_texts(
    text_db_path: str,
    utts: list[str],
    allowed_langs: set[str],
    jsonl_path: str | None = None,
    batch_size: int = 5000,
) -> list[tuple[str, str]]:
    conn = sqlite3.connect(f"file:{text_db_path}?mode=ro", uri=True)
    has_language = _table_has_language(conn)
    lang_map = {}
    if not has_language and jsonl_path:
        print(f"Loading language metadata from {jsonl_path}")
        lang_map = load_language_map(jsonl_path, set(utts))

    samples: list[tuple[str, str]] = []
    missing = 0
    skipped_lang = 0
    for i in tqdm(range(0, len(utts), batch_size), desc="Loading text from sqlite"):
        chunk = utts[i : i + batch_size]
        placeholders = ",".join("?" for _ in chunk)
        if has_language:
            rows = conn.execute(
                f"SELECT utt, text, language FROM utt_text WHERE utt IN ({placeholders})",
                chunk,
            ).fetchall()
            found = {utt: (text, language) for utt, text, language in rows}
        else:
            rows = conn.execute(
                f"SELECT utt, text FROM utt_text WHERE utt IN ({placeholders})",
                chunk,
            ).fetchall()
            found = {utt: (text, lang_map.get(utt, "")) for utt, text in rows}

        for utt in chunk:
            item = found.get(utt)
            if item is None:
                missing += 1
                continue
            text, language = item
            lang = resolve_language(utt, language)
            if not language_allowed(lang, allowed_langs):
                skipped_lang += 1
                continue
            samples.append((utt, text))

    conn.close()
    if missing:
        print(f"Warning: {missing} utts missing in {text_db_path}")
    if skipped_lang:
        print(f"Skipped {skipped_lang} non-target-language samples (allowed={sorted(allowed_langs)})")
    return samples


def build_vocab(
    lance_path: str,
    text_db_path: str,
    out_path: str,
    jsonl_path: str | None = None,
    allowed_langs: set[str] | None = None,
    polyphone: bool = True,
    max_workers: int = 32,
    chunk_size: int = 256,
) -> str:
    allowed_langs = allowed_langs or {"zh", "en"}
    li_path = lance_path + ".li"
    utts = load_li_utts(li_path)
    print(f"Loaded {len(utts)} utts from {li_path}")

    samples = fetch_texts(
        text_db_path,
        utts,
        allowed_langs=allowed_langs,
        jsonl_path=jsonl_path,
    )
    print(f"Collected {len(samples)} zh/en text samples for vocab")

    vocab: set[str] = set()
    batches = [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_batch, batch, polyphone) for batch in batches]
        for future in tqdm(futures, desc="Building vocab"):
            vocab.update(future.result())

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens = sorted(vocab)
    if " " in tokens:
        tokens.remove(" ")
    with open(out, "w", encoding="utf-8") as f:
        f.write(" \n")
        for token in tokens:
            f.write(f"{token}\n")

    print(f"Vocab size: {len(tokens) + 1}")
    print(f"Saved to {out}")
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Build Emilia Lance pinyin vocab.txt")
    parser.add_argument(
        "--lance_path",
        default="/home/work_nfs23/xmren/lance/emilia_wav_token",
        help="Lance index prefix (.li/.co)",
    )
    parser.add_argument(
        "--text_db",
        default="/home/work_nfs23/xmren/lance/emilia_text.db",
        help="utt->text sqlite index",
    )
    parser.add_argument(
        "--jsonl_path",
        default="/home/work_nfs23/hkxie/data/Emilia/emilia_merged.jsonl",
        help="Fallback metadata for language filtering when text db has no language column",
    )
    parser.add_argument(
        "--langs",
        default="zh,en",
        help="Comma-separated language codes to include (from metadata or ZH_/EN_ prefix)",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Output vocab.txt path (default: data/Emilia_Lance_pinyin/vocab.txt under project root)",
    )
    parser.add_argument("--polyphone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    out_path = args.out_path or str(project_root / "data" / "Emilia_Lance_pinyin" / "vocab.txt")
    allowed_langs = {lang.strip().lower() for lang in args.langs.split(",") if lang.strip()}

    build_vocab(
        lance_path=args.lance_path,
        text_db_path=args.text_db,
        out_path=out_path,
        jsonl_path=args.jsonl_path,
        allowed_langs=allowed_langs,
        polyphone=args.polyphone,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
