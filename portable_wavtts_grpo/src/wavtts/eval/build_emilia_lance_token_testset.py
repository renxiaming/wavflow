import argparse
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from wavtts.model.lance.aslp.tools.lance_pack import LanceReader
from wavtts.model.lance.text_index import EmiliaTextIndex
from wavtts.model.lance.utils import load_co


def parse_item(item: str):
    parts = item.strip().split("|")
    if len(parts) < 6:
        return None
    return {
        "utt": parts[0],
        "wav_rowid": int(parts[1]),
        "wav_offset": int(parts[2]),
        "token_rowid": int(parts[3]),
        "token_offset": int(parts[4]),
        "token_length": int(parts[5]),
    }


def lang_from_utt(utt: str) -> str:
    return utt.split("_", 1)[0].lower()


def split_counts(total: int, languages: list[str]) -> dict[str, int]:
    base = total // len(languages)
    rem = total % len(languages)
    return {lang: base + (1 if i < rem else 0) for i, lang in enumerate(languages)}


def reservoir_sample_li(
    li_path: str,
    count: int,
    seed: int,
    max_scan: int | None,
    languages: list[str] | None,
):
    rng = random.Random(seed)
    if languages:
        languages = [lang.lower() for lang in languages]
        target_counts = split_counts(count, languages)
        reservoir = {lang: [] for lang in languages}
        seen = {lang: 0 for lang in languages}
    else:
        target_counts = None
        reservoir = []
        seen = 0

    scanned = 0
    with open(li_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Sampling Lance .li"):
            item = line.strip()
            if not item:
                continue
            scanned += 1
            parsed = parse_item(item)
            if parsed is None:
                continue

            if languages:
                lang = lang_from_utt(parsed["utt"])
                if lang not in target_counts:
                    if max_scan is not None and scanned >= max_scan:
                        break
                    continue
                seen[lang] += 1
                lang_reservoir = reservoir[lang]
                lang_count = target_counts[lang]
                if len(lang_reservoir) < lang_count:
                    lang_reservoir.append(item)
                else:
                    j = rng.randrange(seen[lang])
                    if j < lang_count:
                        lang_reservoir[j] = item
            else:
                seen += 1
                if len(reservoir) < count:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < count:
                        reservoir[j] = item

            if max_scan is not None and scanned >= max_scan:
                break

    if languages:
        missing = {
            lang: target_counts[lang] - len(reservoir[lang])
            for lang in languages
            if len(reservoir[lang]) < target_counts[lang]
        }
        if missing:
            raise RuntimeError(
                f"Not enough samples for languages {missing}. "
                "Use --max-scan 0 to scan the full .li file or reduce --count."
            )
        flat = []
        for lang in languages:
            flat.extend(reservoir[lang])
        rng.shuffle(flat)
        return flat

    rng.shuffle(reservoir)
    return reservoir


def token_to_list(token_data):
    if isinstance(token_data, np.ndarray):
        return token_data.reshape(-1).astype(np.int64).tolist()
    return np.asarray(token_data).reshape(-1).astype(np.int64).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Build a token-to-wave JSONL test set from Emilia Lance training data."
    )
    parser.add_argument(
        "--lance-path",
        default="/home/work_nfs23/xmren/lance/emilia_wav_token",
        help="Lance prefix without .co/.li",
    )
    parser.add_argument(
        "--text-index-path",
        default="/home/work_nfs23/xmren/lance/emilia_text.db",
        help="SQLite utt->text index",
    )
    parser.add_argument(
        "--text-jsonl-path",
        default="/home/work_nfs23/hkxie/data/Emilia/emilia_merged.jsonl",
        help="Fallback text jsonl, used only if sqlite is unavailable",
    )
    parser.add_argument(
        "--out-dir",
        default="data/emilia_lance_token_test_200",
        help="Directory for test.jsonl and extracted wavs",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--languages",
        default="zh,en",
        help="Comma-separated utt language prefixes to keep, e.g. zh,en. Use empty string for no language filter.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="Maximum .li rows to scan for reservoir sampling. Use 0 to scan all rows.",
    )
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()

    lance_path = args.lance_path
    li_path = lance_path + ".li"
    co_path = lance_path + ".co"
    if not os.path.isfile(li_path):
        raise FileNotFoundError(f"Lance .li not found: {li_path}")
    if not os.path.isfile(co_path):
        raise FileNotFoundError(f"Lance .co not found: {co_path}")

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "test.jsonl"

    max_scan = None if args.max_scan == 0 else args.max_scan
    languages = [lang.strip().lower() for lang in args.languages.split(",") if lang.strip()]
    sampled_items = reservoir_sample_li(li_path, args.count, args.seed, max_scan, languages or None)

    conns = load_co(co_path)
    readers = [LanceReader(path, target_cls=cls) for path, cls in conns]
    text_index = EmiliaTextIndex(
        index_path=args.text_index_path if os.path.isfile(args.text_index_path) else None,
        jsonl_path=args.text_jsonl_path,
    )

    written = 0
    tried = 0
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for item in tqdm(sampled_items, desc="Extracting samples"):
            if written >= args.count:
                break
            parsed = parse_item(item)
            if parsed is None:
                continue
            tried += 1
            duration = parsed["token_length"] / 25.0
            if not (args.min_duration <= duration <= args.max_duration):
                continue

            utt = parsed["utt"]
            text = text_index.get_text(utt)
            if text is None:
                continue

            try:
                wav_data = readers[parsed["wav_offset"]].get_datas_by_rowids([parsed["wav_rowid"]])[0]
                audio, source_sample_rate = torchaudio.load(io.BytesIO(wav_data.mp3_binary))
                token_data = readers[parsed["token_offset"]].get_datas_by_rowids([parsed["token_rowid"]])[0].data
            except Exception as exc:
                print(f"Skip {utt}: failed to read Lance rows: {exc}")
                continue

            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if source_sample_rate != args.target_sample_rate:
                audio = torchaudio.functional.resample(audio, source_sample_rate, args.target_sample_rate)

            wavpath = wav_dir / f"{utt}.wav"
            torchaudio.save(str(wavpath), audio.to(torch.float32), args.target_sample_rate)

            token = token_to_list(token_data)
            obj = {
                "utt": utt,
                "text": text,
                "wavpath": str(wavpath),
                "token": token,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    if written < args.count:
        raise RuntimeError(
            f"Only wrote {written}/{args.count} samples after trying {tried}. "
            "Increase --max-scan or check Lance/text paths."
        )
    print(f"Wrote {written} samples to {jsonl_path}")


if __name__ == "__main__":
    main()
