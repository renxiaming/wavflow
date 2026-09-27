from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def same_device(src: Path, dst_parent: Path) -> bool:
    try:
        return src.stat().st_dev == dst_parent.stat().st_dev
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Copy portable GRPO audio files from manifest.")
    parser.add_argument("--manifest", default="tools/audio_manifest.tsv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--copy", action="store_true", help="Force byte copy instead of hardlink when possible.")
    parser.add_argument("--limit", type=int, default=None, help="Debug: copy only the first N entries.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = (root / args.manifest).resolve()
    copied = 0
    linked = 0
    skipped = 0
    missing = 0

    with manifest.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        if header != ["source", "dest_rel"]:
            raise ValueError(f"Unexpected manifest header: {header}")
        for index, line in enumerate(handle, start=1):
            if args.limit is not None and index > args.limit:
                break
            source_text, dest_rel = line.rstrip("\n").split("\t", 1)
            source = Path(source_text)
            dest = root / dest_rel
            if dest.exists():
                skipped += 1
                continue
            if not source.exists():
                missing += 1
                print(f"[missing] {source}", flush=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not args.copy and same_device(source, dest.parent):
                os.link(source, dest)
                linked += 1
            else:
                shutil.copy2(source, dest)
                copied += 1
            done = copied + linked + skipped + missing
            if done % 10000 == 0:
                print(
                    f"processed={done} linked={linked} copied={copied} "
                    f"skipped={skipped} missing={missing}",
                    flush=True,
                )

    print(
        f"done linked={linked} copied={copied} skipped={skipped} missing={missing}",
        flush=True,
    )


if __name__ == "__main__":
    main()
