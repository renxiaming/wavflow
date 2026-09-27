"""Build sqlite utt->text index for Emilia Lance training."""

import argparse
import sys
from pathlib import Path

# Import directly to avoid pulling in the full wavtts.model stack.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "lance"))
from text_index import EmiliaTextIndex


def main():
    parser = argparse.ArgumentParser(description="Build Emilia utt->text sqlite index")
    parser.add_argument(
        "--jsonl_path",
        default="/home/work_nfs23/hkxie/data/Emilia/emilia_merged.jsonl",
        help="Source metadata jsonl with id/utt and text fields",
    )
    parser.add_argument(
        "--db_path",
        default="/home/work_nfs23/hkxie/data/Emilia/emilia_text.db",
        help="Output sqlite database path",
    )
    parser.add_argument(
        "--li_path",
        default=None,
        help="Optional Lance .li path; build a subset index for listed utts only",
    )
    parser.add_argument(
        "--source_db",
        default="/home/work_nfs23/xmren/lance/emilia_text.db",
        help="Existing full sqlite index used with --li_path",
    )
    args = parser.parse_args()

    if args.li_path:
        print(f"Building subset sqlite index from {args.li_path}")
        print(f"Source db: {args.source_db}")
        print(f"Writing to {args.db_path}")
        EmiliaTextIndex.build_sqlite_from_li(args.li_path, args.db_path, args.source_db)
    else:
        print(f"Building sqlite index from {args.jsonl_path}")
        print(f"Writing to {args.db_path}")
        EmiliaTextIndex.build_sqlite(args.jsonl_path, args.db_path)
    print("Done.")


if __name__ == "__main__":
    main()
