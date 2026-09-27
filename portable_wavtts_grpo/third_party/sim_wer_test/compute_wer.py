import argparse
import string

from jiwer import compute_measures
from zhon.hanzi import punctuation as zh_punctuation


PUNCTUATION_ALL = zh_punctuation + string.punctuation + "-"


def read_texts(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            texts.append(parts[1] if len(parts) == 2 else parts[0])
    return texts


def normalize(text, char_level=False):
    for item in PUNCTUATION_ALL:
        if item == "'":
            continue
        text = text.replace(item, "")
    text = " ".join(text.split()).lower()
    if char_level:
        text = " ".join(list(text.replace(" ", "")))
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", type=int, default=1, help="1 表示按字统计，0 表示按词统计")
    parser.add_argument("--v", type=int, default=1, help="兼容旧 compute-wer.py 的参数")
    parser.add_argument("gt_file")
    parser.add_argument("gen_file")
    args = parser.parse_args()

    refs = read_texts(args.gt_file)
    hyps = read_texts(args.gen_file)
    if len(refs) != len(hyps):
        raise ValueError(f"Line count mismatch: gt={len(refs)}, gen={len(hyps)}")

    ref_text = " ".join(normalize(text, bool(args.char)) for text in refs)
    hyp_text = " ".join(normalize(text, bool(args.char)) for text in hyps)
    measures = compute_measures(ref_text, hyp_text)

    print(f"utt_count: {len(refs)}")
    print(f"wer: {measures['wer']:.6f}")
    print(f"wer_percent: {measures['wer'] * 100:.2f}")
    print(f"substitutions: {measures['substitutions']}")
    print(f"deletions: {measures['deletions']}")
    print(f"insertions: {measures['insertions']}")
    print(f"hits: {measures['hits']}")


if __name__ == "__main__":
    main()
