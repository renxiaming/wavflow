import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from infer_xpred_flow import (
    infer_experiment_name,
    infer_model_cfg,
    load_token_model,
    parse_ckpt_steps,
    read_jsonl,
    resolve_ckpt_paths,
)
from wavtts.model.speaker_embedding import SpeakerEmbeddingExtractor


def read_cases(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def complete_output(output_dir: Path, expected_rows: int) -> bool:
    manifest = output_dir / "manifest.jsonl"
    if not manifest.exists():
        return False
    if sum(1 for _ in manifest.open("r", encoding="utf-8")) != expected_rows:
        return False
    return len(list(output_dir.glob("*.wav"))) >= expected_rows


def make_batches(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def write_manifest(output_dir: Path, rows, ckpt_path: str, ckpt_label: str):
    manifest = output_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for row in rows:
            obj = {
                "utt": row["utt"],
                "text": row["text"],
                "ref_wavpath": row["wavpath"],
                "gen_wavpath": str(output_dir / f"{row['utt']}.wav"),
                "token_length": len(row["token"]),
                "ckpt_path": ckpt_path,
                "ckpt_step": ckpt_label,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def precompute_refs(rows, target_sample_rate: int, target_rms: float):
    refs = []
    for row in rows:
        ref_audio, ref_sr = torchaudio.load(row["wavpath"])
        if ref_sr != target_sample_rate:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, target_sample_rate)
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(dim=0, keepdim=True)
        ref_audio = ref_audio.to(torch.float32)
        rms = torch.sqrt(torch.mean(torch.square(ref_audio))).item()
        refs.append(
            {
                "audio": ref_audio,
                "rms": rms,
                "scale": rms / target_rms,
            }
        )
    return refs


def run_case_for_loaded_ckpt(
    *,
    model,
    rows,
    refs,
    model_cfg,
    ckpt_path: str,
    ckpt_label: str,
    output_dir: Path,
    device: str,
    case: dict,
    args,
    spk_embeddings,
    target_sample_rate: int,
    wav_frame_len: int,
    autocast_dtype,
):
    if complete_output(output_dir, len(rows)):
        print(f"[skip] {case['name']} ckpt={ckpt_label}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "refs").mkdir(parents=True, exist_ok=True)

    cfg_strength = float(case["cfg"])
    nfe_step = int(case["nfe"])
    shift = float(case["shift"])
    timestep_power = float(case["power"])

    for batch_rows in tqdm(
        list(make_batches(rows, args.batch_size)),
        desc=f"{case['name']}:{ckpt_label}",
        leave=False,
    ):
        utts = [row["utt"] for row in batch_rows]
        row_indices = [rows.index(row) for row in batch_rows]
        token_tensors = [
            torch.tensor(row["token"], dtype=torch.long, device=device).reshape(-1)
            for row in batch_rows
        ]
        speech_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=-1)
        token_lengths = torch.tensor([t.numel() for t in token_tensors], dtype=torch.long, device=device)
        durations = (token_lengths.float() / 25.0 * target_sample_rate * args.duration_scale).round().long()
        durations = ((durations + wav_frame_len - 1) // wav_frame_len) * wav_frame_len

        cond = torch.zeros((len(batch_rows), 1), dtype=torch.float32, device=device)
        lens = torch.zeros((len(batch_rows),), dtype=torch.long, device=device)
        no_ref_audio = True

        spk_embedding = None
        if spk_embeddings is not None:
            spk_embedding = torch.stack([spk_embeddings[idx].to(device=device) for idx in row_indices], dim=0)

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None and "cuda" in device):
                generated, _ = model.sample(
                    cond=cond,
                    text=None,
                    speech_tokens=speech_tokens,
                    spk_embedding=spk_embedding,
                    lens=lens,
                    duration=durations,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    timestep_mapping=args.timestep_mapping,
                    timestep_power=timestep_power,
                    sway_sampling_coef=args.sway_sampling_coef,
                    shift=shift,
                    use_epss=args.timestep_mapping == "sway_sampling",
                    no_ref_audio=no_ref_audio,
                    seed=args.seed,
                )

        for i, utt in enumerate(utts):
            gen = generated[i, : durations[i].item()].to(torch.float32).cpu()
            ref = refs[row_indices[i]]
            if ref["rms"] < args.target_rms:
                gen = gen * ref["scale"]
            torchaudio.save(str(output_dir / f"{utt}.wav"), gen.unsqueeze(0), target_sample_rate)
            ref_out = output_dir / "refs" / f"{utt}.wav"
            if not ref_out.exists():
                torchaudio.save(str(ref_out), ref["audio"], target_sample_rate)

    write_manifest(output_dir, rows, ckpt_path, ckpt_label)
    print(f"[done] {case['name']} ckpt={ckpt_label}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Grid worker for token-to-wave xpred flow inference.")
    parser.add_argument("--ckpt", "--ckptpath", required=True)
    parser.add_argument("--ckpt-steps", nargs="*", default=None)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--cases-tsv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timestep-mapping", default="power", choices=["uniform", "sway_sampling", "power"])
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load-dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--infer-dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--spk-embedding-model", default=None)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards).")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_steps = parse_ckpt_steps(args.ckpt_steps)
    ckpt_specs = resolve_ckpt_paths(args.ckpt, ckpt_steps)
    cfg_path = infer_model_cfg(ckpt_specs[0][1], args.model_cfg)
    model_cfg = OmegaConf.load(cfg_path)
    rows = read_jsonl(args.jsonl)
    refs = precompute_refs(rows, int(model_cfg.model.waveform.target_sample_rate), args.target_rms)
    cases = read_cases(args.cases_tsv)
    shard_cases = [
        case
        for idx, case in enumerate(
            sorted(cases, key=lambda c: (-int(c["nfe"]), c["name"]))
        )
        if idx % args.num_shards == args.shard_index
    ]

    target_sample_rate = int(model_cfg.model.waveform.target_sample_rate)
    wav_frame_len = int(model_cfg.model.waveform.wav_frame_len)
    spk_embeddings = None
    if bool(getattr(model_cfg.model.arch, "use_spk_embedding", False)):
        speaker_cfg = getattr(model_cfg.model, "speaker_embedding", {}) or {}
        speaker_model_path = args.spk_embedding_model or speaker_cfg.get("model_path")
        speaker_extractor = SpeakerEmbeddingExtractor(
            model_path=speaker_model_path,
            sample_rate=target_sample_rate,
        )
        spk_embeddings = [
            speaker_extractor.extract_from_path(row["wavpath"], device=device).detach()
            for row in rows
        ]

    exp_name = infer_experiment_name(ckpt_specs[0][1])
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    autocast_dtype = dtype_map.get(args.infer_dtype)
    output_root = Path(args.output_root)
    worker_args = SimpleNamespace(**vars(args))

    print(
        f"worker shard={args.shard_index}/{args.num_shards}, cases={len(shard_cases)}, ckpts={len(ckpt_specs)}, device={device}",
        flush=True,
    )

    for ckpt_label, ckpt_path in ckpt_specs:
        print(f"[load] ckpt={ckpt_label} path={ckpt_path}", flush=True)
        model = load_token_model(model_cfg, ckpt_path, device, args.load_dtype)
        model.eval()
        for case in shard_cases:
            output_dir = output_root / case["name"] / exp_name / ckpt_label
            run_case_for_loaded_ckpt(
                model=model,
                rows=rows,
                refs=refs,
                model_cfg=model_cfg,
                ckpt_path=ckpt_path,
                ckpt_label=ckpt_label,
                output_dir=output_dir,
                device=device,
                case=case,
                args=worker_args,
                spk_embeddings=spk_embeddings,
                target_sample_rate=target_sample_rate,
                wav_frame_len=wav_frame_len,
                autocast_dtype=autocast_dtype,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
