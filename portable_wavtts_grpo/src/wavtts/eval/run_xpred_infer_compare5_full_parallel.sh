#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
  CONDA_SH="${CONDA_SH:-/home/work_nfs20/xmren/miniconda3/etc/profile.d/conda.sh}"
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV:-wavtts}"
  fi
fi

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/exp/nar_wav_tts/Emilia_Lance/WavTTS_xpred_spkconcat-Emilia_Lance-tokentowav-xpred-vloss-spkconcat-repa-dit20-stft-vapa60k-8gpus-10000frames_per_gpu-bf16/ckpts}"
CKPT_STEPS="${CKPT_STEPS:-20000 40000 60000}"
JSONL="${JSONL:-data/emilia_lance_token_test_zh_en_200/test5.jsonl}"
OUT_ROOT="${OUT_ROOT:-results/xpred_spkconcat_compare5}"
DEVICE_LIST_STR="${DEVICE_LIST:-0 1 2 3 4 5 6}"
NUM_WORKERS="${NUM_WORKERS:-7}"

CFG_VALUES=(${CFG_VALUES:-0 0.5 1.0 1.5 2.0 2.5})
SHIFT_VALUES=(${SHIFT_VALUES:-1.0 1.5 2.0 2.5})
POWER_VALUES=(${POWER_VALUES:-1.0 1.5 2.0 2.5})
NFE_VALUES=(${NFE_VALUES:-10 15 20 25 30 50 100})

TIMESTEP_MAPPING="${TIMESTEP_MAPPING:-power}"
SWAY_SAMPLING_COEF="${SWAY_SAMPLING_COEF:--1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOAD_DTYPE="${LOAD_DTYPE:-fp32}"
INFER_DTYPE="${INFER_DTYPE:-bf16}"
SEED="${SEED:-}"

EXPECTED_ROWS="$(wc -l < "${JSONL}")"

tag_num() {
  local value="$1"
  value="${value/./p}"
  value="${value%p0}"
  echo "${value}"
}

case_name() {
  local cfg="$1"
  local shift_val="$2"
  local power="$3"
  local nfe="$4"
  echo "cfg$(tag_num "${cfg}")_shift$(tag_num "${shift_val}")_power$(tag_num "${power}")_nfe${nfe}"
}

manifest_complete() {
  local manifest="$1"
  [[ -f "${manifest}" ]] || return 1
  [[ "$(wc -l < "${manifest}")" -eq "${EXPECTED_ROWS}" ]] || return 1
  [[ "$(find "$(dirname "${manifest}")" -maxdepth 1 -type f -name '*.wav' | wc -l)" -ge "${EXPECTED_ROWS}" ]] || return 1
}

case_complete() {
  local name="$1"
  local exp_name
  exp_name="$(basename "$(dirname "${CKPT_DIR}")")"
  local step
  for step in ${CKPT_STEPS}; do
    manifest_complete "${OUT_ROOT}/${name}/${exp_name}/${step}/manifest.jsonl" || return 1
  done
}

run_job() {
  local slot="$1"
  local name="$2"
  local cfg="$3"
  local shift_val="$4"
  local power="$5"
  local nfe="$6"

  local devices=(${DEVICE_LIST_STR})
  local device_index=$(( (slot - 1) % ${#devices[@]} ))
  local gpu="${devices[${device_index}]}"
  local seed_args=()

  if [[ -n "${SEED}" ]]; then
    seed_args=(--seed "${SEED}")
  fi

  if case_complete "${name}"; then
    echo "[skip][gpu ${gpu}] ${name}"
    return 0
  fi

  echo "[run][gpu ${gpu}] ${name}: cfg=${cfg}, shift=${shift_val}, power=${power}, nfe=${nfe}, ckpt_steps=${CKPT_STEPS}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python src/wavtts/eval/infer_xpred_flow.py \
    --ckptpath "${CKPT_DIR}" \
    --ckpt-steps ${CKPT_STEPS} \
    --jsonl "${JSONL}" \
    --output-dir "${OUT_ROOT}/${name}" \
    --batch-size "${BATCH_SIZE}" \
    --nfe-step "${nfe}" \
    --cfg-strength "${cfg}" \
    --timestep-mapping "${TIMESTEP_MAPPING}" \
    --timestep-power "${power}" \
    --sway-sampling-coef "${SWAY_SAMPLING_COEF}" \
    --shift "${shift_val}" \
    --load-dtype "${LOAD_DTYPE}" \
    --infer-dtype "${INFER_DTYPE}" \
    "${seed_args[@]}"
}

write_summary() {
  OUT_ROOT="${OUT_ROOT}" CASE_TSV="${CASE_TSV}" python - <<'PY'
import csv
import os
from pathlib import Path

import torch
import torchaudio

out_root = Path(os.environ["OUT_ROOT"])
case_tsv = Path(os.environ["CASE_TSV"])
summary_tsv = out_root / "config_summary.tsv"

cases = {}
with case_tsv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        cases[row["name"]] = row

with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "name",
        "cfg",
        "nfe",
        "shift",
        "power",
        "ckpt",
        "wav_count",
        "avg_rms",
        "max_peak",
        "avg_clip_pct",
        "path",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for manifest in sorted(out_root.glob("*/*/*/manifest.jsonl")):
        rel = manifest.relative_to(out_root).parts
        name, _exp_name, ckpt = rel[:3]
        meta = cases.get(name, {})
        wavs = sorted(p for p in manifest.parent.glob("*.wav") if p.parent.name != "refs")
        rms_values = []
        peak_values = []
        clip_pcts = []
        for wav in wavs:
            audio, _sr = torchaudio.load(str(wav))
            audio = audio.to(torch.float32)
            rms_values.append(torch.sqrt(torch.mean(audio.square())).item())
            peak_values.append(audio.abs().max().item())
            clip_pcts.append((audio.abs() >= 0.999).to(torch.float32).mean().item() * 100.0)

        writer.writerow(
            {
                "name": name,
                "cfg": meta.get("cfg", ""),
                "nfe": meta.get("nfe", ""),
                "shift": meta.get("shift", ""),
                "power": meta.get("power", ""),
                "ckpt": ckpt,
                "wav_count": len(wavs),
                "avg_rms": f"{sum(rms_values) / len(rms_values):.4f}" if rms_values else "0.0000",
                "max_peak": f"{max(peak_values):.4f}" if peak_values else "0.0000",
                "avg_clip_pct": f"{sum(clip_pcts) / len(clip_pcts):.4f}" if clip_pcts else "0.0000",
                "path": str(manifest.parent),
            }
        )

print(f"Wrote summary to {summary_tsv}")
PY
}

mkdir -p "${OUT_ROOT}/logs"
CASE_TSV="${OUT_ROOT}/cases.tsv"
JOBS_TSV="${OUT_ROOT}/logs/full_grid_jobs.tsv"
JOBLOG="${OUT_ROOT}/logs/full_grid_parallel_joblog.tsv"

printf 'name\tcfg\tnfe\tshift\tpower\n' > "${CASE_TSV}"
printf 'name\tcfg\tshift\tpower\tnfe\n' > "${JOBS_TSV}"
for cfg in "${CFG_VALUES[@]}"; do
  for shift_val in "${SHIFT_VALUES[@]}"; do
    for power in "${POWER_VALUES[@]}"; do
      for nfe in "${NFE_VALUES[@]}"; do
        name="$(case_name "${cfg}" "${shift_val}" "${power}" "${nfe}")"
        printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${cfg}" "${nfe}" "${shift_val}" "${power}" >> "${CASE_TSV}"
        printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${cfg}" "${shift_val}" "${power}" "${nfe}" >> "${JOBS_TSV}"
      done
    done
  done
done

export CKPT_DIR CKPT_STEPS JSONL OUT_ROOT DEVICE_LIST_STR NUM_WORKERS
export TIMESTEP_MAPPING SWAY_SAMPLING_COEF BATCH_SIZE LOAD_DTYPE INFER_DTYPE SEED EXPECTED_ROWS
export -f tag_num manifest_complete case_complete run_job

echo "Prepared $(($(wc -l < "${JOBS_TSV}") - 1)) config jobs under ${OUT_ROOT}"
tail -n +2 "${JOBS_TSV}" | parallel --joblog "${JOBLOG}" --colsep '\t' -j "${NUM_WORKERS}" --halt soon,fail=1 run_job "{%}" {1} {2} {3} {4} {5}

write_summary
