#!/usr/bin/env bash
set -euo pipefail

source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EXP_DIR="${EXP_DIR:-}"
CKPT_DIR="${CKPT_DIR:-${EXP_DIR:+${EXP_DIR}/ckpts}}"
MODEL_CFG="${MODEL_CFG:-${EXP_DIR:+${EXP_DIR}/.hydra/config.yaml}}"
JSONL="${JSONL:-${PROJECT_ROOT}/data/grpo_seed/zh/seedtest_zh_cosyvoice2_tokens.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/token_to_wave_infer}"

CKPT_STEPS="${CKPT_STEPS:-last}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
NFE_STEP="${NFE_STEP:-10,20,30,40,50}"
CFG_STRENGTH="${CFG_STRENGTH:-1.5,2.0}"
SHIFT="${SHIFT:-3.0}" 
TIMESTEP_MAPPING="${TIMESTEP_MAPPING:-power}"
TIMESTEP_POWER="${TIMESTEP_POWER:-2.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOAD_DTYPE="${LOAD_DTYPE:-fp32}"
INFER_DTYPE="${INFER_DTYPE:-bf16}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: bash src/wavtts/infer/infer.sh [options] [-- extra infer_xpred_flow.py args]

Options:
  --jsonl PATH             JSONL input.
  --ckpt-dir PATH          Checkpoint directory (use infer_grpo.sh for a GRPO model_last.pt file).
  --model-cfg PATH         Model config YAML.
  --ckpt-steps STEPS       Checkpoint steps, e.g. "140000" or "140000,180000".
  --output-dir DIR         Output directory.
  --gpus IDS               CUDA_VISIBLE_DEVICES.
  --procs-per-gpu N        Worker processes per GPU. Default: ${PROCS_PER_GPU}
  --nfe-step N             ODE steps. Accepts comma/space lists. Default: ${NFE_STEP}
  --cfg-strength X         CFG strength. Accepts comma/space lists. Default: ${CFG_STRENGTH}
  --shift X                Timestep shift. Accepts comma/space lists. Default: ${SHIFT}
  --timestep-mapping NAME  uniform | sway_sampling | power.
  --timestep-power X       Power exponent. Accepts comma/space lists. Default: ${TIMESTEP_POWER}
  --batch-size N           Inference batch size. Default: ${BATCH_SIZE}
  --load-dtype TYPE        Checkpoint dtype. Default: ${LOAD_DTYPE}
  --infer-dtype TYPE        Inference dtype. Default: ${INFER_DTYPE}
  -h, --help               Show this help.
EOF
}

split_values() {
  local raw="$1"
  local -n out="$2"
  local item

  raw="${raw//，/,}"
  raw="${raw//,/ }"
  out=()
  for item in ${raw}; do
    [[ -n "${item}" ]] && out+=("${item}")
  done

  if [[ "${#out[@]}" -eq 0 ]]; then
    echo "No values parsed from: ${raw}" >&2
    exit 1
  fi
}

tag_num() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
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

merge_shard_manifests() {
  local root="$1"
  python - "${root}" <<'PY'
from collections import defaultdict
from pathlib import Path
import sys

root = Path(sys.argv[1])
groups = defaultdict(list)
for manifest in root.rglob("manifest.shard*-of-*.jsonl"):
    groups[manifest.parent].append(manifest)

for parent, manifests in groups.items():
    merged = parent / "manifest.jsonl"
    with merged.open("w", encoding="utf-8") as fout:
        for manifest in sorted(manifests):
            with manifest.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    print(f"Merged {len(manifests)} shard manifests -> {merged}")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jsonl) JSONL="$2"; shift 2 ;;
    --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
    --model-cfg) MODEL_CFG="$2"; shift 2 ;;
    --ckpt-steps) CKPT_STEPS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --procs-per-gpu) PROCS_PER_GPU="$2"; shift 2 ;;
    --nfe-step) NFE_STEP="$2"; shift 2 ;;
    --cfg-strength) CFG_STRENGTH="$2"; shift 2 ;;
    --shift) SHIFT="$2"; shift 2 ;;
    --timestep-mapping) TIMESTEP_MAPPING="$2"; shift 2 ;;
    --timestep-power) TIMESTEP_POWER="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --load-dtype) LOAD_DTYPE="$2"; shift 2 ;;
    --infer-dtype) INFER_DTYPE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${CKPT_DIR}" ]]; then
  echo "Missing checkpoint. Set CKPT_DIR or EXP_DIR, or pass --ckpt-dir." >&2
  exit 1
fi
if [[ -z "${MODEL_CFG}" ]]; then
  echo "Missing model config. Set MODEL_CFG or EXP_DIR, or pass --model-cfg." >&2
  exit 1
fi
if [[ ! -e "${CKPT_DIR}" ]]; then
  echo "Checkpoint path not found: ${CKPT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_CFG}" ]]; then
  echo "Model config not found: ${MODEL_CFG}" >&2
  exit 1
fi

if ! [[ "${PROCS_PER_GPU}" =~ ^[0-9]+$ ]] || [[ "${PROCS_PER_GPU}" -lt 1 ]]; then
  echo "--procs-per-gpu must be a positive integer, got: ${PROCS_PER_GPU}" >&2
  exit 1
fi

split_values "${CKPT_STEPS}" CKPT_STEP_ARGS
split_values "${GPUS}" GPU_VALUES
split_values "${NFE_STEP}" NFE_VALUES
split_values "${CFG_STRENGTH}" CFG_VALUES
split_values "${SHIFT}" SHIFT_VALUES
split_values "${TIMESTEP_POWER}" POWER_VALUES
TOTAL_CASES=$(( ${#NFE_VALUES[@]} * ${#CFG_VALUES[@]} * ${#SHIFT_VALUES[@]} * ${#POWER_VALUES[@]} ))
TOTAL_WORKERS=$(( ${#GPU_VALUES[@]} * PROCS_PER_GPU ))

echo "JSONL: ${JSONL}"
echo "Checkpoint steps: ${CKPT_STEP_ARGS[*]}"
echo "Output dir: ${OUTPUT_DIR}"
echo "GPUs: ${GPU_VALUES[*]}"
echo "Processes per GPU: ${PROCS_PER_GPU}"
echo "Total worker processes: ${TOTAL_WORKERS}"
echo "NFE steps: ${NFE_VALUES[*]}"
echo "CFG strengths: ${CFG_VALUES[*]}"
echo "Shifts: ${SHIFT_VALUES[*]}"
echo "Timestep powers: ${POWER_VALUES[*]}"
echo "Total inference configs: ${TOTAL_CASES}"

mkdir -p "${OUTPUT_DIR}"
CASE_TSV="${OUTPUT_DIR}/cases.tsv"
printf 'name\tcfg\tnfe\tshift\tpower\toutput_dir\n' > "${CASE_TSV}"

for cfg in "${CFG_VALUES[@]}"; do
  for shift_val in "${SHIFT_VALUES[@]}"; do
    for power in "${POWER_VALUES[@]}"; do
      for nfe in "${NFE_VALUES[@]}"; do
        name="$(case_name "${cfg}" "${shift_val}" "${power}" "${nfe}")"
        case_output_dir="${OUTPUT_DIR}"
        if [[ "${TOTAL_CASES}" -gt 1 ]]; then
          case_output_dir="${OUTPUT_DIR}/${name}"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${cfg}" "${nfe}" "${shift_val}" "${power}" "${case_output_dir}" >> "${CASE_TSV}"

        echo "======== ${name}: cfg=${cfg}, shift=${shift_val}, mapping=${TIMESTEP_MAPPING}, power=${power}, nfe=${nfe}"
        mkdir -p "${case_output_dir}/logs"

        pids=()
        worker_index=0
        for gpu in "${GPU_VALUES[@]}"; do
          for ((proc_idx = 0; proc_idx < PROCS_PER_GPU; proc_idx++)); do
            log_file="${case_output_dir}/logs/worker$(printf '%03d' "${worker_index}")_gpu${gpu}_proc${proc_idx}.log"
            echo "Starting worker ${worker_index}/${TOTAL_WORKERS} on GPU ${gpu}, log: ${log_file}"

            CUDA_VISIBLE_DEVICES="${gpu}" python src/wavtts/eval/infer_xpred_flow.py \
              --ckpt "${CKPT_DIR}" \
              --ckpt-steps "${CKPT_STEP_ARGS[@]}" \
              --jsonl "${JSONL}" \
              --model-cfg "${MODEL_CFG}" \
              --output-dir "${case_output_dir}" \
              --batch-size "${BATCH_SIZE}" \
              --nfe-step "${nfe}" \
              --cfg-strength "${cfg}" \
              --timestep-mapping "${TIMESTEP_MAPPING}" \
              --timestep-power "${power}" \
              --shift "${shift_val}" \
              --num-shards "${TOTAL_WORKERS}" \
              --shard-index "${worker_index}" \
              --load-dtype "${LOAD_DTYPE}" \
              --infer-dtype "${INFER_DTYPE}" \
              "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1 &

            pids+=("$!")
            worker_index=$((worker_index + 1))
          done
        done

        failed=0
        for pid in "${pids[@]}"; do
          if ! wait "${pid}"; then
            failed=1
          fi
        done
        if [[ "${failed}" -ne 0 ]]; then
          echo "One or more workers failed for ${name}. Check logs in ${case_output_dir}/logs" >&2
          exit 1
        fi

        merge_shard_manifests "${case_output_dir}"
      done
    done
  done
done

echo "Wrote config list to ${CASE_TSV}"
