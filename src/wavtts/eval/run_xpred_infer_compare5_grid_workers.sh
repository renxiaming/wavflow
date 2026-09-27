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
CKPT_STEPS="${CKPT_STEPS:-120000}"
JSONL="${JSONL:-data/emilia_lance_token_test_zh_en_200/test5.jsonl}"
OUT_ROOT="${OUT_ROOT:-results/fp32}"
DEVICE_LIST_STR="${DEVICE_LIST:-0 1 2 3 4 5 6 7}"

CFG_VALUES=(${CFG_VALUES:-0 0.5 1.0 1.5 2.0})
SHIFT_VALUES=(${SHIFT_VALUES:-0 0.5 1.0 1.5 2.0})
POWER_VALUES=(${POWER_VALUES:-0 0.5 1.0 1.5 2.0})
NFE_VALUES=(${NFE_VALUES:-15 20 25 30 35 50})

TIMESTEP_MAPPING="${TIMESTEP_MAPPING:-power}"
SWAY_SAMPLING_COEF="${SWAY_SAMPLING_COEF:--1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOAD_DTYPE="${LOAD_DTYPE:-fp32}"
INFER_DTYPE="${INFER_DTYPE:-fp32}"
SEED="${SEED:-}"

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

mkdir -p "${OUT_ROOT}/logs"
CASE_TSV="${OUT_ROOT}/cases.tsv"
printf 'name\tcfg\tnfe\tshift\tpower\n' > "${CASE_TSV}"
for cfg in "${CFG_VALUES[@]}"; do
  for shift_val in "${SHIFT_VALUES[@]}"; do
    for power in "${POWER_VALUES[@]}"; do
      for nfe in "${NFE_VALUES[@]}"; do
        name="$(case_name "${cfg}" "${shift_val}" "${power}" "${nfe}")"
        printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${cfg}" "${nfe}" "${shift_val}" "${power}" >> "${CASE_TSV}"
      done
    done
  done
done

devices=(${DEVICE_LIST_STR})
num_shards="${#devices[@]}"
echo "Prepared $(($(wc -l < "${CASE_TSV}") - 1)) config cases under ${OUT_ROOT}"
echo "Launching ${num_shards} workers on devices: ${DEVICE_LIST_STR}"

pids=()
for shard in "${!devices[@]}"; do
  gpu="${devices[${shard}]}"
  log_path="${OUT_ROOT}/logs/grid_worker_${shard}_gpu${gpu}.log"
  echo "worker ${shard}/${num_shards} -> gpu ${gpu}, log ${log_path}"
  seed_args=()
  if [[ -n "${SEED}" ]]; then
    seed_args=(--seed "${SEED}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src python src/wavtts/eval/infer_xpred_flow_grid_worker.py \
    --ckptpath "${CKPT_DIR}" \
    --ckpt-steps ${CKPT_STEPS} \
    --jsonl "${JSONL}" \
    --cases-tsv "${CASE_TSV}" \
    --output-root "${OUT_ROOT}" \
    --shard-index "${shard}" \
    --num-shards "${num_shards}" \
    --batch-size "${BATCH_SIZE}" \
    --timestep-mapping "${TIMESTEP_MAPPING}" \
    --sway-sampling-coef "${SWAY_SAMPLING_COEF}" \
    --load-dtype "${LOAD_DTYPE}" \
    --infer-dtype "${INFER_DTYPE}" \
    "${seed_args[@]}" \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
