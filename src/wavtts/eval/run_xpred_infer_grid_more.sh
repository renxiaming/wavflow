#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/exp/nar_wav_tts/Emilia_Lance/WavTTS_xpred_spkconcat-Emilia_Lance-tokentowav-xpred-vloss-spkconcat-repa-dit20-stft-vapa60k-8gpus-10000frames_per_gpu-bf16/ckpts}"
CKPT_STEPS="${CKPT_STEPS:-200000}"
JSONL="${JSONL:-data/emilia_lance_token_test_zh_en_200/test.jsonl}"
OUT_ROOT="${OUT_ROOT:-results/xpredflow_grid_more}"
SHIFT="${SHIFT:-2.0}"
TIMESTEP_MAPPING="${TIMESTEP_MAPPING:-power}"
TIMESTEP_POWER="${TIMESTEP_POWER:-2.0}"
SWAY_SAMPLING_COEF="${SWAY_SAMPLING_COEF:--1.0}"

cfg_tag() {
  local value="$1"
  value="${value/./p}"
  value="${value%p0}"
  echo "${value}"
}

num_tag() {
  local value="$1"
  value="${value/./p}"
  value="${value%p0}"
  echo "${value}"
}

run_case() {
  local cfg="$1"
  local nfe="$2"
  local cfg_name
  cfg_name="$(cfg_tag "${cfg}")"
  local shift_name
  local power_name
  shift_name="$(num_tag "${SHIFT}")"
  power_name="$(num_tag "${TIMESTEP_POWER}")"
  local name="cfg${cfg_name}_shift${shift_name}_power${power_name}_nfe${nfe}"

  echo "======== ${name}: cfg=${cfg}, shift=${SHIFT}, mapping=${TIMESTEP_MAPPING}, power=${TIMESTEP_POWER}, sway=${SWAY_SAMPLING_COEF}, nfe=${nfe}"
  PYTHONPATH=src python src/wavtts/eval/infer_xpred_flow.py \
    --ckptpath "${CKPT_DIR}" \
    --ckpt-steps "${CKPT_STEPS}" \
    --jsonl "${JSONL}" \
    --output-dir "${OUT_ROOT}/${name}" \
    --nfe-step "${nfe}" \
    --cfg-strength "${cfg}" \
    --timestep-mapping "${TIMESTEP_MAPPING}" \
    --timestep-power "${TIMESTEP_POWER}" \
    --sway-sampling-coef "${SWAY_SAMPLING_COEF}" \
    --shift "${SHIFT}"
}

cfgs=(0.5 1.0 1.5 2.0)
nfe_steps=(25 50 100)

for nfe in "${nfe_steps[@]}"; do
  for cfg in "${cfgs[@]}"; do
    run_case "${cfg}" "${nfe}"
  done
done
