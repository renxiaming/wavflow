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
OUT_ROOT="${OUT_ROOT:-results/xpredflow_grid}"
NFE_STEP="${NFE_STEP:-50}"

run_case() {
  local name="$1"
  local cfg="$2"
  local shift_val="$3"
  local mapping="$4"
  local power="$5"
  local sway="$6"

  echo "======== ${name}: cfg=${cfg}, shift=${shift_val}, mapping=${mapping}, power=${power}, sway=${sway}, nfe=${NFE_STEP}"
  PYTHONPATH=src python src/wavtts/eval/infer_xpred_flow.py \
    --ckptpath "${CKPT_DIR}" \
    --ckpt-steps "${CKPT_STEPS}" \
    --jsonl "${JSONL}" \
    --output-dir "${OUT_ROOT}/${name}" \
    --nfe-step "${NFE_STEP}" \
    --cfg-strength "${cfg}" \
    --timestep-mapping "${mapping}" \
    --timestep-power "${power}" \
    --sway-sampling-coef "${sway}" \
    --shift "${shift_val}"
}

run_case cfg0p5_shift3_power2_nfe50 0.5 3.0 power 2.0 -1.0
run_case cfg1_shift1_power2_nfe50 1.0 1.0 power 2.0 -1.0
run_case cfg1_shift2_power2_nfe50 1.0 2.0 power 2.0 -1.0
run_case cfg1_shift3_power2_nfe50 1.0 3.0 power 2.0 -1.0
run_case cfg1p5_shift3_power2_nfe50 1.5 3.0 power 2.0 -1.0
run_case cfg1_shift3_sway-1_nfe50 1.0 3.0 sway_sampling 2.0 -1.0
