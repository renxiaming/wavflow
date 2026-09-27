#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
PYTHONPATH="${PROJECT_ROOT}/src" python src/wavtts/eval/infer_xpred_flow.py \
  --ckptpath "${PROJECT_ROOT}/exp/nar_wav_tts/Emilia_Lance/WavTTS_xpred_spkconcat-Emilia_Lance-tokentowav-xpred-vloss-spkconcat-repa-dit20-stft-vapa60k-8gpus-10000frames_per_gpu-bf16/ckpts" \
  --ckpt-steps 200000 \
  --jsonl data/emilia_lance_token_test_zh_en_200/test.jsonl \
  --output-dir results/xpredflow_grid/cfg1p5_shift3_power2_nfe50 \
  --nfe-step 50 \
  --cfg-strength 1.5 \
  --timestep-mapping power \
  --timestep-power 2.0 \
  --shift 3.0
