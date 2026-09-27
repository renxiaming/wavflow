#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export MODEL_CFG="${MODEL_CFG:-${PROJECT_ROOT}/src/wavtts/configs/WavTTS_xpred_spkconcat_flow_grpo.yaml}"
export JSONL="${JSONL:-${PROJECT_ROOT}/data/seed_infer/zh/seedtest_zh_cosyvoice2_tokens_infer.jsonl}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/result/grpo}"
export CKPT_STEPS="${CKPT_STEPS:-last}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/grpo_infer}"
export SIM_OUTPUT_DIR="${SIM_OUTPUT_DIR:-${PROJECT_ROOT}/results/sim/zh}"
export DATA_MODE="paired"
export LANGUAGE="${LANGUAGE:-zh}"

exec bash "${SCRIPT_DIR}/infer_grpo.sh" "$@"
