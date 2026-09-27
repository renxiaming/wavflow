#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias for the consolidated token-to-wave inference grid.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export EXP_DIR="${EXP_DIR:-${PROJECT_ROOT}/exp/nar_wav_tts/Emilia_Lance/WavTTS_xpred_spkconcat_dit9-Emilia_Lance-tokentowav-xpred-vloss-spkconcat-repa-dit9-stft-vapa60k-8gpus-10000frames_per_gpu-bf16-node61}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/sim_wer_seed_zh_wavtts3.1_grpo_dit9}"
export CKPT_STEPS="${CKPT_STEPS:-200000,220000}"

exec "${SCRIPT_DIR}/infer.sh" "$@"
