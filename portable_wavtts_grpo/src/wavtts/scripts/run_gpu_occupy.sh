#!/usr/bin/env bash
set -euo pipefail

# Occupy 8 GPUs with dynamic utilization. Process name shows as cosyvoice2 in ps/nvidia-smi.
#
# Usage:
#   bash src/wavtts/scripts/run_gpu_occupy.sh
#   bash src/wavtts/scripts/run_gpu_occupy.sh --gpus 0,1,2,3,4,5,6,7 --vram-gib 22 --process-name cosyvoice2
#   nohup bash src/wavtts/scripts/run_gpu_occupy.sh > /tmp/gpu_occupy.log 2>&1 &

source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS="0,1,2,3,4,5,6,7"
VRAM_GIB="20"
PROCESS_NAME="cosyvoice2"
MIN_UTIL="0.15"
MAX_UTIL="0.85"
CYCLE_SEC="8"

usage() {
  cat <<EOF
Usage: bash src/wavtts/scripts/run_gpu_occupy.sh [options]

Options:
  --gpus IDS            GPU ids, default: ${GPUS}
  --vram-gib N          VRAM per GPU in GiB, default: ${VRAM_GIB}
  --process-name NAME   Process name in ps/nvidia-smi, default: ${PROCESS_NAME}
  --min-util F          Min utilization target [0-1], default: ${MIN_UTIL}
  --max-util F          Max utilization target [0-1], default: ${MAX_UTIL}
  --cycle-sec N         Utilization oscillation period, default: ${CYCLE_SEC}
  -h, --help            Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --vram-gib) VRAM_GIB="$2"; shift 2 ;;
    --process-name) PROCESS_NAME="$2"; shift 2 ;;
    --min-util) MIN_UTIL="$2"; shift 2 ;;
    --max-util) MAX_UTIL="$2"; shift 2 ;;
    --cycle-sec) CYCLE_SEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python src/wavtts/scripts/gpu_occupy.py \
  --gpus "${GPUS}" \
  --vram-gib "${VRAM_GIB}" \
  --process-name "${PROCESS_NAME}" \
  --min-util "${MIN_UTIL}" \
  --max-util "${MAX_UTIL}" \
  --cycle-sec "${CYCLE_SEC}"
