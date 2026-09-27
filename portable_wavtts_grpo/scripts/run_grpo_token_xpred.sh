#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GRPO_CONDA_SH:-}" ]]; then
  # Example:
  #   export GRPO_CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
  #   export GRPO_CONDA_ENV=wavtts
  source "${GRPO_CONDA_SH}"
  conda activate "${GRPO_CONDA_ENV:-wavtts}"
else
  source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
REWARD_DEVICE="${GRPO_REWARD_DEVICE:-}"
REWARD_SERVER_HOST="${GRPO_REWARD_SERVER_HOST:-127.0.0.1}"
REWARD_SERVER_PORT="${GRPO_REWARD_SERVER_PORT:-29631}"
BACKEND=""
MODEL_CFG="${PROJECT_ROOT}/configs/WavTTS_xpred_spkconcat_grpo.yaml"
INIT_CKPT="${GRPO_INIT_CKPT:-}"
JSONL="${GRPO_JSONL:-}"
OUTPUT_DIR="${GRPO_OUTPUT_DIR:-}"

usage() {
  cat <<EOF
Usage:
  bash src/wavtts/train/run_grpo_token_xpred.sh [options]

Options:
  --gpus IDS             Training GPU(s), default: ${GPUS}
  --reward-device DEV    Dedicated reward GPU; only required by server backend
  --reward-server-host H Reward server host, default: ${REWARD_SERVER_HOST}
  --reward-server-port P Reward server port, default: ${REWARD_SERVER_PORT}
  --sim-device DEV       Single-process SIM device; DDP resident mode uses each local GPU
  --wer-device DEV       Single-process WER device; DDP resident mode uses each local GPU
  --model-cfg PATH       GRPO/SFT Hydra config
  --init-ckpt PATH       SFT checkpoint, default: grpo.init_ckpt in config
  --jsonl PATH           Prompt/target JSONL, default: grpo.jsonl in config
  --output-dir DIR       GRPO checkpoint directory, default: grpo.output_dir in config
  --group-size N         Samples per prompt, default from config
  --steps N              SDE rollout steps, default from config
  --max-updates N        Training updates, default from config
  --learning-rate X      Optimizer learning rate, default from config
  --max-grad-norm X      Gradient clipping norm, default from config
  --ppo-epochs N         PPO epochs per rollout batch, default from config
  --clip-eps X           PPO clipping range, default from config
  --kl-coef X            Reference KL coefficient, default from config
  --noise-scale X        Fixed SDE noise scale, default from config
  --cfg-strength X       CFG strength, default from config
  --shift X              Timestep shift, default from config
  --lora-rank N          LoRA rank; 0 switches back to full fine-tuning
  --lora-alpha X         LoRA alpha, default from config
  --lora-dropout X       LoRA dropout, default from config
  --lora-targets LIST    Comma-separated LoRA target suffixes
  --wer-weight X         WER reward weight, default from config
  --sim-weight X         Speaker similarity reward weight, default from config
  --reward-language L    zh, en, or auto; default from config
  --reward-root DIR      SIM/WER project root, default from config or local path
  --max-rows N            Debug with only N JSONL rows
  --reward-backend NAME   resident (default from config), server, script, or direct
  --reward-work-dir DIR   Keep temporary reward files under DIR
  --keep-reward-work      Keep temporary reward files for debugging
  --keep-rollouts         Keep generated rollout wav files under output_dir/rollouts
  --sim-python PATH       Python executable for resident SIM worker
  --wer-python PATH       Python executable for resident WER worker
  --no-reference          Keep only one policy/forward model per GPU
  --with-reference        Add one frozen reference model per GPU
  -h, --help              Show this help
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --reward-device) REWARD_DEVICE="$2"; shift 2 ;;
    --reward-server-host) REWARD_SERVER_HOST="$2"; shift 2 ;;
    --reward-server-port) REWARD_SERVER_PORT="$2"; shift 2 ;;
    --sim-device) EXTRA_ARGS+=(--sim-device "$2"); shift 2 ;;
    --wer-device) EXTRA_ARGS+=(--wer-device "$2"); shift 2 ;;
    --model-cfg) MODEL_CFG="$2"; shift 2 ;;
    --init-ckpt) INIT_CKPT="$2"; shift 2 ;;
    --jsonl) JSONL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --group-size) EXTRA_ARGS+=(--group-size "$2"); shift 2 ;;
    --steps) EXTRA_ARGS+=(--steps "$2"); shift 2 ;;
    --max-updates) EXTRA_ARGS+=(--max-updates "$2"); shift 2 ;;
    --learning-rate) EXTRA_ARGS+=(--learning-rate "$2"); shift 2 ;;
    --max-grad-norm) EXTRA_ARGS+=(--max-grad-norm "$2"); shift 2 ;;
    --ppo-epochs) EXTRA_ARGS+=(--ppo-epochs "$2"); shift 2 ;;
    --clip-eps) EXTRA_ARGS+=(--clip-eps "$2"); shift 2 ;;
    --kl-coef) EXTRA_ARGS+=(--kl-coef "$2"); shift 2 ;;
    --noise-scale) EXTRA_ARGS+=(--noise-scale "$2"); shift 2 ;;
    --cfg-strength) EXTRA_ARGS+=(--cfg-strength "$2"); shift 2 ;;
    --shift) EXTRA_ARGS+=(--shift "$2"); shift 2 ;;
    --lora-rank) EXTRA_ARGS+=(--lora-rank "$2"); shift 2 ;;
    --lora-alpha) EXTRA_ARGS+=(--lora-alpha "$2"); shift 2 ;;
    --lora-dropout) EXTRA_ARGS+=(--lora-dropout "$2"); shift 2 ;;
    --lora-targets) EXTRA_ARGS+=(--lora-targets "$2"); shift 2 ;;
    --wer-weight) EXTRA_ARGS+=(--wer-weight "$2"); shift 2 ;;
    --sim-weight) EXTRA_ARGS+=(--sim-weight "$2"); shift 2 ;;
    --reward-language) EXTRA_ARGS+=(--reward-language "$2"); shift 2 ;;
    --reward-root) EXTRA_ARGS+=(--reward-root "$2"); shift 2 ;;
    --max-rows) EXTRA_ARGS+=(--max-rows "$2"); shift 2 ;;
    --reward-backend) BACKEND="$2"; shift 2 ;;
    --reward-work-dir) EXTRA_ARGS+=(--reward-work-dir "$2"); shift 2 ;;
    --keep-reward-work) EXTRA_ARGS+=(--keep-reward-work); shift ;;
    --keep-rollouts) EXTRA_ARGS+=(--keep-rollouts); shift ;;
    --sim-python) EXTRA_ARGS+=(--sim-python "$2"); shift 2 ;;
    --wer-python) EXTRA_ARGS+=(--wer-python "$2"); shift 2 ;;
    --no-reference) EXTRA_ARGS+=(--no-reference); shift ;;
    --with-reference) EXTRA_ARGS+=(--with-reference); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${BACKEND}" ]]; then
  BACKEND="$(
    python - "${MODEL_CFG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
print(OmegaConf.select(cfg, "grpo.reward_backend", default="resident"))
PY
  )"
fi

mapfile -t CFG_DEFAULTS < <(
  python - "${MODEL_CFG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
for key, default in (
    ("grpo.init_ckpt", ""),
    ("grpo.jsonl", ""),
    ("grpo.output_dir", "./exp/nar_wav_tts_grpo"),
):
    value = OmegaConf.select(cfg, key, default=default)
    print("" if value is None else value)
PY
)
[[ -z "${INIT_CKPT}" ]] && INIT_CKPT="${CFG_DEFAULTS[0]}"
[[ -z "${JSONL}" ]] && JSONL="${CFG_DEFAULTS[1]}"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${CFG_DEFAULTS[2]}"

if [[ "${BACKEND}" == "server" ]]; then
  if [[ -z "${REWARD_DEVICE}" ]]; then
    echo "server backend requires --reward-device with a GPU excluded from --gpus" >&2
    exit 1
  fi
  REWARD_GPU="${REWARD_DEVICE#cuda:}"
  if [[ ! "${REWARD_GPU}" =~ ^[0-9]+$ ]]; then
    echo "--reward-device must be a physical GPU id such as 3 or cuda:3" >&2
    exit 1
  fi
  if [[ ",${GPUS}," == *",${REWARD_GPU},"* ]]; then
    echo "Reward GPU ${REWARD_GPU} is also in --gpus=${GPUS}; exclude it from training GPUs." >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${BACKEND}" == "server" ]]; then
  env CUDA_VISIBLE_DEVICES="${REWARD_GPU}" PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m wavtts.rewards.reward_server \
      --host "${REWARD_SERVER_HOST}" \
      --port "${REWARD_SERVER_PORT}" \
      --sim-device cuda:0 \
      --wer-device cuda:0 \
      >"${OUTPUT_DIR}/reward_server.log" 2>&1 &
  SERVER_PID=$!
  READY=0
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      cat "${OUTPUT_DIR}/reward_server.log" >&2 || true
      exit 1
    fi
    if python - "${REWARD_SERVER_HOST}" "${REWARD_SERVER_PORT}" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
raise SystemExit(0)
PY
    then
      READY=1
      break
    fi
    sleep 1
  done
  if [[ "${READY}" -ne 1 ]]; then
    echo "Reward server did not become ready on ${REWARD_SERVER_HOST}:${REWARD_SERVER_PORT}" >&2
    cat "${OUTPUT_DIR}/reward_server.log" >&2 || true
    exit 1
  fi
fi

TRAIN_ARGS=(
  --model-cfg "${MODEL_CFG}"
  --device cuda:0
  --reward-backend "${BACKEND}"
  --reward-server-host "${REWARD_SERVER_HOST}"
  --reward-server-port "${REWARD_SERVER_PORT}"
  "${EXTRA_ARGS[@]}"
)

if [[ -n "${INIT_CKPT}" ]]; then
  TRAIN_ARGS+=(--init-ckpt "${INIT_CKPT}")
fi
if [[ -n "${JSONL}" ]]; then
  TRAIN_ARGS+=(--jsonl "${JSONL}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  TRAIN_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${REWARD_DEVICE}" ]]; then
  TRAIN_ARGS+=(--reward-device "${REWARD_DEVICE}")
fi

NPROC="$(awk -F',' '{print NF}' <<< "${GPUS}")"
if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC}" \
    -m wavtts.train.train_grpo "${TRAIN_ARGS[@]}"
else
  python -m wavtts.train.train_grpo \
    "${TRAIN_ARGS[@]}"
fi
