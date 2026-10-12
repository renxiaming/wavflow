#!/usr/bin/env bash
set -euo pipefail

source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

GRPO_WEIGHT_ROOT="${PROJECT_ROOT}/portable_wavtts_grpo/pretrain_models/grpo_rewards"
if [[ -d "${GRPO_WEIGHT_ROOT}/SpeechMOS" ]]; then
  export UTMOS_HUB_REPO="${UTMOS_HUB_REPO:-${GRPO_WEIGHT_ROOT}/SpeechMOS}"
fi
if [[ -f "${GRPO_WEIGHT_ROOT}/utmos/utmos22_strong_step7459_v1.pt" ]]; then
  export UTMOS_CHECKPOINT="${UTMOS_CHECKPOINT:-${GRPO_WEIGHT_ROOT}/utmos/utmos22_strong_step7459_v1.pt}"
fi

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
REWARD_DEVICE="${GRPO_REWARD_DEVICE:-}"
REWARD_SERVER_HOST="${GRPO_REWARD_SERVER_HOST:-127.0.0.1}"
REWARD_SERVER_PORT="${GRPO_REWARD_SERVER_PORT:-29631}"
BACKEND=""
WER_WEIGHT=""
SIM_WEIGHT=""
MOS_WEIGHT=""
REWARD_ROOT=""
MODEL_CFG="${PROJECT_ROOT}/src/wavtts/configs/WavTTS_xpred_spkconcat_flow_grpo_promptwav.yaml"
INIT_CKPT="${GRPO_INIT_CKPT:-}"
JSONL="${GRPO_JSONL:-}"
OUTPUT_DIR="${GRPO_OUTPUT_DIR:-}"
KEEP_ROLLOUTS="${GRPO_KEEP_ROLLOUTS:-1}"
DATA_MODE="${GRPO_DATA_MODE:-self_prefix}"

usage() {
  cat <<EOF
Usage:
  bash src/wavtts/train/run_grpo_token_xpred_promptwav.sh [options]

Options:
  --gpus IDS             Training GPU(s), default: ${GPUS}
  --reward-device DEV    Dedicated reward GPU; only required by server backend
  --reward-server-host H Reward server host, default: ${REWARD_SERVER_HOST}
  --reward-server-port P Reward server port, default: ${REWARD_SERVER_PORT}
  --sim-device DEV       Single-process SIM device; DDP resident mode uses each local GPU
  --wer-device DEV       Single-process WER device; DDP resident mode uses each local GPU
  --mos-device DEV       Single-process MOS device; DDP resident mode uses each local GPU
  --model-cfg PATH       GRPO/SFT Hydra config
  --model-dtype TYPE     Persistent parameter dtype; GRPO requires fp32
  --compute-dtype TYPE   Rollout/replay autocast dtype, default from config
  --init-ckpt PATH       SFT checkpoint, default: grpo.init_ckpt in config
  --jsonl PATH           Prompt/target JSONL, default: grpo.jsonl in config
  --output-dir DIR       GRPO checkpoint directory, default: grpo.output_dir in config
  --group-size N         Samples per prompt, default from config
  --prompts-per-update N Distinct prompts collected before one PPO update
  --batch-size-frames N  Per-GPU rollout/replay frame budget, default from config
  --data-mode MODE       paired or self_prefix, default: paired/config
  --prompt-fraction-min X Minimum self-prefix fraction, default from config
  --prompt-fraction-max X Maximum self-prefix fraction, default from config
  --steps N              SDE rollout steps, default from config
  --rollout-steps-min N  Random minimum rollout steps; 0 disables randomization
  --sde-window-size N    Number of adjacent stochastic/PPO steps
  --sde-window-start-min N Minimum random SDE-window start index
  --sde-window-start-max N Maximum random SDE-window start index
  --max-updates N        Training updates, default from config
  --learning-rate X      Optimizer learning rate, default from config
  --lr-scheduler NAME    constant or linear, default from config
  --max-grad-norm X      Gradient clipping norm, default from config
  --ppo-epochs N         PPO epochs per rollout batch, default from config
  --train-timestep-fraction X Fraction of rollout timesteps used for PPO replay
  --clip-eps X           PPO clipping range, default from config
  --adv-clip-max X       Advantage clipping maximum, <=0 disables it
  --kl-coef X            Reference KL coefficient, default from config
  --noise-scale X        Fixed SDE noise scale, default from config
  --use-cfg              Enable CFG (paper default)
  --no-cfg               Disable CFG; forces cfg strength to 1.0
  --cfg-strength X       CFG strength, default from config
  --timestep-mapping M   uniform, power, or sway_sampling; default from config
  --timestep-power X     Power exponent for --timestep-mapping power
  --sway-sampling-coef X Sway sampling coefficient
  --shift X              Timestep shift, default from config
  --use-lora              Train LoRA adapters (full-parameter training is default)
  --no-lora               Full-parameter fine-tuning
  --lora-rank N          LoRA rank; 0 switches back to full fine-tuning
  --lora-alpha X         LoRA alpha, default from config
  --lora-dropout X       LoRA dropout, default from config
  --lora-targets LIST    Comma-separated LoRA target suffixes
  --wer-weight X         WER reward weight, default from config
  --sim-weight X         Speaker similarity reward weight, default from config
  --mos-weight X         Perceptual MOS reward weight, default from config
  --reward-std-eps X     Minimum component std before it contributes
  --sft-anchor-coef X    Prompt waveform preservation loss coefficient
  --target-rms X         Normalize low-RMS prompt audio, default from config
  --reward-language L    zh, en, or auto; default from config
  --reward-root DIR      SIM/WER project root, default from config or local path
  --max-rows N            Debug with only N JSONL rows
  --reward-backend NAME   resident (default from config), server, script, or direct
  --reward-work-dir DIR   Keep temporary reward files under DIR
  --keep-reward-work      Keep temporary reward files for debugging
  --keep-rollouts         Keep rollout wavs and append aggregate reward JSONL (default)
  --discard-rollouts      Delete rollout wavs after reward scoring; keep JSONL records
  --sim-python PATH       Python executable for resident SIM worker
  --wer-python PATH       Python executable for resident WER worker
  --mos-python PATH       Python executable for resident UTMOS worker
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
    --mos-device) EXTRA_ARGS+=(--mos-device "$2"); shift 2 ;;
    --model-cfg) MODEL_CFG="$2"; shift 2 ;;
    --model-dtype) EXTRA_ARGS+=(--model-dtype "$2"); shift 2 ;;
    --compute-dtype) EXTRA_ARGS+=(--compute-dtype "$2"); shift 2 ;;
    --init-ckpt) INIT_CKPT="$2"; shift 2 ;;
    --jsonl) JSONL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --group-size) EXTRA_ARGS+=(--group-size "$2"); shift 2 ;;
    --prompts-per-update) EXTRA_ARGS+=(--prompts-per-update "$2"); shift 2 ;;
    --batch-size-frame|--batch-size-frames|--batch-size-per-gpu)
      EXTRA_ARGS+=(--batch-size-frames "$2"); shift 2 ;;
    --data-mode) DATA_MODE="$2"; shift 2 ;;
    --prompt-fraction-min) EXTRA_ARGS+=(--prompt-fraction-min "$2"); shift 2 ;;
    --prompt-fraction-max) EXTRA_ARGS+=(--prompt-fraction-max "$2"); shift 2 ;;
    --steps) EXTRA_ARGS+=(--steps "$2"); shift 2 ;;
    --rollout-steps-min) EXTRA_ARGS+=(--rollout-steps-min "$2"); shift 2 ;;
    --sde-window-size) EXTRA_ARGS+=(--sde-window-size "$2"); shift 2 ;;
    --sde-window-start-min) EXTRA_ARGS+=(--sde-window-start-min "$2"); shift 2 ;;
    --sde-window-start-max) EXTRA_ARGS+=(--sde-window-start-max "$2"); shift 2 ;;
    --max-updates) EXTRA_ARGS+=(--max-updates "$2"); shift 2 ;;
    --learning-rate) EXTRA_ARGS+=(--learning-rate "$2"); shift 2 ;;
    --lr-scheduler) EXTRA_ARGS+=(--lr-scheduler "$2"); shift 2 ;;
    --max-grad-norm) EXTRA_ARGS+=(--max-grad-norm "$2"); shift 2 ;;
    --ppo-epochs) EXTRA_ARGS+=(--ppo-epochs "$2"); shift 2 ;;
    --train-timestep-fraction) EXTRA_ARGS+=(--train-timestep-fraction "$2"); shift 2 ;;
    --clip-eps) EXTRA_ARGS+=(--clip-eps "$2"); shift 2 ;;
    --adv-clip-max) EXTRA_ARGS+=(--adv-clip-max "$2"); shift 2 ;;
    --kl-coef) EXTRA_ARGS+=(--kl-coef "$2"); shift 2 ;;
    --noise-scale) EXTRA_ARGS+=(--noise-scale "$2"); shift 2 ;;
    --use-cfg) EXTRA_ARGS+=(--use-cfg); shift ;;
    --no-cfg) EXTRA_ARGS+=(--no-cfg); shift ;;
    --cfg-strength) EXTRA_ARGS+=(--cfg-strength "$2"); shift 2 ;;
    --timestep-mapping) EXTRA_ARGS+=(--timestep-mapping "$2"); shift 2 ;;
    --timestep-power) EXTRA_ARGS+=(--timestep-power "$2"); shift 2 ;;
    --sway-sampling-coef) EXTRA_ARGS+=(--sway-sampling-coef "$2"); shift 2 ;;
    --shift) EXTRA_ARGS+=(--shift "$2"); shift 2 ;;
    --use-lora) EXTRA_ARGS+=(--use-lora); shift ;;
    --no-lora) EXTRA_ARGS+=(--no-lora); shift ;;
    --lora-rank) EXTRA_ARGS+=(--lora-rank "$2"); shift 2 ;;
    --lora-alpha) EXTRA_ARGS+=(--lora-alpha "$2"); shift 2 ;;
    --lora-dropout) EXTRA_ARGS+=(--lora-dropout "$2"); shift 2 ;;
    --lora-targets) EXTRA_ARGS+=(--lora-targets "$2"); shift 2 ;;
    --wer-weight) WER_WEIGHT="$2"; EXTRA_ARGS+=(--wer-weight "$2"); shift 2 ;;
    --sim-weight) SIM_WEIGHT="$2"; EXTRA_ARGS+=(--sim-weight "$2"); shift 2 ;;
    --mos-weight) MOS_WEIGHT="$2"; EXTRA_ARGS+=(--mos-weight "$2"); shift 2 ;;
    --reward-std-eps) EXTRA_ARGS+=(--reward-std-eps "$2"); shift 2 ;;
    --sft-anchor-coef) EXTRA_ARGS+=(--sft-anchor-coef "$2"); shift 2 ;;
    --target-rms) EXTRA_ARGS+=(--target-rms "$2"); shift 2 ;;
    --reward-language) EXTRA_ARGS+=(--reward-language "$2"); shift 2 ;;
    --reward-root) REWARD_ROOT="$2"; EXTRA_ARGS+=(--reward-root "$2"); shift 2 ;;
    --max-rows) EXTRA_ARGS+=(--max-rows "$2"); shift 2 ;;
    --reward-backend) BACKEND="$2"; shift 2 ;;
    --reward-work-dir) EXTRA_ARGS+=(--reward-work-dir "$2"); shift 2 ;;
    --keep-reward-work) EXTRA_ARGS+=(--keep-reward-work); shift ;;
    --keep-rollouts) KEEP_ROLLOUTS=1; shift ;;
    --discard-rollouts) KEEP_ROLLOUTS=0; shift ;;
    --sim-python) EXTRA_ARGS+=(--sim-python "$2"); shift 2 ;;
    --wer-python) EXTRA_ARGS+=(--wer-python "$2"); shift 2 ;;
    --mos-python) EXTRA_ARGS+=(--mos-python "$2"); shift 2 ;;
    --no-reference) EXTRA_ARGS+=(--no-reference); shift ;;
    --with-reference) EXTRA_ARGS+=(--with-reference); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${KEEP_ROLLOUTS}" != "0" && "${KEEP_ROLLOUTS}" != "1" ]]; then
  echo "GRPO_KEEP_ROLLOUTS must be 0 or 1, got: ${KEEP_ROLLOUTS}" >&2
  exit 1
fi
if [[ "${KEEP_ROLLOUTS}" == "1" ]]; then
  EXTRA_ARGS+=(--keep-rollouts)
else
  EXTRA_ARGS+=(--discard-rollouts)
fi

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
    ("grpo.wer_weight", 0.2),
    ("grpo.sim_weight", 0.8),
    ("grpo.mos_weight", 0.0),
    ("grpo.reward_root", ""),
    ("grpo.data_mode", "paired"),
):
    value = OmegaConf.select(cfg, key, default=default)
    print("" if value is None else value)
PY
)
[[ -z "${INIT_CKPT}" ]] && INIT_CKPT="${CFG_DEFAULTS[0]}"
[[ -z "${JSONL}" ]] && JSONL="${CFG_DEFAULTS[1]}"
[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${CFG_DEFAULTS[2]}"
[[ -z "${WER_WEIGHT}" ]] && WER_WEIGHT="${CFG_DEFAULTS[3]}"
[[ -z "${SIM_WEIGHT}" ]] && SIM_WEIGHT="${CFG_DEFAULTS[4]}"
[[ -z "${MOS_WEIGHT}" ]] && MOS_WEIGHT="${CFG_DEFAULTS[5]}"
[[ -z "${REWARD_ROOT}" ]] && REWARD_ROOT="${CFG_DEFAULTS[6]}"
[[ -z "${DATA_MODE}" ]] && DATA_MODE="${CFG_DEFAULTS[7]}"

case "${DATA_MODE}" in
  paired|self_prefix) ;;
  *) echo "data mode must be paired or self_prefix, got: ${DATA_MODE}" >&2; exit 1 ;;
esac
EXTRA_ARGS+=(--data-mode "${DATA_MODE}")

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
      --mos-device cuda:0 \
      --wer-weight "${WER_WEIGHT}" \
      --sim-weight "${SIM_WEIGHT}" \
      --mos-weight "${MOS_WEIGHT}" \
      --sim-root "${REWARD_ROOT}" \
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
