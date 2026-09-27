#!/usr/bin/env bash
set -euo pipefail

# Full-scale Emilia token-to-wave x-pred/v-loss training with online speaker
# embedding, REPA alignment loss, and late-start STFT/VAPA loss.
# Speaker embedding is fused CosyVoice-style as a frame-wise input condition:
# x_h + cond_h + token_h + spk_h -> fuse.
#
# Usage:
#   bash src/wavtts/train/run_main_train_token_xpred_spkconcat.sh --gpus 4,5,6,7
#   bash src/wavtts/train/run_main_train_token_xpred_spkconcat.sh --gpus 4,5 --batch-size-per-gpu 10000 --num-workers 1
source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_NAME="WavTTS_xpred_spkconcat"
FLOW_VARIANT="xpred-vloss-spkconcat-repa-dit20-stft-vapa60k"
DATASET_NAME="Emilia_Lance"
DATASET_TYPE="EmiliaLance"
LANCE_PATH="/home/work_nfs23/xmren/lance/emilia_wav_token"
TEXT_INDEX_PATH="/home/work_nfs23/xmren/lance/emilia_text.db"
TEXT_JSONL_PATH="/home/work_nfs23/hkxie/data/Emilia/emilia_merged.jsonl"
VOCAB_PATH="${PROJECT_ROOT}/data/Emilia_Lance_pinyin/vocab.txt"
SPEECH_TOKEN_NUM_EMBEDS="6561"
CAMPPLUS_ONNX="/home/work_nfs23/xmren/huawei_proj/huawei_streaming_cosyvoice/pretrain_models/CosyVoice2-0.5B/campplus.onnx"
REPA_TEACHER_MODEL="${WAVTTS_REPA_TEACHER_MODEL:-/home/work_nfs20/xmren/code/wavtts3/pretrain_models/wavlm-base-plus}"
# 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
# Set CUDA_VISIBLE_DEVICES before invoking this script or pass --gpus explicitly.
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES=""
NUM_MACHINES="1"
MIXED_PRECISION="bf16"
# REPA keeps a WavLM graph/activation buffer for the generator branch.  The
# default is half of the old frame budget; override with --batch-size-per-gpu
# after checking peak memory on the selected GPUs.
BATCH_SIZE_PER_GPU="10000"
NUM_WORKERS="1"
MASTER_PORT="49603"
OUTPUT_ROOT="./exp/nar_wav_tts"

usage() {
  cat <<EOF
Usage: bash src/wavtts/train/run_main_train_token_xpred_spkconcat.sh [options]

Full Emilia Lance token-to-wave xpred/vloss spk-concat training with REPA and
STFT/VAPA losses (~40M zh+en).

Options:
  --gpus IDS              Comma-separated GPU ids, e.g. 4,5,6,7. Required unless CUDA_VISIBLE_DEVICES is already set.
  --num-processes N       Number of training processes (default: number of GPUs)
  --batch-size-per-gpu N  Frame batch size per GPU (default: ${BATCH_SIZE_PER_GPU})
  --num-workers N         DataLoader workers per GPU (default: ${NUM_WORKERS})
  --master-port PORT      Accelerate main process port (default: ${MASTER_PORT})
  --output-root DIR       Output root (default: ${OUTPUT_ROOT})
  --campplus-onnx PATH    CAM++ ONNX for online speaker embedding (default: ${CAMPPLUS_ONNX})
  --repa-teacher-model ID Hugging Face WavLM teacher (default: ${REPA_TEACHER_MODEL})
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --batch-size-per-gpu)
      BATCH_SIZE_PER_GPU="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --campplus-onnx)
      CAMPPLUS_ONNX="$2"
      shift 2
      ;;
    --repa-teacher-model)
      REPA_TEACHER_MODEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${GPUS}" ]]; then
  echo "Please pass --gpus or set CUDA_VISIBLE_DEVICES to unused GPU ids so this run does not disturb an existing training job." >&2
  usage
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ -z "${NUM_PROCESSES}" ]]; then
  NUM_PROCESSES="${#GPU_ARR[@]}"
fi

if [[ "${NUM_PROCESSES}" -le 0 ]]; then
  echo "NUM_PROCESSES must be > 0" >&2
  exit 1
fi


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
export WANDB_ENTITY="${WANDB_ENTITY:-xiamingren91-nwpu}"
export WANDB_MODE="online"
# wandb.init 连接超时默认 90s, 网络慢时加大 (单位: 秒)
export WANDB_INIT_TIMEOUT=180
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

EXP_NAME="${CONFIG_NAME}-${DATASET_NAME}-tokentowav-${FLOW_VARIANT}-${NUM_PROCESSES}gpus-${BATCH_SIZE_PER_GPU}frames_per_gpu-${MIXED_PRECISION}"
OUTDIR="${OUTPUT_ROOT}/${DATASET_NAME}/${EXP_NAME}"
LOG_DIR="${OUTDIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"
echo "Config: ${CONFIG_NAME}"
echo "Flow variant: ${FLOW_VARIANT} (model.cfm.prediction=x_pred, model.cfm.loss_space=v, model.cfm.cond_mode=prefix, model.cfm.repa_layer=19 / DiT block 20)"
echo "Dataset: ${DATASET_NAME} (token-to-wave full ~40M)"
echo "Dataset type: ${DATASET_TYPE}"
echo "Condition type: speech_token"
echo "GPUs (CUDA_VISIBLE_DEVICES): ${CUDA_VISIBLE_DEVICES}"
echo "Num processes: ${NUM_PROCESSES}"
echo "Lance path: ${LANCE_PATH}"
echo "Text index: ${TEXT_INDEX_PATH}"
echo "Text vocab (kept for compatibility): ${VOCAB_PATH}"
echo "CosyVoice2 speech token embeds: ${SPEECH_TOKEN_NUM_EMBEDS}"
echo "CAM++ speaker embedding model: ${CAMPPLUS_ONNX}"
echo "REPA WavLM teacher model: ${REPA_TEACHER_MODEL}"
echo "REPA WavLM teacher feature: final last_hidden_state"
echo "Exp Name: ${EXP_NAME}"
echo "Output dir: ${OUTDIR}"
echo "Master port: ${MASTER_PORT}"
echo "Project root: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${CAMPPLUS_ONNX}" ]]; then
  echo "CAM++ ONNX not found: ${CAMPPLUS_ONNX}" >&2
  exit 1
fi

python - <<'PY'
import importlib.util
if importlib.util.find_spec("onnxruntime") is None:
    raise SystemExit(
        "onnxruntime is required for online speaker embedding extraction. "
        "Install onnxruntime or onnxruntime-gpu in the wavtts environment."
    )
PY

accelerate launch \
  --main_process_port "${MASTER_PORT}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  src/wavtts/train/train.py \
  --config-name "${CONFIG_NAME}.yaml" \
  ++hydra.run.dir="${OUTDIR}" \
  ++ckpts.save_dir="${OUTDIR}/ckpts" \
  ++ckpts.exp_name="${EXP_NAME}" \
  ++model.tokenizer="custom" \
  ++model.tokenizer_path="${VOCAB_PATH}" \
  ++model.cfm.prediction="x_pred" \
  ++model.cfm.loss_space="v" \
  ++model.cfm.cond_mode="prefix" \
  ++model.cfm.prefix_drop_prob=0.5 \
  ++model.cfm.prefix_frac=0.3 \
  ++model.cfm.training_cfg_rate=0.2 \
  ++model.cfm.joint_cond_drop_prob=0.0 \
  ++model.cfm.repa_layer=19 \
  ++model.cfm.use_repa_loss=true \
  ++model.cfm.repa_teacher_model="${REPA_TEACHER_MODEL}" \
  ++model.cfm.repa_loss_weight=0.005 \
  ++model.cfm.use_stft_loss=true \
  ++model.cfm.stft_loss_weight=0.0008 \
  ++model.cfm.stft_gamma=1.0 \
  ++model.cfm.stft_eps=0.01 \
  ++model.cfm.stft_start_update=60000 \
  ++model.arch.condition_type="speech_token" \
  ++model.arch.speech_token_num_embeds="${SPEECH_TOKEN_NUM_EMBEDS}" \
  ++model.arch.use_spk_embedding=true \
  ++model.arch.spk_embed_dim=192 \
  ++model.speaker_embedding.online_extract=true \
  ++model.speaker_embedding.model_path="${CAMPPLUS_ONNX}" \
  ++datasets.load_speech_tokens=true \
  ++datasets.batch_size_per_gpu="${BATCH_SIZE_PER_GPU}" \
  ++datasets.num_workers="${NUM_WORKERS}" \
  ++datasets.dataset_type="${DATASET_TYPE}" \
  ++datasets.lance_path="${LANCE_PATH}" \
  ++datasets.text_index_path="${TEXT_INDEX_PATH}" \
  ++datasets.text_jsonl_path="${TEXT_JSONL_PATH}"
