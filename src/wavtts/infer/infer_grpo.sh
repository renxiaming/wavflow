#!/usr/bin/env bash
set -euo pipefail

source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_CFG="${MODEL_CFG:-${PROJECT_ROOT}/src/wavtts/configs/WavTTS_xpred_spkconcat_flow_grpo.yaml}"
JSONL="${JSONL:-${PROJECT_ROOT}/data/seed_infer/zh/seedtest_zh_cosyvoice2_tokens_infer.jsonl}"
CKPT="${GRPO_CKPT:-}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/result/grpo_flowtts_v2}"
CKPT_STEPS="${CKPT_STEPS:-800,1200,1600,2000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/grpo_flowtts_v2}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
PROCS_PER_GPU="${PROCS_PER_GPU:-4}"
SAMPLER="${SAMPLER:-ode}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bf16}"
STEPS="${STEPS:-${NFE_STEP:-10,20,30}}"


NOISE_SCALE="${NOISE_SCALE:-0.05}"
CFG_STRENGTH="${CFG_STRENGTH:-1,1.5,2.0}"
TIMESTEP_MAPPING="${TIMESTEP_MAPPING:-power}"
TIMESTEP_POWER="${TIMESTEP_POWER:-2.0}"
SHIFT="${SHIFT:-3.0}"
TARGET_RMS="${TARGET_RMS:-0.1}"
TOKEN_RATE="${TOKEN_RATE:-25.0}"
MAX_ROWS="${MAX_ROWS:-}"
DATA_MODE="${DATA_MODE:-paired}"
PROMPT_FRACTION_MIN="${PROMPT_FRACTION_MIN:-0.0}"
PROMPT_FRACTION_MAX="${PROMPT_FRACTION_MAX:-0.3}"
PREFIX_SEED="${PREFIX_SEED:-666}"
LANGUAGE="${LANGUAGE:-zh}"
RUN_SIM="${RUN_SIM:-0}"
SIM_SCRIPT="${SIM_SCRIPT:-/home/A02_tmpdata0/wenhaoli/code/seed_eval/run_sim.sh}"
SIM_OUTPUT_DIR="${SIM_OUTPUT_DIR:-${PROJECT_ROOT}/results/sim/flowtts_v2}"
SIM_TARGET_SPK="${SIM_TARGET_SPK:-/home/A02_tmpdata0/wenhaoli/data/test_seed_dir/zh}"
SIM_DEVICE="${SIM_DEVICE:-7}"
DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash src/wavtts/infer/infer_grpo.sh --ckpt-dir DIR --ckpt-steps VALUES [options]

This launches one worker per process, shards JSONL rows, and writes every
checkpoint/steps/cfg combination into its own output directory. Speaker
similarity evaluation is optional and disabled by default.

Options:
  --ckpt PATH              One GRPO checkpoint (compatibility mode).
  --ckpt-dir DIR           Directory containing model_<update>.pt checkpoints.
  --ckpt-steps VALUES      Checkpoint updates, e.g. 3200,4200,last or all.
  --model-cfg PATH         GRPO model config.
  --jsonl PATH             Prompt/target JSONL.
  --output-dir DIR         Experiment output directory.
  --gpus IDS               Physical GPU ids, e.g. 0,1,2,3.
  --procs-per-gpu N        Processes per GPU, default: ${PROCS_PER_GPU}
  --sampler NAME           ode or sde, default: ${SAMPLER}
  --model-dtype TYPE       Persistent model dtype, default: ${MODEL_DTYPE}
  --compute-dtype TYPE     Forward autocast dtype, default: ${COMPUTE_DTYPE}
  --steps VALUES           Denoising/NFE values, e.g. 10,20,40,50.
  --nfe-step VALUES        Alias for --steps.
  --noise-scale X          SDE noise scale, default: ${NOISE_SCALE}
  --cfg-strength VALUES    CFG values, e.g. 1.0,1.5,2.0.
  --timestep-mapping NAME  uniform, power, or sway_sampling.
  --timestep-power X       Power exponent, default: ${TIMESTEP_POWER}
  --shift X                Timestep shift, default: ${SHIFT}
  --target-rms X           Prompt RMS normalization, default: ${TARGET_RMS}
  --token-rate X           Speech token rate, default: ${TOKEN_RATE}
  --max-rows N             Limit rows for a smoke test.
  --data-mode MODE         paired or self_prefix, default: ${DATA_MODE}
  --prompt-fraction-min X  Minimum self-prefix fraction.
  --prompt-fraction-max X  Maximum self-prefix fraction.
  --prefix-seed N          Stable self-prefix sampling seed.
  --language LANG          zh or en, used for self-prefix SIM metadata.
  --sim-output-dir DIR     SIM result root, default: ${SIM_OUTPUT_DIR}
  --sim-script PATH        SIM entry script, default: ${SIM_SCRIPT}
  --sim-target-spk DIR     SeedTTS reference root, default: ${SIM_TARGET_SPK}
  --sim-device DEV         SIM device, default: first inference GPU.
  --skip-sim               Do not run speaker-similarity evaluation.
  --dry-run                Print the worker grid without launching inference.
  -h, --help               Show this help.
EOF
}

split_values() {
  local raw="$1"
  local -n output="$2"
  raw="${raw//，/,}"
  raw="${raw//,/ }"
  output=()
  for value in ${raw}; do
    [[ -n "${value}" ]] && output+=("${value}")
  done
  if [[ "${#output[@]}" -eq 0 ]]; then
    echo "No values parsed from: ${raw}" >&2
    exit 1
  fi
}

tag_num() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  value="${value%p0}"
  echo "${value}"
}

merge_manifests() {
  local case_dir="$1"
  local num_shards="$2"
  python - "${case_dir}" "${num_shards}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
num_shards = int(sys.argv[2])
for stem, suffix in (("manifest", ".jsonl"), ("meta_zh", ".lst"), ("meta_en", ".lst")):
    parts = sorted(root.glob(f"{stem}.shard*-of-{num_shards:05d}{suffix}"))
    merged = root / f"{stem}{suffix}"
    if not parts:
        if num_shards > 1 and stem != "manifest":
            merged.unlink(missing_ok=True)
        continue
    with merged.open("w", encoding="utf-8") as output:
        for part in parts:
            output.write(part.read_text(encoding="utf-8"))
    print(f"Merged {len(parts)} {stem} shards -> {merged}")
PY
}

inference_case_complete() {
  local case_dir="$1"
  python - "${case_dir}" "${JSONL}" "${MAX_ROWS}" <<'PY'
import json
from pathlib import Path
import sys

case_dir = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
max_rows = int(sys.argv[3]) if sys.argv[3] else None
manifest_path = case_dir / "manifest.jsonl"
if not manifest_path.is_file():
    raise SystemExit(1)

expected = sum(1 for line in jsonl_path.open(encoding="utf-8") if line.strip())
if max_rows is not None:
    expected = min(expected, max_rows)

try:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

if len(records) != expected:
    raise SystemExit(1)
for record in records:
    wav_path = Path(record.get("gen_wav", ""))
    if not wav_path.is_file() or wav_path.stat().st_size <= 44:
        raise SystemExit(1)

meta_count = 0
for meta_name in ("meta_zh.lst", "meta_en.lst"):
    meta_path = case_dir / meta_name
    if meta_path.is_file():
        meta_count += sum(1 for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip())
if meta_count != expected:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

sim_case_complete() {
  local sim_case_dir="$1"
  local meta_path="$2"
  python - "${sim_case_dir}" "${meta_path}" <<'PY'
from pathlib import Path
import sys

result_path = Path(sys.argv[1]) / "total_similarity.txt"
meta_path = Path(sys.argv[2])
if not result_path.is_file() or not meta_path.is_file():
    raise SystemExit(1)

result_lines = [
    line.strip()
    for line in result_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
expected = sum(
    1
    for line in meta_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)
if not result_lines or not result_lines[0].startswith("Average similarity:"):
    raise SystemExit(1)
if len(result_lines) != expected + 1:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --ckpt-dir) CKPT_DIR="$2"; CKPT=""; shift 2 ;;
    --ckpt-steps) CKPT_STEPS="$2"; CKPT=""; shift 2 ;;
    --model-cfg) MODEL_CFG="$2"; shift 2 ;;
    --jsonl) JSONL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --procs-per-gpu) PROCS_PER_GPU="$2"; shift 2 ;;
    --sampler) SAMPLER="$2"; shift 2 ;;
    --model-dtype) MODEL_DTYPE="$2"; shift 2 ;;
    --compute-dtype) COMPUTE_DTYPE="$2"; shift 2 ;;
    --steps|--nfe-step) STEPS="$2"; shift 2 ;;
    --noise-scale) NOISE_SCALE="$2"; shift 2 ;;
    --cfg-strength) CFG_STRENGTH="$2"; shift 2 ;;
    --timestep-mapping) TIMESTEP_MAPPING="$2"; shift 2 ;;
    --timestep-power) TIMESTEP_POWER="$2"; shift 2 ;;
    --shift) SHIFT="$2"; shift 2 ;;
    --target-rms) TARGET_RMS="$2"; shift 2 ;;
    --token-rate) TOKEN_RATE="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --data-mode) DATA_MODE="$2"; shift 2 ;;
    --prompt-fraction-min) PROMPT_FRACTION_MIN="$2"; shift 2 ;;
    --prompt-fraction-max) PROMPT_FRACTION_MAX="$2"; shift 2 ;;
    --prefix-seed) PREFIX_SEED="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --sim-output-dir) SIM_OUTPUT_DIR="$2"; shift 2 ;;
    --sim-script) SIM_SCRIPT="$2"; shift 2 ;;
    --sim-target-spk) SIM_TARGET_SPK="$2"; shift 2 ;;
    --sim-device) SIM_DEVICE="$2"; shift 2 ;;
    --skip-sim) RUN_SIM=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# run_sim.sh changes its working directory before launching verification2.py,
# so every filesystem argument passed across that boundary must be absolute.
MODEL_CFG="$(realpath -m "${MODEL_CFG}")"
JSONL="$(realpath -m "${JSONL}")"
CKPT_DIR="$(realpath -m "${CKPT_DIR}")"
if [[ -n "${CKPT}" ]]; then
  CKPT="$(realpath -m "${CKPT}")"
fi
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
SIM_OUTPUT_DIR="$(realpath -m "${SIM_OUTPUT_DIR}")"
SIM_SCRIPT="$(realpath -m "${SIM_SCRIPT}")"
SIM_TARGET_SPK="$(realpath -m "${SIM_TARGET_SPK}")"

if [[ ! -f "${MODEL_CFG}" ]]; then
  echo "Model config not found: ${MODEL_CFG}" >&2
  exit 1
fi
if [[ ! -f "${JSONL}" ]]; then
  echo "JSONL not found: ${JSONL}" >&2
  exit 1
fi
if ! [[ "${PROCS_PER_GPU}" =~ ^[0-9]+$ ]] || [[ "${PROCS_PER_GPU}" -lt 1 ]]; then
  echo "--procs-per-gpu must be a positive integer, got: ${PROCS_PER_GPU}" >&2
  exit 1
fi
if [[ "${SAMPLER}" != "ode" && "${SAMPLER}" != "sde" ]]; then
  echo "--sampler must be ode or sde, got: ${SAMPLER}" >&2
  exit 1
fi
if [[ "${DATA_MODE}" != "paired" && "${DATA_MODE}" != "self_prefix" ]]; then
  echo "--data-mode must be paired or self_prefix, got: ${DATA_MODE}" >&2
  exit 1
fi
if [[ "${LANGUAGE}" != "zh" && "${LANGUAGE}" != "en" ]]; then
  echo "--language must be zh or en, got: ${LANGUAGE}" >&2
  exit 1
fi
if [[ "${RUN_SIM}" -eq 1 ]]; then
  if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "SIM script not found: ${SIM_SCRIPT}" >&2
    exit 1
  fi
  if [[ ! -d "${SIM_TARGET_SPK}" ]]; then
    echo "SIM reference directory not found: ${SIM_TARGET_SPK}" >&2
    exit 1
  fi
fi

split_values "${GPUS}" GPU_VALUES
split_values "${STEPS}" STEP_VALUES
split_values "${CFG_STRENGTH}" CFG_VALUES
if [[ -z "${SIM_DEVICE}" ]]; then
  SIM_DEVICE="cuda:${GPU_VALUES[0]#cuda:}"
elif [[ "${SIM_DEVICE}" =~ ^[0-9]+$ ]]; then
  SIM_DEVICE="cuda:${SIM_DEVICE}"
fi

CKPT_PATHS=()
if [[ -n "${CKPT}" ]]; then
  if [[ ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
  fi
  CKPT_PATHS+=("${CKPT}")
else
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
    exit 1
  fi
  if [[ "${CKPT_STEPS}" == "all" ]]; then
    while IFS= read -r path; do
      CKPT_PATHS+=("${path}")
    done < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'model_[0-9]*.pt' -print | sort -V)
  else
    split_values "${CKPT_STEPS}" CKPT_STEP_VALUES
    for ckpt_step in "${CKPT_STEP_VALUES[@]}"; do
      if [[ "${ckpt_step}" == "last" ]]; then
        ckpt_path="${CKPT_DIR}/model_last.pt"
      elif [[ "${ckpt_step}" == model_*.pt ]]; then
        ckpt_path="${CKPT_DIR}/${ckpt_step}"
      elif [[ "${ckpt_step}" =~ ^[0-9]+$ ]]; then
        ckpt_path="${CKPT_DIR}/model_${ckpt_step}.pt"
      else
        echo "Invalid checkpoint step: ${ckpt_step}" >&2
        exit 1
      fi
      if [[ ! -f "${ckpt_path}" ]]; then
        echo "Checkpoint not found: ${ckpt_path}" >&2
        exit 1
      fi
      CKPT_PATHS+=("${ckpt_path}")
    done
  fi
fi
if [[ "${#CKPT_PATHS[@]}" -eq 0 ]]; then
  echo "No checkpoints selected." >&2
  exit 1
fi

TOTAL_WORKERS=$(( ${#GPU_VALUES[@]} * PROCS_PER_GPU ))

echo "Checkpoints (${#CKPT_PATHS[@]}): ${CKPT_PATHS[*]}"
echo "JSONL: ${JSONL}"
echo "Output root: ${OUTPUT_DIR}"
echo "Sampler: ${SAMPLER}"
echo "GPUs: ${GPU_VALUES[*]}"
echo "Processes per GPU: ${PROCS_PER_GPU}"
echo "Workers per case: ${TOTAL_WORKERS}"
echo "Steps/NFE: ${STEP_VALUES[*]}"
echo "CFG: ${CFG_VALUES[*]}"
echo "Data mode: ${DATA_MODE}"
if [[ "${DATA_MODE}" == "self_prefix" ]]; then
  echo "Prompt fraction: [${PROMPT_FRACTION_MIN}, ${PROMPT_FRACTION_MAX}], seed=${PREFIX_SEED}"
fi
echo "Run SIM: ${RUN_SIM}"
if [[ "${RUN_SIM}" -eq 1 ]]; then
  echo "SIM output root: ${SIM_OUTPUT_DIR}"
  echo "SIM device: ${SIM_DEVICE}"
fi
echo "Dry run: ${DRY_RUN}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_DIR}"
  if [[ "${RUN_SIM}" -eq 1 ]]; then
    mkdir -p "${SIM_OUTPUT_DIR}"
  fi
fi

for ckpt_path in "${CKPT_PATHS[@]}"; do
  ckpt_name="$(basename "${ckpt_path}" .pt)"
  for cfg in "${CFG_VALUES[@]}"; do
    for steps in "${STEP_VALUES[@]}"; do
      case_name="${SAMPLER}_steps$(tag_num "${steps}")_cfg$(tag_num "${cfg}")"
      if [[ "${DATA_MODE}" == "self_prefix" ]]; then
        case_name+="_prefix$(tag_num "${PROMPT_FRACTION_MIN}")to$(tag_num "${PROMPT_FRACTION_MAX}")_seed${PREFIX_SEED}"
      fi
      case_dir="${OUTPUT_DIR}/${ckpt_name}/${case_name}"
      if [[ "${DRY_RUN}" -eq 0 ]]; then
        mkdir -p "${case_dir}/logs"
      fi
      echo "======== ${ckpt_name}/${case_name}"

      case_is_complete=0
      if inference_case_complete "${case_dir}"; then
        case_is_complete=1
        echo "Skipping inference; all expected wav files already exist."
      fi

      if [[ "${case_is_complete}" -eq 0 ]]; then
        pids=()
        worker_index=0
        for gpu in "${GPU_VALUES[@]}"; do
          gpu_id="${gpu#cuda:}"
          for ((proc_idx = 0; proc_idx < PROCS_PER_GPU; proc_idx++)); do
            log_file="${case_dir}/logs/worker$(printf '%03d' "${worker_index}")_gpu${gpu_id}_proc${proc_idx}.log"
            echo "Starting worker ${worker_index}/${TOTAL_WORKERS} on GPU ${gpu_id}; log=${log_file}"
            worker_args=(
              --model-cfg "${MODEL_CFG}"
              --ckpt "${ckpt_path}"
              --jsonl "${JSONL}"
              --output-dir "${case_dir}"
              --device cuda:0
              --model-dtype "${MODEL_DTYPE}"
              --compute-dtype "${COMPUTE_DTYPE}"
              --sampler "${SAMPLER}"
              --steps "${steps}"
              --noise-scale "${NOISE_SCALE}"
              --cfg-strength "${cfg}"
              --timestep-mapping "${TIMESTEP_MAPPING}"
              --timestep-power "${TIMESTEP_POWER}"
              --shift "${SHIFT}"
              --target-rms "${TARGET_RMS}"
              --token-rate "${TOKEN_RATE}"
              --data-mode "${DATA_MODE}"
              --prompt-fraction-min "${PROMPT_FRACTION_MIN}"
              --prompt-fraction-max "${PROMPT_FRACTION_MAX}"
              --prefix-seed "${PREFIX_SEED}"
              --language "${LANGUAGE}"
              --num-shards "${TOTAL_WORKERS}"
              --shard-index "${worker_index}"
            )
            if [[ -n "${MAX_ROWS}" ]]; then
              worker_args+=(--max-rows "${MAX_ROWS}")
            fi
            worker_command=(env "CUDA_VISIBLE_DEVICES=${gpu_id}" python -m wavtts.eval.infer_grpo "${worker_args[@]}")
            if [[ "${DRY_RUN}" -eq 1 ]]; then
              printf 'DRY-RUN:'
              printf ' %q' "${worker_command[@]}"
              printf '\n'
            else
              "${worker_command[@]}" >"${log_file}" 2>&1 &
              pids+=("$!")
            fi
            worker_index=$((worker_index + 1))
          done
        done

        if [[ "${DRY_RUN}" -eq 0 ]]; then
          failed=0
          for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
              failed=1
            fi
          done
          if [[ "${failed}" -ne 0 ]]; then
            echo "One or more workers failed for ${ckpt_name}/${case_name}; check ${case_dir}/logs" >&2
            exit 1
          fi
          merge_manifests "${case_dir}" "${TOTAL_WORKERS}"
          if ! inference_case_complete "${case_dir}"; then
            echo "Inference output is incomplete for ${ckpt_name}/${case_name}." >&2
            exit 1
          fi
        fi
      fi

      if [[ "${RUN_SIM}" -eq 1 ]]; then
        sim_case_dir="${SIM_OUTPUT_DIR}/${ckpt_name}/${case_name}"
        meta_path="${case_dir}/meta_${LANGUAGE}.lst"
        if sim_case_complete "${sim_case_dir}" "${meta_path}"; then
          echo "Skipping SIM; complete result already exists: ${sim_case_dir}/total_similarity.txt"
        elif [[ "${DRY_RUN}" -eq 0 ]]; then
          mkdir -p "${sim_case_dir}"
          (
            flock 9
            if sim_case_complete "${sim_case_dir}" "${meta_path}"; then
              echo "Skipping SIM after lock wait; complete result already exists."
            else
              echo "Running SIM -> ${sim_case_dir}"
              bash "${SIM_SCRIPT}" \
                "${case_dir}" \
                "${SIM_TARGET_SPK}" \
                "${meta_path}" \
                "${sim_case_dir}" \
                "${SIM_DEVICE}" \
                >"${sim_case_dir}/run.log" 2>&1
              if ! sim_case_complete "${sim_case_dir}" "${meta_path}"; then
                echo "SIM output is incomplete for ${ckpt_name}/${case_name}; check ${sim_case_dir}/run.log" >&2
                exit 1
              fi
            fi
          ) 9>"${sim_case_dir}/.sim.lock"
        else
          printf 'DRY-RUN: bash %q %q %q %q %q %q\n' \
            "${SIM_SCRIPT}" "${case_dir}" "${SIM_TARGET_SPK}" \
            "${meta_path}" "${sim_case_dir}" "${SIM_DEVICE}"
        fi
      fi
    done
  done
done

echo "Inference grid completed: ${OUTPUT_DIR}"
