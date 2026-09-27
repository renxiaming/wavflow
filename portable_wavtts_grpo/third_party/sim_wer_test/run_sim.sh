#!/usr/bin/env bash
set -e

# 用法：
#   bash run_sim.sh <生成wav目录> <meta.lst> <输出目录> [GPU/DEVICE]
#
# 例子：
#   bash run_sim.sh /path/to/gen_wavs testsets/seedtest/zh/meta.lst output/sim/test cuda:0
#   bash run_sim.sh /path/to/gen_wavs testsets/seedtest/zh/meta.lst output/sim/test 7
#
# meta 第三列里的 prompt-wavs/xxx.wav 会按 meta.lst 所在目录解析。
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/environment/zhguo/anaconda3}"
CONDA_ENV="${CONDA_ENV:-$CONDA_ROOT/envs/torch112_dualvc}"

if [ $# -lt 3 ]; then
    echo "用法：bash run_sim.sh <生成wav目录> <meta.lst> <输出目录> [GPU]"
    exit 1
fi

syn_wav_dir="$1"
meta_lst="$2"
output_dir="$3"
device="${4:-cuda:0}"
if [[ "$device" =~ ^[0-9]+$ ]]; then
    device="cuda:$device"
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$ROOT"
python3 verification2.py \
    --wavs "$syn_wav_dir" \
    --metalst "$meta_lst" \
    --output "$output_dir" \
    --device "$device"
