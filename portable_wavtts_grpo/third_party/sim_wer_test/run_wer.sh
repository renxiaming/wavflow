#!/usr/bin/env bash
set -e

# 用法：
#   bash run_wer.sh <meta.lst> <zh|en> <实验名> <并行数> <GPU> <生成wav目录> [输出目录]
#
# 例子：
#   bash run_wer.sh testsets/seedtest/zh/meta.lst zh test_exp 3 0 /path/to/gen_wavs output/wer_runs
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/environment/zhguo/anaconda3}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if [ $# -lt 6 ]; then
    echo "用法：bash run_wer.sh <meta.lst> <zh|en> <实验名> <并行数> <GPU> <生成wav目录> [输出目录]"
    exit 1
fi

output_dir="${7:-output/wer_runs}"

bash "$ROOT/cal_wer.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$output_dir"
