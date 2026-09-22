#!/usr/bin/env bash
# 在本机跑完 Extended YaleB 的全部实验。
# 用法：  bash run_yaleb_local.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 数据集在上两层的 data/ 里，直接指过去，不需要复制或软链接
DATA="$(cd "$HERE/../../.." && pwd)/data"

if [[ ! -d "$DATA/CroppedYaleB" ]]; then
  echo "找不到数据集: $DATA/CroppedYaleB" >&2
  echo "请确认 Assignment 1/data/ 下有 ORL/ 和 CroppedYaleB/ 两个文件夹" >&2
  exit 1
fi

echo "数据集路径: $DATA"
echo "输出路径:   $HERE/results"
echo

echo "=== 1/3  YaleB 定性图（噪声示例 / 重建 / 基底）==="
python3 make_figures.py    --dataset yaleb --data-root "$DATA" --output results --max-iter 250

echo "=== 2/3  YaleB 主网格：13 个噪声条件 x 5 个算法 x 5 次重复 ==="
python3 run_experiment.py  --dataset yaleb --data-root "$DATA" --output results --max-iter 250 --runs 5

echo "=== 3/3  YaleB 收敛实验 ==="
python3 run_convergence.py --dataset yaleb --data-root "$DATA" --output results --max-iter 250

echo
echo "YALEB_ALL_DONE  全部完成，结果在 $HERE/results"
