#!/usr/bin/env bash
# 全量重测 part2: TokenLearner 两条训练重跑 (L=8 浓缩 + 基线仅训头)
# 协议与原实验一致: 100类x500张子集, seed=0 (数据切分可复现), 15 epochs, 冻结骨干
# 用法: bash scripts/run_retest_tokenlearner.sh
set -e
PY=${PY:-python}  # 可移植: 默认用当前环境python, 可用 PY=/path/to/python bash scripts/xxx.sh 覆盖
export HF_ENDPOINT=https://hf-mirror.com
cd "$(dirname "$0")/.."
OUT=results_retest_20260924
DATA=/newdisk/data/ImageNet/train_all   # 原实验协议 (checkpoint args 证实); train/ 有1066条目勿用
mkdir -p "$OUT/logs"

echo "===== TokenLearner L=8 重训 $(date) ====="
"$PY" scripts/train_tokenlearner.py --data "$DATA" --epochs 15 --num-tokens 8 \
    --out "$OUT/tokenlearner" 2>&1 | tee "$OUT/logs/tl_L8_retrain.log"

echo "===== TokenLearner 基线(仅训头,197token) 重训 $(date) ====="
"$PY" scripts/train_tokenlearner.py --data "$DATA" --epochs 15 --num-tokens 0 \
    --out "$OUT/tl_baseline" 2>&1 | tee "$OUT/logs/tl_baseline_retrain.log"

echo "===== part2 完成 $(date) ====="
echo "model,mode,best_val_acc_100cls" > "$OUT/tokenlearner.csv"
a=$(grep '完成' "$OUT/logs/tl_baseline_retrain.log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "deit3_small,frozen backbone + head only (197 tokens),$a" >> "$OUT/tokenlearner.csv"
b=$(grep '完成' "$OUT/logs/tl_L8_retrain.log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "deit3_small,frozen backbone + TokenLearner L=8 + head,$b" >> "$OUT/tokenlearner.csv"
cat "$OUT/tokenlearner.csv"
