#!/usr/bin/env bash
# 实验9补跑: 4 条 budget-MPM 精度 (首次因 run_acc shift bug 未跑成)
# 写入 results_retest_20260924/mpm_accuracy.csv (pure-MPM 行由主脚本稍后追加)
set -e
PY=${PY:-python}  # 可移植: 默认用当前环境python, 可用 PY=/path/to/python bash scripts/xxx.sh 覆盖
cd "$(dirname "$0")/.."
OUT=results_retest_20260924
MODEL=deit3_small; DATA=data/imagenet_val
export HF_ENDPOINT=https://hf-mirror.com

echo "name,top1,top5,final_tokens" > "$OUT/mpm_accuracy.csv"
run_acc() {
  name=$1; tok=$2; shift 2
  log="$OUT/logs/acc_${name}.log"
  echo "--- acc: $name $* ---"
  "$PY" scripts/eval_accuracy.py --model "$MODEL" --data "$DATA" --batch-size 256 "$@" 2>&1 | tee "$log"
  t1=$(grep 'Top-1' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  t5=$(grep 'Top-1' "$log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  echo "$name,$t1,$t5,$tok" >> "$OUT/mpm_accuracy.csv"
}
run_acc mpm_r4  149 --tome-r 4  --matcher mutual
run_acc mpm_r8  101 --tome-r 8  --matcher mutual
run_acc mpm_r12 53  --tome-r 12 --matcher mutual
run_acc adaptive_mpm_r12 53 --tome-r 12 --matcher mutual --tome-strength 0.7
echo "===== 补跑完成 $(date) ====="
cat "$OUT/mpm_accuracy.csv"
