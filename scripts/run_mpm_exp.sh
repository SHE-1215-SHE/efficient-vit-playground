#!/usr/bin/env bash
# 实验9: MPM 互最近邻匹配 vs ToMe 二分匹配 (arXiv 2604.05718)
# 9a 统计 -> 9c 速度+精度 (budget-MPM 三档 + 自适应MPM r=12 + pure-MPM batch=1)
# 结果写入 results_retest_20260924/mpm_*.csv
# 用法: bash scripts/run_mpm_exp.sh
set -e
PY=${PY:-python}  # 可移植: 默认用当前环境python, 可用 PY=/path/to/python bash scripts/xxx.sh 覆盖
cd "$(dirname "$0")/.."
OUT=results_retest_20260924
mkdir -p "$OUT/logs"
MODEL=deit3_small; DATA=data/imagenet_val; BS=64
export HF_ENDPOINT=https://hf-mirror.com

echo "===== 9.0 正确性自检 $(date) ====="
"$PY" tests/test_mutual_pair.py --pretrained 2>&1 | tee "$OUT/logs/test_mutual_pair.log"

echo "===== 9a 互配对统计 $(date) ====="
"$PY" scripts/mpm_stats.py --data "$DATA" --r 12 --limit 100 2>&1 | tee "$OUT/logs/mpm_stats.log"

echo "===== 9c-1 速度 (batch=$BS) ====="
echo "name,throughput_img_s,latency_ms_per_img" > "$OUT/mpm_speed.csv"
run_speed() {
  name=$1; shift
  log="$OUT/logs/speed_${name}.log"
  echo "--- speed: $name $* ---"
  "$PY" scripts/eval_speed.py --model "$MODEL" --device cuda --batch-size "$BS" --no-pretrained "$@" 2>&1 | tee "$log"
  thr=$(grep '吞吐量' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  lat=$(grep '延迟' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "$name,$thr,$lat" >> "$OUT/mpm_speed.csv"
}
run_speed mpm_r4   --tome-r 4  --matcher mutual
run_speed mpm_r8   --tome-r 8  --matcher mutual
run_speed mpm_r12  --tome-r 12 --matcher mutual
run_speed adaptive_mpm_r12 --tome-r 12 --matcher mutual --tome-strength 0.7

echo "===== 9c-2 精度 (官方val 5万张) ====="
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

echo "===== 9c-3 pure-MPM 论文原味 (batch=1, 浮动token) ====="
log="$OUT/logs/acc_pure_mpm.log"
"$PY" scripts/eval_accuracy.py --model "$MODEL" --data "$DATA" --batch-size 1 --tome-r 12 --matcher mutual --mpm-pure 2>&1 | tee "$log"
t1=$(grep 'Top-1' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
t5=$(grep 'Top-1' "$log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
echo "pure_mpm_batch1,$t1,$t5,variable" >> "$OUT/mpm_accuracy.csv"

echo "===== 实验9完成 $(date) ====="
echo "--- mpm_speed.csv ---"; cat "$OUT/mpm_speed.csv"
echo "--- mpm_accuracy.csv ---"; cat "$OUT/mpm_accuracy.csv"
