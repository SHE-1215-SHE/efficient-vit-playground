#!/usr/bin/env bash
# 实验9补跑: 4 条 MPM 速度 (fvcore scatter_ 崩溃修复后重测)
set -e
PY=${PY:-python}  # 可移植: 默认用当前环境python, 可用 PY=/path/to/python bash scripts/xxx.sh 覆盖
cd "$(dirname "$0")/.."
OUT=results_retest_20260924
MODEL=deit3_small
echo "name,throughput_img_s,latency_ms_per_img" > "$OUT/mpm_speed.csv"
run_speed() {
  name=$1; shift
  log="$OUT/logs/speed_${name}.log"
  "$PY" scripts/eval_speed.py --model "$MODEL" --device cuda --batch-size 64 --no-pretrained "$@" 2>&1 | tee "$log"
  thr=$(grep '吞吐量' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  lat=$(grep '延迟' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "$name,$thr,$lat" >> "$OUT/mpm_speed.csv"
}
run_speed mpm_r4   --tome-r 4  --matcher mutual
run_speed mpm_r8   --tome-r 8  --matcher mutual
run_speed mpm_r12  --tome-r 12 --matcher mutual
run_speed adaptive_mpm_r12 --tome-r 12 --matcher mutual --tome-strength 0.7
echo "===== 速度补跑完成 $(date) ====="
cat "$OUT/mpm_speed.csv"
