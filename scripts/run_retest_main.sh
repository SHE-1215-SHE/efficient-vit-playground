#!/usr/bin/env bash
# 全量重测 part1: 正确性自检 + 速度10组 + 精度11组（含 force-front 消融）
# 协议与 run_phase2.sh / run_adaptive_v2.sh 完全一致, 结果写入 results_retest_20260924/
# 用法: bash scripts/run_retest_main.sh
set -e
PY=${PY:-python}  # 可移植: 默认用当前环境python, 可用 PY=/path/to/python bash scripts/xxx.sh 覆盖
export HF_ENDPOINT=https://hf-mirror.com
cd "$(dirname "$0")/.."
OUT=results_retest_20260924
mkdir -p "$OUT/logs"
DEV=cuda; BS=64; MODEL=deit3_small; DATA=data/imagenet_val

echo "===== env snapshot $(date) ====="
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/env_snapshot.txt"
"$PY" -c "import torch,timm;print('torch',torch.__version__,'timm',timm.__version__)" >> "$OUT/env_snapshot.txt"
cat "$OUT/env_snapshot.txt"

echo "===== 0. 正确性自检 ====="
"$PY" tests/test_tome.py --pretrained 2>&1 | tee "$OUT/logs/test_tome.log"
"$PY" tests/test_evit.py --pretrained 2>&1 | tee "$OUT/logs/test_evit.log"

echo "===== 1. 速度 (batch=$BS, 同一天背靠背) ====="
echo "name,throughput_img_s,latency_ms_per_img" > "$OUT/speed.csv"
run_speed() {
  name=$1; shift
  log="$OUT/logs/speed_${name}.log"
  echo "--- speed: $name $* ---"
  "$PY" scripts/eval_speed.py --model "$MODEL" --device "$DEV" --batch-size "$BS" --no-pretrained "$@" 2>&1 | tee "$log"
  thr=$(grep '吞吐量' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  lat=$(grep '延迟' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "$name,$thr,$lat" >> "$OUT/speed.csv"
}
run_speed baseline
run_speed tome_r4   --tome-r 4
run_speed tome_r8   --tome-r 8
run_speed tome_r12  --tome-r 12
run_speed adaptive_r4  --tome-r 4  --tome-strength 0.7
run_speed adaptive_r8  --tome-r 8  --tome-strength 0.7
run_speed adaptive_r12 --tome-r 12 --tome-strength 0.7
run_speed evit_k148 --evit-k 148
run_speed evit_k100 --evit-k 100
run_speed evit_k52  --evit-k 52

echo "===== 2. 精度 (官方val 5万张, batch=256) ====="
echo "name,top1,top5,final_tokens_measured" > "$OUT/accuracy.csv"
run_acc() {
  name=$1; shift
  log="$OUT/logs/acc_${name}.log"
  echo "--- acc: $name $* ---"
  "$PY" scripts/eval_accuracy.py --model "$MODEL" --data "$DATA" --batch-size 256 "$@" 2>&1 | tee "$log"
  t1=$(grep 'Top-1' "$log" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  t5=$(grep 'Top-1' "$log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  tok=$(grep '实际最终token数' "$log" | grep -oE '[0-9]+' | tail -1 || true)
  echo "$name,$t1,$t5,${tok:-197}" >> "$OUT/accuracy.csv"
}
run_acc baseline
run_acc tome_r4   --tome-r 4
run_acc tome_r8   --tome-r 8
run_acc tome_r12  --tome-r 12
run_acc adaptive_r4  --tome-r 4  --tome-strength 0.7
run_acc adaptive_r8  --tome-r 8  --tome-strength 0.7
run_acc adaptive_r12 --tome-r 12 --tome-strength 0.7
run_acc ablation_front_r12 --tome-r 12 --tome-strength 0.7 --tome-force-schedule front
run_acc evit_k148 --evit-k 148
run_acc evit_k100 --evit-k 100
run_acc evit_k52  --evit-k 52

echo "===== part1 完成 $(date) ====="
echo "--- speed.csv ---"; cat "$OUT/speed.csv"
echo "--- accuracy.csv ---"; cat "$OUT/accuracy.csv"
