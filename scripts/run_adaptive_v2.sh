#!/bin/bash
# 自适应(调度版v2)重测: 速度3配置 + 精度3配置
# 用法: bash scripts/run_adaptive_v2.sh
set -e
DEV=cuda
BS=64
MODEL=deit3_small
DATA=data/imagenet_val

echo "================ 1. 速度 (batch=$BS) ================"
for R in 4 8 12; do
    python scripts/eval_speed.py --model $MODEL --device $DEV --batch-size $BS \
        --no-pretrained --tome-r $R --tome-strength 0.7
done

echo "================ 2. 精度 (官方val, batch=256) ================"
for R in 4 8 12; do
    python scripts/eval_accuracy.py --model $MODEL --data $DATA --batch-size 256 \
        --tome-r $R --tome-strength 0.7
done

echo "================ 完成，结果已同步 tee ================"
