#!/usr/bin/env bash
# 阶段②全部实验一键运行（服务器 4090）
# 用法: bash scripts/run_phase2.sh 2>&1 | tee results/phase2_log.txt
#
# 对比设计: ToMe 与 EViT 用相同 token 预算逐级对齐，控制变量
#   级别1: 149 token (ToMe r=4  / EViT K=148)
#   级别2: 101 token (ToMe r=8  / EViT K=100)
#   级别3:  53 token (ToMe r=12 / EViT K=52)

set -e
export HF_ENDPOINT=https://hf-mirror.com

DEV=cuda
BS=64
DATA=data/imagenet_val   # 官方ImageNet val(prepare_imagenet_val.py整理); 备用: data/imagenetv2-matched-frequency-format-val
MODEL=deit3_small

echo "================ 0. 正确性自检 ================"
python tests/test_tome.py --pretrained
python tests/test_evit.py --pretrained

echo "================ 1. 速度对比 (batch=$BS) ================"
# 固定预算 vs 熵引导自适应（本项目创新点），同总预算公平对比
for setting in "" "--tome-r 4" "--tome-r 8" "--tome-r 12" \
               "--tome-r 4 --tome-strength 0.7" \
               "--tome-r 8 --tome-strength 0.7" \
               "--tome-r 12 --tome-strength 0.7" \
               "--evit-k 148" "--evit-k 100" "--evit-k 52"; do
    echo "--- $MODEL $setting ---"
    python scripts/eval_speed.py --model $MODEL --device $DEV --batch-size $BS --no-pretrained $setting
done

echo "================ 2. 精度对比 (ImageNet-V2) ================"
for setting in "" "--tome-r 4" "--tome-r 8" "--tome-r 12" \
               "--tome-r 4 --tome-strength 0.7" \
               "--tome-r 8 --tome-strength 0.7" \
               "--tome-r 12 --tome-strength 0.7" \
               "--evit-k 148" "--evit-k 100" "--evit-k 52"; do
    echo "--- $MODEL $setting ---"
    python scripts/eval_accuracy.py --model $MODEL --data $DATA --batch-size 256 $setting
done

echo "================ 完成，结果已同步 tee 到 results/phase2_log.txt ================"
