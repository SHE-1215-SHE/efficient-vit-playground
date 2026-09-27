"""统计真实数据下每层"互最近邻对"数量与预算 r 的关系（budget-MPM 可行性依据）。

输入: ImageFolder 格式验证集（数字类目目录）；模型默认 deit3_small，matcher 固定 mutual。
输出: 终端打印逐层统计表 layer, mean_pairs, min_pairs, budget_r, shortfall?，
      回答两个问题:
      1. 互配对数量是否普遍 >= r（决定 budget-MPM 是否可行、回退频率多高）
      2. 互配对数量随深度如何变化（层间冗余结构）
典型用法:
    python scripts/mpm_stats.py --data data/imagenet_val --r 12 --limit 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
from timm.data import create_transform, resolve_model_data_config
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from evit_lab.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit3_small")
    parser.add_argument("--data", default="data/imagenet_val")
    parser.add_argument("--r", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=100, help="只统计前 N 个 batch")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, pretrained=True, tome_r=args.r,
                        tome_matcher="mutual").to(device).eval()
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    dataset = ImageFolder(args.data, transform=transform)
    # NOTE: 与 eval_accuracy.py 相同，数字类目目录需重映射为整型标签，
    # 避免 ImageFolder 字典序编号与标准 ImageNet 类别索引错位
    if all(c.isdigit() for c in dataset.classes):
        dataset.samples = [(p, int(dataset.classes[c])) for p, c in dataset.samples]
        dataset.targets = [s[1] for s in dataset.samples]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    n_layers = len(model.blocks)
    sums = [0] * n_layers
    mins = [float("inf")] * n_layers
    n_batches = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, total=args.limit or None,
                                             desc="统计互配对", ncols=80)):
            if args.limit and i >= args.limit:
                break
            model(images.to(device, non_blocking=True))
            # NOTE: info["pairs"] 按前向顺序记录各次前向的互配对数量，取最近一次
            # 前向值；min_pairs 低于预算 r 的层即存在预算缺口，是统计回退频率的依据
            for l, blk in enumerate(model.blocks):
                pairs = blk._tome_cfg.info.get("pairs")
                if pairs:
                    v = pairs[-1]
                    sums[l] += v
                    mins[l] = min(mins[l], v)
            n_batches += 1

    print(f"\n模型 {args.model} | r={args.r} | {n_batches} batches x {args.batch_size} 张")
    print("layer, mean_pairs, min_pairs, budget_r, shortfall?")
    for l in range(n_layers):
        mean = sums[l] / max(n_batches, 1)
        short = "YES" if mins[l] < args.r else "no"
        print(f"{l:5d}, {mean:10.1f}, {mins[l]:9d}, {args.r:8d}, {short}")


if __name__ == "__main__":
    main()
