"""精度评测脚本：在 ImageFolder 格式验证集（ImageNet / ImageNet-V2）上评测 Top-1 / Top-5。

输入:
    1. ImageNet 验证集:   data/imagenet/val/<类别名>/<图片>.JPEG
    2. ImageNet-V2(推荐): data/imagenetv2/matched-frequency/<0..999>/<图片>.jpeg
       下载脚本: python scripts/download_imagenetv2.py --out data/imagenetv2
输出: 终端打印 Top-1 / Top-5（%）；启用自适应 ToMe 时额外打印每层实际合并数。
典型用法:
    python scripts/eval_accuracy.py --model deit_small --data data/imagenetv2/matched-frequency --dataset v2

NOTE: 标签映射规则——ImageNet-V2 类别目录为数字 0-999，与模型输出类别索引直接
对应；ImageNet 验证集为 wnid 目录名，需经 timm 内置的 wnid -> index 映射表转换。
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


@torch.no_grad()
def evaluate(model, loader, device):
    """遍历验证集，返回 top-1 / top-5 正确率（%）。

    映射说明: ImageFolder 的 class_to_idx 按目录名排序生成，ImageNet-V2 的
    目录名本身就是 0-999 的类别索引，所以预测索引可直接对上；普通 ImageNet
    验证集则用 timm 内置的 wnid 映射表做转换。
    """
    correct1 = correct5 = total = 0
    for images, targets in tqdm(loader, desc="评测中", ncols=80):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        # topk(5): 单次 topk 同时服务 Top-1 与 Top-5 统计，避免对 logits 排序两次
        _, pred = logits.topk(5, dim=1)
        correct = pred.eq(targets.unsqueeze(1))
        correct1 += correct[:, 0].sum().item()
        correct5 += correct.any(dim=1).sum().item()
        total += targets.numel()
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit_small")
    parser.add_argument("--data", required=True, help="验证集 ImageFolder 根目录")
    parser.add_argument("--dataset", default="v2", choices=["v2"],
                        help="当前仅支持 v2=ImageNet-V2(数字类目直接对应输出索引)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="只测前N个batch（0=全量，调试用）")
    parser.add_argument("--tome-r", type=int, default=0,
                        help=">0 时启用 ToMe（每层合并 r 个 token）；需配合 deit3/vit 权重")
    parser.add_argument("--tome-strength", type=float, default=0.0,
                        help="ToMe 自适应强度 0~1（0=固定预算原版；>0=熵引导预算调度，本项目创新点）")
    parser.add_argument("--tome-force-schedule", default=None, choices=["front", "back"],
                        help="消融用: 强制指定调度模板, 跳过熵决策")
    parser.add_argument("--matcher", default="bipartite", choices=["bipartite", "mutual"],
                        help="ToMe 合并的匹配算法: bipartite=原版二分匹配, mutual=MPM互最近邻")
    parser.add_argument("--mpm-pure", action="store_true",
                        help="MPM 论文原味模式: 合并全部互配对, 输出长度数据相关, 仅支持 batch=1")
    parser.add_argument("--evit-k", type=int, default=0,
                        help=">0 时启用 EViT（保留 k 个普通 token）；需配合 deit3/vit 权重")
    parser.add_argument("--evit-start", type=int, default=4,
                        help="EViT 从第几层开始剪枝（论文默认4, 1=复现过早剪枝消融）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    model = build_model(args.model, pretrained=True, tome_r=args.tome_r,
                        tome_strength=args.tome_strength, evit_k=args.evit_k,
                        evit_start=args.evit_start,
                        tome_force_schedule=args.tome_force_schedule,
                        tome_matcher=args.matcher,
                        tome_budget=not args.mpm_pure).to(device)

    # NOTE: 评测预处理必须与 timm 训练配置一致（resize 尺寸 / 归一化均值方差），
    # 不一致会引入系统性精度损失，是评测结果异常偏低的常见原因
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)

    dataset = ImageFolder(args.data, transform=transform)
    # NOTE: ImageNet-V2 类别目录名为数字 0-999，ImageFolder 按字典序编号会把
    # "10" 排在 "2" 之前，导致标签错乱（Top-1 仅约 1%）。此处将标签重映射为
    # 目录名的整数值，与标准 ImageNet 类别索引一一对应。
    if all(c.isdigit() for c in dataset.classes):
        dataset.samples = [(p, int(dataset.classes[c])) for p, c in dataset.samples]
        dataset.targets = [s[1] for s in dataset.samples]
        dataset.class_to_idx = {c: int(c) for c in dataset.classes}
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"模型 {args.model} | 样本数 {len(dataset)} | 开始评测(带进度条)...")
    top1, top5 = evaluate(model, loader, device)

    print(f"Top-1: {top1:.2f}%   Top-5: {top5:.2f}%")
    # 自适应模式的诊断信息: 每层实际合并数（取最后一个 batch 的采样值），
    # 用于核对熵引导的预算调度是否生效
    if args.tome_r > 0 and args.tome_strength > 0:
        from evit_lab.models.tome import tome_token_counts
        counts = tome_token_counts(model)
        print(f"实际每层合并数(末batch采样): {counts}  合计 {sum(counts)}")
        print(f"实际最终token数: {197 - sum(counts)}")


if __name__ == "__main__":
    main()
