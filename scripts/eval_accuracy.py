"""精度评测脚本：在验证集（ImageNet 或 ImageNet-V2）上测 top-1 / top-5。

数据集准备（二选一，都是 ImageFolder 格式）:
    1. ImageNet 验证集:   data/imagenet/val/<类别名>/<图片>.JPEG
    2. ImageNet-V2(推荐): data/imagenetv2/matched-frequency/<0..999>/<图片>.jpeg
       下载脚本: python scripts/download_imagenetv2.py --out data/imagenetv2

注意: ImageNet-V2 的类别目录是数字 0-999，与模型输出的类别索引直接对应；
     ImageNet 验证集的类别名需要 wnid -> index 的映射（timm 提供）。

用法:
    python scripts/eval_accuracy.py --model deit_small --data data/imagenetv2/matched-frequency --dataset v2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
from timm.data import create_transform, resolve_model_data_config
from torchvision.datasets import ImageFolder

from evit_lab.models import build_model


@torch.no_grad()
def evaluate(model, loader, device):
    """遍历验证集，返回 top-1 / top-5 正确率（%）。

    映射说明: ImageFolder 的 class_to_idx 按目录名排序生成，ImageNet-V2 的
    目录名本身就是 0-999 的类别索引，所以预测索引可直接对上；普通 ImageNet
    验证集则用 timm 内置的 wnid 映射表做转换。
    """
    correct1 = correct5 = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        # topk(5): 取概率最大的5个类，看正确答案排第几
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    model = build_model(args.model, pretrained=True, tome_r=args.tome_r).to(device)

    # 预处理必须和 timm 模型训练时一致（resize尺寸/归一化均值方差），
    # 用错预处理是新手最常见的「精度莫名其妙低」的原因
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)

    dataset = ImageFolder(args.data, transform=transform)
    # 关键修复: ImageNet-V2 类别目录是数字名 0-999，ImageFolder 按字典序编号会把
    # "10" 排到 "2" 前面导致标签错乱(Top-1只有1%)。这里把标签重映射为目录名本身
    # 的整数值，与模型输出的标准 ImageNet 类别索引一一对应。
    if all(c.isdigit() for c in dataset.classes):
        dataset.samples = [(p, int(dataset.classes[c])) for p, c in dataset.samples]
        dataset.targets = [s[1] for s in dataset.samples]
        dataset.class_to_idx = {c: int(c) for c in dataset.classes}
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    top1, top5 = evaluate(model, loader, device)

    print(f"模型 {args.model} | 样本数 {len(dataset)}")
    print(f"Top-1: {top1:.2f}%   Top-5: {top5:.2f}%")


if __name__ == "__main__":
    main()
