"""TokenLearner PoC 训练脚本：在 ImageNet-V2 上微调软路由。

定位（诚实声明）: ImageNet-V2 本是验证集，这里切成 8k训练/2k验证 只是
为了"零额外数据"验证 token 浓缩思想可行，属于概念验证（PoC），
不代表该方法在标准协议下的上限。简历表述请用"PoC验证"字样。

训练策略:
    - 冻结全部 ViT 骨干权重（省显存，24GB共享卡友好）
    - 只训练: TokenLearner 路由 + 第6层后的norm + 分类头
    - deit3_small 预训练权重做初始化

用法（服务器）:
    python scripts/train_tokenlearner.py --data data/imagenetv2-matched-frequency-format-val --epochs 10
"""

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from timm.data import create_transform, resolve_model_data_config
from torchvision.datasets import ImageFolder

from evit_lab.models import build_model


def split_dataset(dataset, train_ratio=0.8, seed=0):
    """随机划分训练/验证子集（固定种子保证可复现）。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    n_train = int(len(dataset) * train_ratio)
    return Subset(dataset, perm[:n_train]), Subset(dataset, perm[n_train:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit3_small")
    parser.add_argument("--data", required=True)
    parser.add_argument("--num-tokens", type=int, default=8, help="浓缩后token数L")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="只训小模块, 可用较大lr")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 数据: ImageNet-V2 按 8:2 划分 (PoC)
    model0 = build_model(args.model, pretrained=True, tokenlearner=args.num_tokens)
    config = resolve_model_data_config(model0)
    tf_train = create_transform(**config, is_training=True)
    tf_eval = create_transform(**config, is_training=False)

    ds = ImageFolder(args.data)  # 先不挂transform, 子集分别用不同tf
    ds.samples = [(p, int(c)) for p, c in ds.samples]  # 数字类目直接映射(见eval_accuracy)
    ds.targets = [s[1] for s in ds.samples]
    train_set, val_set = split_dataset(ds)
    train_set.dataset = copy.copy(ds); train_set.dataset.transform = tf_train
    val_set.dataset = copy.copy(ds); val_set.dataset.transform = tf_eval
    print(f"训练/验证: {len(train_set)}/{len(val_set)}")

    model = build_model(args.model, pretrained=True, tokenlearner=args.num_tokens).to(device)
    # 冻结骨干, 只训路由 + 分类头 (+ 浓缩层之后的norm已被冻结也无妨, PoC够用)
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model._tokenlearner.parameters()) + list(model.head.parameters())
    for p in trainable:
        p.requires_grad = True
    print(f"可训练参数: {sum(p.numel() for p in trainable)} (骨干已冻结)")

    loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for images, targets in loader_train:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * targets.size(0)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in loader_val:
                logits = model(images.to(device))
                correct += (logits.argmax(1).cpu() == targets).sum().item()
                total += targets.size(0)
        acc = 100.0 * correct / total
        mark = ""
        if acc > best:
            best, mark = acc, "  <- best"
            torch.save(model.state_dict(), "results/tokenlearner_best.pth")
        print(f"epoch {epoch+1}: loss {run_loss/len(train_set):.4f} | "
              f"val acc {acc:.2f}%{mark} | {time.time()-t0:.0f}s")

    print(f"\n完成, 最佳 val acc {best:.2f}%, 权重在 results/tokenlearner_best.pth")


if __name__ == "__main__":
    main()
