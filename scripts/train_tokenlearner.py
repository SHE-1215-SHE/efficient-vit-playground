"""TokenLearner 训练脚本：冻结 ViT 骨干，在 ImageNet train 子集上训练软路由。

输入: ImageNet train 目录（wnid 类目结构），取子集训练（默认 100 类 x 500 张）。
输出: {out}_best.pth / {out}_last.pth checkpoint 与终端训练日志（loss / val acc）。
定位: 冻结全部骨干参数，只训练 TokenLearner 路由（约 3k 参数）+ 分类头；
      类索引取 wnid 排序序号，与预训练模型的输出类别空间一致。
典型用法（服务器）:
    python scripts/train_tokenlearner.py --data /newdisk/data/ImageNet/train --epochs 5
    # 扩大数据规模: --classes 300 --per-class 1000
"""

import argparse
import copy
import random
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


def build_subset(root, classes=100, per_class=500, val_ratio=0.1, seed=0):
    """从 wnid 目录结构取子集：按排序后类序取前 classes 类、每类 per_class 张，再按 val_ratio 切分训练/验证集。"""
    full = ImageFolder(root)          # ImageFolder 类索引按目录名排序生成，wnid 排序即 torchvision/timm 约定的类索引
    rng = random.Random(seed)
    by_cls = {}
    for i, (_, t) in enumerate(full.samples):
        by_cls.setdefault(t, []).append(i)
    chosen = []
    for c in sorted(by_cls)[:classes]:
        chosen += rng.sample(by_cls[c], min(per_class, len(by_cls[c])))
    rng.shuffle(chosen)
    n_val = int(len(chosen) * val_ratio)
    return (Subset(full, chosen[n_val:]), Subset(full, chosen[:n_val]), full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit3_small")
    parser.add_argument("--data", required=True, help="ImageNet train 目录(wnid结构)")
    parser.add_argument("--classes", type=int, default=100, help="子集类别数")
    parser.add_argument("--per-class", type=int, default=500, help="每类图片数")
    parser.add_argument("--num-tokens", type=int, default=8, help="浓缩后token数L; 0=基线(只训头,不浓缩,197 token)")
    parser.add_argument("--resume", default="", help="从checkpoint续训, 如 --resume results/tokenlearner_last.pth")
    parser.add_argument("--out", default="results/tokenlearner", help="checkpoint前缀, 生成 {前缀}_best.pth / _last.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 数据: ImageNet train 子集，训练/验证按 9:1 切分（build_subset 内 val_ratio=0.1）
    tf_train_model = build_model(args.model, pretrained=True, tokenlearner=args.num_tokens)
    config = resolve_model_data_config(tf_train_model)
    tf_train = create_transform(**config, is_training=True)
    tf_eval = create_transform(**config, is_training=False)

    train_set, val_set, full = build_subset(args.data, args.classes, args.per_class)
    train_set.dataset = copy.copy(full); train_set.dataset.transform = tf_train
    val_set.dataset = copy.copy(full); val_set.dataset.transform = tf_eval
    print(f"训练/验证: {len(train_set)}/{len(val_set)} ({args.classes}类)")

    model = build_model(args.model, pretrained=True, tokenlearner=args.num_tokens).to(device)
    # NOTE: 骨干参数全部冻结，可训练参数仅 TokenLearner 路由与分类头，
    # 优化目标规模极小，无需大 batch 或长训练周期
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model.head.parameters())
    if args.num_tokens > 0:
        trainable = list(model._tokenlearner.parameters()) + trainable
    for p in trainable:
        p.requires_grad = True

    loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best, start_epoch = 0.0, 0

    # NOTE: 断点续训须同时恢复模型权重、优化器动量、调度器进度与已完成 epoch 数，
    # 否则学习率曲线与最佳精度记录会错位
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        if isinstance(ckpt, dict) and "opt" in ckpt:      # 新格式: 含优化器/调度器状态的完整 checkpoint
            opt.load_state_dict(ckpt["opt"])
            ckpt["sched"] and sched.load_state_dict(ckpt["sched"])
            start_epoch, best = ckpt.get("epoch", 0), ckpt.get("best", 0.0)
            print(f"续训: 从epoch {start_epoch} 恢复, 历史最佳 {best:.2f}%")
        else:
            print("续训: 旧格式(仅权重)已加载, 优化器从头开始")
    mode = f"TokenLearner L={args.num_tokens}" if args.num_tokens > 0 else "基线(仅训头, 197 token)"
    print(f"模式: {mode} | 可训练参数: {sum(p.numel() for p in trainable)} (骨干已冻结)")

    for epoch in range(start_epoch, args.epochs):
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
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "epoch": epoch + 1, "best": best,
                "args": vars(args)}
        if acc > best:
            best, mark = acc, "  <- best"
            torch.save(ckpt, f"{args.out}_best.pth")
        torch.save(ckpt, f"{args.out}_last.pth")          # 每轮覆盖保存，供断点续训恢复
        print(f"epoch {epoch+1}: loss {run_loss/len(train_set):.4f} | "
              f"val acc {acc:.2f}%{mark} | {time.time()-t0:.0f}s")

    print(f"\n完成, 最佳 val acc {best:.2f}%, 权重在 {args.out}_best.pth")


if __name__ == "__main__":
    main()
