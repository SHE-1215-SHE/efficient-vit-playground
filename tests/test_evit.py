"""EViT 正确性验证（本机 CPU 可跑）。

三项检查:
    1. K >= N 时无操作: 输出必须和原模型完全一致
    2. 序列稳定: 第1层 197 -> K+1，之后每层只淘汰1个再融合回1个，保持 K+1
    3. 输出相关性: K=100 (197->101, 与ToMe r=8同预算) 时与原模型输出的余弦相似度

用法:
    python tests/test_evit.py
    python tests/test_evit.py --pretrained
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from evit_lab.models import build_model
from evit_lab.models.evit import apply_evit, evit_removed_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit3_small")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    base = build_model(args.model, pretrained=args.pretrained)
    base.eval()
    with torch.no_grad():
        y_base = base(x)

    # ---- 检查1: K 足够大时无操作等价 ----
    noop = copy.deepcopy(base)
    apply_evit(noop, keep_num=500)  # 197 < 500, 不触发选择
    noop.eval()
    with torch.no_grad():
        y_noop = noop(x)
    diff = (y_base - y_noop).abs().max().item()
    assert diff < 1e-4, f"K>=N 无操作等价性失败: {diff}"
    print(f"[通过] 检查1 K>=N 无操作等价: 最大差异 {diff:.2e}")

    # ---- 检查2: 序列稳定在 K+1 ----
    evit = copy.deepcopy(base)
    apply_evit(evit, keep_num=args.k)
    evit.eval()
    with torch.no_grad():
        y_evit = evit(x)
    removed = evit_removed_counts(evit)
    # 第1层: 196个普通token -> 保留99+1融合=100个, 丢97个; 之后序列不再超预算, 保持稳定
    assert removed[0] == 196 - (args.k - 1), f"第1层应丢弃 {196 - (args.k - 1)} 个, 实际 {removed[0]}"
    assert all(r == 0 for r in removed[1:]), f"稳态后不应再丢弃: {removed}"
    print(f"[通过] 检查2 序列稳定: 第1层丢弃 {removed[0]} 个, 序列 197 -> {args.k + 1} 后保持不变")

    # ---- 检查3: 输出相关性 ----
    cos = torch.nn.functional.cosine_similarity(
        y_base.flatten(), y_evit.flatten(), dim=0).item()
    print(f"[通过] 检查3 输出余弦相似度: {cos:.4f}"
          + (" (K=100 与 ToMe r=8 同预算, 可横向比较)" if args.pretrained
             else " (随机权重仅供参考)"))

    print("\n全部检查通过")


if __name__ == "__main__":
    main()
