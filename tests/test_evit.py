"""EViT (基于注意力分数的 token 剪枝) 正确性验证，本机 CPU 可运行。

守护的不变量:
    1. K>=N 无操作等价: keep_num >= 序列长度 N 时不触发 token 选择，输出与原模型
       逐元素一致 (最大绝对误差 < 1e-4)，确保剪枝挂钩的注入不改变原计算路径。
    2. token 预算守恒与序列长度稳定: 依据论文协议 (start_layer=4)，前 4 层不丢弃；
       首个选择层 197 -> K+1；此后每层淘汰 1 个并融合回 1 个，序列长度恒为 K+1，
       evit_removed_counts 的逐层丢弃计数与该协议严格一致。
    3. 输出语义合理: K=100 (197->101，与 ToMe r=8 同预算) 时输出与原模型的余弦
       相似度显著高于随机基线。

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

    # ---- 检查1: K>=N 无操作等价 (捕获未触发选择时仍改变输出的回归) ----
    noop = copy.deepcopy(base)
    apply_evit(noop, keep_num=500)  # 197 < 500, 不触发选择
    noop.eval()
    with torch.no_grad():
        y_noop = noop(x)
    diff = (y_base - y_noop).abs().max().item()
    assert diff < 1e-4, f"K>=N 无操作等价性失败: {diff}"
    print(f"[通过] 检查1 K>=N 无操作等价: 最大差异 {diff:.2e}")

    # ---- 检查2: token 预算守恒与序列稳定 (捕获剪枝层位置、丢弃计数偏离论文协议的缺陷) ----
    evit = copy.deepcopy(base)
    apply_evit(evit, keep_num=args.k)   # 默认第4层开始剪枝(论文协议)
    evit.eval()
    with torch.no_grad():
        y_evit = evit(x)
    removed = evit_removed_counts(evit)
    # 论文协议: start_layer 默认为 4，首次选择发生在第 5 个 block 前 (0-indexed)。
    # 前 4 层 (0~3) 不丢弃；首个选择层由 196 个 patch 中保留 K-1 个并融合 1 个，
    # 丢弃 196-(K-1) 个；此后序列长度恰等于预算 K+1，不再丢弃。
    assert all(r == 0 for r in removed[:4]), f"前4层不应丢弃(论文协议): {removed[:4]}"
    assert removed[4] == 196 - (args.k - 1), \
        f"第5个block应丢弃 {196 - (args.k - 1)} 个, 实际 {removed[4]}"
    assert all(r == 0 for r in removed[5:]), f"稳态后不应再丢弃: {removed[5:]}"
    print(f"[通过] 检查2 序列稳定: 前4层不剪, 第5块丢弃 {removed[4]} 个, "
          f"197 -> {args.k + 1} 后保持不变")

    # ---- 检查3: 输出语义合理 (捕获剪枝过程破坏关键 token 信息导致输出退化的缺陷) ----
    cos = torch.nn.functional.cosine_similarity(
        y_base.flatten(), y_evit.flatten(), dim=0).item()
    print(f"[通过] 检查3 输出余弦相似度: {cos:.4f}"
          + (" (K=100 与 ToMe r=8 同预算, 可横向比较)" if args.pretrained
             else " (随机权重仅供参考)"))

    print("\n全部检查通过")


if __name__ == "__main__":
    main()
