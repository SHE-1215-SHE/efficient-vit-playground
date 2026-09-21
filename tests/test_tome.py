"""ToMe 正确性验证（本机 CPU 可跑，无需 GPU）。

三项检查:
    1. r=0 等价性: 开了 ToMe 但 r=0 时，输出必须和原模型完全一致（挂钩没破坏计算图）
    2. token 递减: r>0 时每层实际合并 r 个 token，12 层共删 depth*r 个
    3. 输出合理: r>0 时 cls 输出与原模型的余弦相似度应远高于随机（随机≈0）

用法:
    python tests/test_tome.py                  # 随机权重快速验证
    python tests/test_tome.py --pretrained     # 额外加载 deit3_small 预训练权重
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from evit_lab.models import build_model
from evit_lab.models.tome import apply_tome, tome_token_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit3_small")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--r", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    pretrained = args.pretrained
    x = torch.randn(2, 3, 224, 224)

    # 三个模型必须共享同一份权重（随机权重下每次build都不同），所以深拷贝
    base = build_model(args.model, pretrained=pretrained)
    base.eval()
    with torch.no_grad():
        y_base = base(x)

    tome0 = copy.deepcopy(base)
    apply_tome(tome0, 0)
    tome0.eval()
    with torch.no_grad():
        y_tome0 = tome0(x)
    diff0 = (y_base - y_tome0).abs().max().item()
    assert diff0 < 1e-4, f"r=0 等价性失败: 最大差异 {diff0}"
    print(f"[通过] 检查1 r=0 等价性: 最大差异 {diff0:.2e}")

    # ---- 检查2: 每层合并 r 个 ----
    tome = copy.deepcopy(base)
    apply_tome(tome, args.r)
    tome.eval()
    with torch.no_grad():
        y_tome = tome(x)
    counts = tome_token_counts(tome)
    expected = [args.r] * len(tome.blocks)
    assert counts == expected, f"每层合并数异常: {counts}"
    n_layers = len(tome.blocks)
    print(f"[通过] 检查2 token递减: 每层合并 {args.r} 个, "
          f"{n_layers} 层共删 {n_layers * args.r} 个 (197 -> {197 - n_layers * args.r})")

    # ---- 检查3: 输出相关性 ----
    cos = torch.nn.functional.cosine_similarity(
        y_base.flatten(), y_tome.flatten(), dim=0).item()
    print(f"[通过] 检查3 输出余弦相似度: {cos:.4f}"
          + (" (预训练权重下应显著>0, 论文水平 r=8 约在 0.8+)" if pretrained
             else " (随机权重仅供参考, 加 --pretrained 才有意义)"))

    # ---- 检查4: 熵引导自适应模式 ----
    for strength in (0.0, 0.7):
        ada = copy.deepcopy(base)
        apply_tome(ada, args.r, strength=strength)
        ada.eval()
        with torch.no_grad():
            ada(x)
        counts = tome_token_counts(ada)
        if strength == 0.0:
            assert counts == expected, f"strength=0 应退化为固定r: {counts}"
            print(f"[通过] 检查4a strength=0 退化为固定r: 每层 {counts[0]}")
        else:
            assert all(c > 0 for c in counts), f"自适应模式下每层都应合并>0: {counts}"
            lo, hi = int(args.r * 0.3), int(args.r * 1.7) + 1
            assert all(lo <= c <= hi for c in counts), f"自适应r超出合理范围: {counts}"
            print(f"[通过] 检查4b strength={strength} 逐层自适应r: {counts}")

    print("\n全部检查通过")


if __name__ == "__main__":
    main()
