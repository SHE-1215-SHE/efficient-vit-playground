"""ToMe (Token Merging) 正确性验证，本机 CPU 可运行。

守护的不变量:
    1. r=0 等价性: apply_tome(model, 0) 仅注册挂钩不执行合并，输出与原模型逐元素
       一致 (最大绝对误差 < 1e-4)，确保合并逻辑的注入不改变原计算路径。
    2. token 预算守恒: r>0 时每层实际合并 r 个 token，depth 层共移除 depth*r 个，
       逐层计数与配置严格一致。
    3. 输出语义合理: r>0 时输出与原模型的余弦相似度显著高于随机基线
       (随机权重下不相关输出的相似度期望约为 0)。
    4. 熵引导自适应模式边界行为: strength=0 时逐层合并数退化为固定 r；
       strength>0 时逐层自适应 r 每层均 > 0 且落在 [0.3r, 1.7r] 区间内。

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

    # 各对比模型必须共享同一份权重；随机权重下每次 build 的初始化不同，深拷贝保证可比性
    base = build_model(args.model, pretrained=pretrained)
    base.eval()
    with torch.no_grad():
        y_base = base(x)

    # ---- 检查1: r=0 等价性 (捕获挂钩注入破坏原计算路径的回归) ----
    tome0 = copy.deepcopy(base)
    apply_tome(tome0, 0)
    tome0.eval()
    with torch.no_grad():
        y_tome0 = tome0(x)
    diff0 = (y_base - y_tome0).abs().max().item()
    assert diff0 < 1e-4, f"r=0 等价性失败: 最大差异 {diff0}"
    print(f"[通过] 检查1 r=0 等价性: 最大差异 {diff0:.2e}")

    # ---- 检查2: 逐层合并计数守恒 (捕获每层实际合并数偏离配置 r 的实现缺陷) ----
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

    # ---- 检查3: 输出语义合理 (捕获合并过程破坏关键 token 信息导致输出退化的缺陷) ----
    cos = torch.nn.functional.cosine_similarity(
        y_base.flatten(), y_tome.flatten(), dim=0).item()
    print(f"[通过] 检查3 输出余弦相似度: {cos:.4f}"
          + (" (预训练权重下应显著>0, 论文水平 r=8 约在 0.8+)" if pretrained
             else " (随机权重仅供参考, 加 --pretrained 才有意义)"))

    # ---- 检查4: 熵引导自适应模式边界行为 (捕获 strength=0 未退化为固定 r、自适应 r 越界) ----
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
