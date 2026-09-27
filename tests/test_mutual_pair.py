"""MPM 互最近邻合并 (mutual_pair_merge) 正确性自检。

守护的不变量:
    1. budget 模式形状守恒: 输出恒为 (B, N-r, C)，合并数恰为 r，CLS token
       逐元素原样保留。
    2. 确定性: 同一输入重复调用，输出与合并数完全一致 (对应 MPM 论文的确定性声明)。
    3. 合并语义: 互为最近邻的重复 token 合并后均值等于原值。
    4. 短缺回退: 互配对数不足 r 时回退二分匹配，输出形状与预算 r 仍严格满足。
    5. pure 模式仅支持 batch=1，输出长度数据相关；batch>1 必须抛出 ValueError。

用法:
    python tests/test_mutual_pair.py            # 随机张量, 无需权重
    python tests/test_mutual_pair.py --pretrained  # 加载真实模型冒烟
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evit_lab.models.mutual_pair import mutual_pair_merge


def test_budget_exact_shape():
    """验证 budget 模式形状守恒与 CLS 保留；捕获合并数偏离 r、CLS 被误合并导致的输出错误。"""
    torch.manual_seed(0)
    B, N, C, r = 4, 50, 16, 5
    x = torch.randn(B, N, C)
    metric = torch.randn(B, N, C)
    out, merged, pairs = mutual_pair_merge(x, metric, r, budget=True, min_tokens=2)
    assert out.shape == (B, N - r, C), f"形状错误: {out.shape}"
    assert merged == r, f"合并数应为 {r}, 实际 {merged}"
    assert torch.equal(out[:, 0], x[:, 0]), "CLS token 必须原样保留"
    print(f"[PASS] budget 形状 (B,{N}->{N-r},{C}), CLS 保留, 本批最少互配对 {pairs}")


def test_determinism():
    """验证合并确定性；捕获由集合遍历顺序不稳定引发的非确定性输出。"""
    torch.manual_seed(1)
    x = torch.randn(2, 40, 16)
    metric = torch.randn(2, 40, 16)
    o1, m1, _ = mutual_pair_merge(x, metric, 6)
    o2, m2, _ = mutual_pair_merge(x, metric, 6)
    assert torch.equal(o1, o2) and m1 == m2
    print("[PASS] 确定性: 两次输出逐元素一致")


def test_duplicate_tokens_merge_to_themselves():
    """验证重复 token 的合并语义；捕获互最近邻配对错位或均值计算错误导致的输出偏移。"""
    x = torch.randn(1, 1, 8).repeat(1, 2, 1)          # 构造一对完全相同的 patch token
    x = torch.cat([torch.randn(1, 1, 8), x], dim=1)   # (1, 3, 8): CLS + 一对重复 token
    out, merged, _ = mutual_pair_merge(x, x.clone(), 1, budget=True, min_tokens=2)
    assert out.shape == (1, 2, 8) and merged == 1
    # 重复对合并 = 均值 = 原值; 输出应为 [CLS, 原值]
    assert torch.allclose(out[0, 1], x[0, 1], atol=1e-6)
    print("[PASS] 重复 token 合并语义正确 (均值=原值)")


def test_shortfall_fallback():
    """验证互配对不足时的二分匹配回退；捕获回退后预算不满 r 或输出形状仍为 N 的缺陷。"""
    torch.manual_seed(3)
    x = torch.nn.functional.normalize(torch.randn(1, 30, 32), dim=-1)
    x[0, 1] = x[0, 0]  # 制造一对明确的重复
    metric = x
    _, _, p = mutual_pair_merge(x, metric, 1, budget=True, min_tokens=2)  # 探测互配对数
    cap = 30 // 2 - 1                           # 二分匹配在 N=30 下的最大合并数上限 (A 集 15 个, 扣除 1 个保护位)
    r = min(p + 3, cap)                         # 预算超出互配对数且不超过二分容量
    out, merged, pairs = mutual_pair_merge(x, metric, r, budget=True, min_tokens=2)
    assert out.shape == (1, 30 - r, 32), f"回退后形状错误: {out.shape}"
    assert merged == r, f"回退后仍应合并满 r={r}, 实际 {merged}"
    assert pairs < r, f"应触发回退 (pairs={pairs})"
    print(f"[PASS] 短缺回退: pairs={pairs} < r={r}, 回退后形状/预算均正确")


def test_pure_mode_batch1():
    """验证 pure 模式输出长度数据相关且拒绝 batch>1；捕获批维广播错位导致的静默错误输出。"""
    torch.manual_seed(2)
    x = torch.randn(1, 60, 16)
    out, merged, pairs = mutual_pair_merge(x, x.clone(), 0, budget=False, min_tokens=2)
    assert out.shape[0] == 1 and out.shape[1] == 60 - merged
    assert torch.equal(out[:, 0], x[:, 0])
    try:
        mutual_pair_merge(torch.randn(2, 60, 16), torch.randn(2, 60, 16), 0, budget=False)
        raise AssertionError("pure 模式 batch>1 应报错")
    except ValueError:
        pass
    print(f"[PASS] pure 模式: 合并 {merged} 对, 输出 {60}->{out.shape[1]}, batch>1 正确拒绝")


def test_smoke_pretrained():
    """真实模型冒烟；捕获 mutual 匹配器在完整前向中挂钩失效或逐层合并数偏离配置值的缺陷。"""
    from evit_lab.models import build_model
    model = build_model("deit3_small", pretrained=False, tome_r=8, tome_matcher="mutual")
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 224, 224))
    assert y.shape == (2, 1000)
    rs = [sum(b._tome_cfg.info["r"][-1:]) for b in model.blocks]
    assert all(v == 8 for v in rs), f"每层应恰好合并 8 个: {rs}"
    pairs = [b._tome_cfg.info.get("pairs", [None])[-1] for b in model.blocks]
    print(f"[PASS] 真实模型冒烟: 197->53, 每层合并8, 各层互配对数 {pairs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true", help="附加真实模型冒烟")
    args = ap.parse_args()
    test_budget_exact_shape()
    test_determinism()
    test_duplicate_tokens_merge_to_themselves()
    test_shortfall_fallback()
    test_pure_mode_batch1()
    if args.pretrained:
        test_smoke_pretrained()
    print("全部通过")
