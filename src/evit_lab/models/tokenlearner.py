"""TokenLearner: 可学习软路由的 token 浓缩。

用线性路由把 N 个 patch token 压缩为 L 个(L<<N)自适应 token: 每个输出 token
是全体 patch 的凸组合，组合权重由 patch 内容经 softmax 归一化得到。
与 ToMe/EViT 的训练免费不同，路由参数需要训练(本项目仅作 PoC)。
"""

import types
from types import SimpleNamespace

import torch
import torch.nn as nn


class TokenLearner(nn.Module):
    """N -> L 软浓缩模块。

    forward 数学: w = softmax_N(W_r x) ∈ (B, N, L)，在 token 维归一化；
                  out_l = Σ_n w_{n,l} x_n，即 einsum("bnl,bnc->blc") -> (B, L, C)。

    NOTE: softmax 作用在 token 维 N 而非槽位维 L —— 语义是「每个槽位在全体
    patch 上做权重和为 1 的分配」，输出数值尺度不随 N 变化。
    """

    def __init__(self, dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.router = nn.Linear(dim, num_tokens)  # patch 内容 -> L 个槽位的分配 logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        w = self.router(x)  # (B, N, L)
        w = torch.softmax(w, dim=1)  # 在 token 维归一化 = 各槽位对全体 patch 的凸组合系数
        return torch.einsum("bnl,bnc->blc", w, x)  # (B, L, C) 加权汇聚


def _make_condense_forward(condense: TokenLearner):
    """生成替换 block 前向的闭包: 完整执行该层 attn+MLP 后，在块尾浓缩。

    NOTE: 浓缩置于块尾，本层计算量不省，省的是其后所有层的 attention/MLP；
    CLS 不参与浓缩并保持在位置 0，输出 (B, 1+L, C)。
    """

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=False):
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        cls, patches = x[:, :1], x[:, 1:]  # CLS 单独摘出，不参与浓缩
        patches = condense(patches)  # (B, N-1, C) -> (B, L, C)
        return torch.cat([cls, patches], dim=1)

    return forward


def apply_tokenlearner(model: torch.nn.Module, num_tokens: int = 8,
                       split_at: int = 6) -> torch.nn.Module:
    """在第 split_at 个 block(1-based) 的块尾接入 TokenLearner 浓缩。

    Args:
        num_tokens: 浓缩后的槽位数 L
        split_at: 浓缩位置；该 block 自身计算不省，其后所有层在长度 1+L 的
            序列上运行
    Side effects:
        引入可训练参数(TokenLearner.router)，经 tokenlearner_params() 收集
        后注册进优化器训练
    """
    blocks = list(model.blocks)
    assert 0 < split_at < len(blocks), f"split_at 应在 1~{len(blocks)-1}"
    condense = TokenLearner(blocks[0].attn.qkv.out_features // 3, num_tokens)  # qkv 输出为 3C，除 3 还原嵌入维 C
    forward = _make_condense_forward(condense)
    blocks[split_at - 1].forward = types.MethodType(forward, blocks[split_at - 1])
    model._tokenlearner = condense  # 模块引用挂在 model 上，供 tokenlearner_params 与训练脚本访问
    model._tl_split_at = split_at
    return model


def tokenlearner_params(model: torch.nn.Module):
    """返回新增路由参数(唯一需要训练的部分)，供优化器单独注册。"""
    return list(model._tokenlearner.parameters())
