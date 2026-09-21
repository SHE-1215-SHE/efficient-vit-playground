"""TokenLearner (Ryoo et al., NeurIPS 2021) 最小可运行集成（PoC）。

核心思想：与"筛选/合并已有 token"不同，TokenLearner 用一个可学习的软路由
把 N 个 patch token **浓缩**成 L 个（如 8 个）自适应 token：
    w = softmax(MLP(X), dim=tokens)        # (B, N, L) 每个token对每个槽位的权重
    Z_l = sum_i w[i,l] * X_i               # 加权汇聚 -> (B, L, C)
即每个新 token 是全体 patch 的可学习加权组合（注意力池化的特例，
但权重由 token 内容经 MLP 产生而非 QK 相似度）。

架构切法（论文做法）：前一半 block 正常自注意力 -> TokenLearner 浓缩 197->L
-> 后一半 block 在 L+1 个 token 上继续（cls 保留在第0位不参与浓缩）。

PoC 定位说明：TokenLearner 的路由参数必须训练，官方无 PyTorch 权重。
本实现用于小数据（ImageNet-V2 划分出的子集）微调验证"token 浓缩"思想，
不追求 SOTA；训练脚本 scripts/train_tokenlearner.py。
"""

import types
from types import SimpleNamespace

import torch
import torch.nn as nn


class TokenLearner(nn.Module):
    """把 (B, N, C) 的 patch token 浓缩成 (B, L, C)。

    参数量只有 C*L + L（一个小线性层），训练的就是这个路由。
    """

    def __init__(self, dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.router = nn.Linear(dim, num_tokens)  # token内容 -> 各槽位权重

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        w = self.router(x)                        # (B, N, L)
        w = torch.softmax(w, dim=1)               # 对 token 维归一化 = 软分配
        return torch.einsum("bnl,bnc->blc", w, x)  # (B, L, C) 加权汇聚


def _make_condense_forward(condense: TokenLearner):
    """生成一个 block forward：先正常注意力，再在 MLP 前做浓缩。

    浓缩放在 block 内（attention后/MLP前），这样该 block 的 MLP 只处理
    L+1 个 token，立刻开始省算力。
    """

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=False):
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        cls, patches = x[:, :1], x[:, 1:]
        patches = condense(patches)
        return torch.cat([cls, patches], dim=1)

    return forward


def apply_tokenlearner(model: torch.nn.Module, num_tokens: int = 8,
                       split_at: int = 6) -> torch.nn.Module:
    """把 TokenLearner 插到第 split_at 层 block 的 MLP 之前。

    Args:
        model: timm ViT
        num_tokens: 浓缩后保留的 token 数 L（论文用 8）
        split_at: 在第几层做浓缩（12层模型取 6 = 中点，前半理解后半汇总）
    """
    blocks = list(model.blocks)
    assert 0 < split_at < len(blocks), f"split_at 应在 1~{len(blocks)-1}"
    condense = TokenLearner(blocks[0].attn.qkv.out_features // 3, num_tokens)
    forward = _make_condense_forward(condense)
    blocks[split_at - 1].forward = types.MethodType(forward, blocks[split_at - 1])
    model._tokenlearner = condense
    model._tl_split_at = split_at
    return model


def tokenlearner_params(model: torch.nn.Module):
    """返回需要训练的参数（路由 + 后半段norm/head可按需加入）。"""
    return list(model._tokenlearner.parameters())
