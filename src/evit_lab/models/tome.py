"""ToMe (Token Merging, Bolya et al., ICLR 2023) 训练免费的 token 合并。

核心思想：ViT 每层前向后、MLP 前，把相似的 token 两两合并（取均值），
每层固定减少 r 个 token，全程不需要训练。

数学逻辑（双向软匹配 bipartite soft matching）:
    1. 把 N 个 token 按奇偶位分成集合 A(偶数位)、B(奇数位)，各约 N/2 个
    2. 用注意力的 K 投影作为 token 表征：metric = K(x)
       余弦相似度矩阵 S = cos(a_i, b_j)，即 L2 归一化后的 a @ b^T
    3. 对每个 a_i 取最相似的 b_j：edge(i) = argmax_j S[i, j]
    4. 只保留相似度最高的 r 条边，其余 A 中 token 原样保留
    5. 被选中的 r 个 a_i 与其目标 b_j 按元素取均值（scatter_reduce mean）

CLS token 处于偶数位（位置0），把它的相似度置为 -inf 保证永不被合并，
且合并输出时保留 token 按原始顺序排，CLS 始终在序列第 0 位（分类头依赖）。

设计说明：这里不依赖官方 tome 库（其魔改的 timm 版本与我们服务器上的
timm 1.x 冲突），只借用论文算法，在 timm 1.x 的 Block.forward 上挂钩。
注意：合并后 token 顺序会变，但 ViT 自注意力对顺序不敏感（位置信息已在
词向量里），对 deit3/vit 这类只有 cls 的模型是安全的；deit 原版权重带
dist token，合并会打乱其位置，因此 ToMe 实验统一用 deit3 权重。
"""

import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def bipartite_merge(x: torch.Tensor, metric: torch.Tensor, r: int,
                    protect_cls: bool = True):
    """执行一步双向软匹配合并。

    Args:
        x: (B, N, C) 当前 token 序列
        metric: (B, N, C) 用于算相似度的表征（K 投影）
        r: 本层要删除的 token 数
        protect_cls: 保护位置0的 cls token 不被合并
    Returns:
        (合并后的 (B, N-r, C), 实际合并数 r)
    """
    B, N, C = x.shape
    if r <= 0 or N <= 2:
        return x, 0

    with torch.no_grad():
        metric = F.normalize(metric, dim=-1)          # L2归一化 -> 内积=余弦相似度
        a_m, b_m = metric[:, ::2], metric[:, 1::2]    # A=偶数位, B=奇数位
        scores = a_m @ b_m.transpose(1, 2)            # (B, nA, nB) 余弦相似度矩阵

        if protect_cls:
            scores[:, 0, :] = float("-inf")           # cls 所在行永不被选中

        # 每个A token 找它最相似的 B token
        node_max, node_idx = scores.max(dim=-1)       # (B, nA)
        order = node_max.argsort(dim=-1, descending=True)
        r = min(r, order.shape[1] - (1 if protect_cls else 0))
        if r <= 0:
            return x, 0

        src_idx = order[..., :r, None]                # (B, r, 1)   要合并掉的A
        # 保留的A按原始顺序排（保证cls仍是最前面的那个），这是分类头正确的前提
        unm_idx = order[..., r:, None].sort(dim=-2).values
        dst_idx = node_idx.gather(-1, src_idx.squeeze(-1)).unsqueeze(-1)  # (B, r, 1)

    src, dst = x[:, ::2], x[:, 1::2]
    unm = src.gather(1, unm_idx.expand(-1, -1, C))    # 保留的token
    src_sel = src.gather(1, src_idx.expand(-1, -1, C))
    # 被选中的 src 并入各自的目标 dst（按元素均值）
    dst = dst.scatter_reduce(1, dst_idx.expand(-1, -1, C), src_sel,
                             reduce="mean", include_self=False)
    out = torch.cat([unm, dst], dim=1)                # (B, N-r, C)
    return out, r


def _key_metric(norm_x: torch.Tensor, attn: torch.nn.Module) -> torch.Tensor:
    """用该层注意力的 K 投影计算 token 表征（论文指定做法）。"""
    w = attn.qkv.weight                                # (3C, C)
    c = w.shape[1]
    metric = norm_x @ w[c:2 * c].t()                   # 取K部分的权重
    if attn.qkv.bias is not None:
        metric = metric + attn.qkv.bias[c:2 * c]
    return metric


def _tome_block_forward(self, x: torch.Tensor, attn_mask=None, is_causal=False):
    """替换 timm Block.forward：attention -> 合并 -> MLP。"""
    cfg = self._tome_cfg
    norm_x = self.norm1(x)
    attn_out = self.attn(norm_x, attn_mask=attn_mask, is_causal=is_causal)
    x = x + self.drop_path1(self.ls1(attn_out))

    if cfg.r > 0 and x.shape[1] > cfg.min_tokens:
        metric = _key_metric(norm_x, self.attn)
        x, merged = bipartite_merge(x, metric, cfg.r, protect_cls=True)
        cfg.info["r"].append(merged)

    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def apply_tome(model: torch.nn.Module, r: int, min_tokens: int = 8) -> torch.nn.Module:
    """给一个 timm VisionTransformer 挂上 ToMe 合并逻辑。

    Args:
        model: timm 创建的 ViT（要求每个 block 有 norm1/attn/mlp 标准结构）
        r: 每层删除的 token 数；12层模型 r=8 时共删 96 个（197 -> 101）
        min_tokens: 序列短于该值后停止合并，防止把 token 合没了
    """
    for blk in model.blocks:
        blk._tome_cfg = SimpleNamespace(r=r, min_tokens=min_tokens,
                                        info={"r": []})
        blk.forward = types.MethodType(_tome_block_forward, blk)
    model._tome_r = r
    return model


def tome_token_counts(model: torch.nn.Module) -> list:
    """返回最近一次前向中每层实际合并的 token 数（验证/记录用）。"""
    return [sum(blk._tome_cfg.info["r"]) and blk._tome_cfg.info["r"][-1]
            if blk._tome_cfg.info["r"] else 0 for blk in model.blocks]
