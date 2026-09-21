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
    """替换 timm Block.forward：attention -> 合并 -> MLP。

    自适应模式（strength>0）下，额外从该层注意力计算 CLS 行的归一化熵，
    熵低（CLS已聚焦/信息清晰）多删，熵高（混乱/难图）少删，
    每层实际合并数 r_t = r_base * (1 + strength * (1 - 2*H_norm))，
    strength=0 退化为固定 r 的原版 ToMe（可做消融对照）。
    """
    cfg = self._tome_cfg
    norm_x = self.norm1(x)

    r_t = cfg.r
    if cfg.strength > 0 and cfg.r > 0:
        r_t = _adaptive_r(norm_x, self.attn, cfg)
    attn_out = self.attn(norm_x, attn_mask=attn_mask, is_causal=is_causal)
    x = x + self.drop_path1(self.ls1(attn_out))

    if r_t > 0 and x.shape[1] > cfg.min_tokens:
        metric = _key_metric(norm_x, self.attn)
        x, merged = bipartite_merge(x, metric, r_t, protect_cls=True)
        cfg.info["r"].append(merged)

    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def _adaptive_r(norm_x: torch.Tensor, attn: torch.nn.Module, cfg) -> int:
    """用 CLS 注意力行的归一化熵决定本层合并数（batch 均值，逐层自适应）。

    熵的直觉: CLS 对各 token 的注意力越均匀(熵高)说明模型还没分清主次,
    此时少删; 越尖锐(熵低)说明关键 token 已明确, 其余冗余度高, 多删。
    返回本层的 r_t, 四舍五入到整数。
    """
    B, N, C = norm_x.shape
    H, d = attn.num_heads, C // attn.num_heads
    qkv = attn.qkv(norm_x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    # CLS 行注意力概率 (B, N): 只算一行, 开销可忽略
    cls_row = (q[:, :, 0:1] @ k.transpose(-1, -2)) * attn.scale
    p = cls_row.softmax(dim=-1).squeeze(2).mean(dim=1)        # (B, N)
    # 去掉 cls 自身与 patch token 里接近零的概率, 归一化到 [0,1]
    p = p[:, 1:]
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    h = -(p * (p + 1e-9).log()).sum(-1)                       # (B,) 熵
    h_norm = (h / torch.log(torch.tensor(float(p.shape[-1]), device=h.device))).mean()
    f = 1.0 + cfg.strength * (1.0 - 2.0 * h_norm.item())
    return max(1, round(cfg.r * f))


def apply_tome(model: torch.nn.Module, r: int, min_tokens: int = 8,
               strength: float = 0.0) -> torch.nn.Module:
    """给一个 timm VisionTransformer 挂上 ToMe 合并逻辑。

    Args:
        model: timm 创建的 ViT（要求每个 block 有 norm1/attn/mlp 标准结构）
        r: 每层基准删除 token 数；12层模型 r=8 时共删 96 个（197 -> 101）
        min_tokens: 序列短于该值后停止合并，防止把 token 合没了
        strength: 自适应强度 (0~1)。0=固定r(原版ToMe)；>0 时每层按注意力熵
                  缩放 r（本项目创新点：熵引导的逐层自适应预算）
    """
    for blk in model.blocks:
        blk._tome_cfg = SimpleNamespace(r=r, min_tokens=min_tokens,
                                        strength=strength, info={"r": []})
        blk.forward = types.MethodType(_tome_block_forward, blk)
    model._tome_r = r
    model._tome_strength = strength
    return model


def tome_token_counts(model: torch.nn.Module) -> list:
    """返回最近一次前向中每层实际合并的 token 数（验证/记录用）。"""
    return [sum(blk._tome_cfg.info["r"]) and blk._tome_cfg.info["r"][-1]
            if blk._tome_cfg.info["r"] else 0 for blk in model.blocks]
