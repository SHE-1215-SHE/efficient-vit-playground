"""EViT (Expediting Vision Transformers via Token Re-centering, Pan et al., 2021)。

核心思想：与 ToMe 的"合并"不同，EViT 直接**丢弃**注意力低的 token，只保留
CLS 最关注的 top-K 个，被丢弃的 token 融合成 1 个"融合 token"接在序列末尾，
把背景信息浓缩保留。每层都做一次选择，序列长度从 197 稳定到 K+1。

数学逻辑:
    1. 取该层注意力的 CLS 行：p = softmax(q_cls @ K^T / sqrt(d))，跨头平均
       得到每个 token 被 CLS 关注的程度 cls_attn (B, N)
    2. 按 cls_attn 从高到低排序（CLS 自身不参与），保留 top-(K-1) 个
    3. 其余 token 均值融合成 1 个 token，拼在 [cls, 保留token] 之后
    4. 输出序列: [cls(0位不变), 按注意力排序的K-1个, fused] 共 K+1 个

与 ToMe 的关键区别（面试常问）:
    - ToMe 合并（信息保留更完整，有额外计算开销）；EViT 丢弃+浓缩（更快，信息损失略大）
    - ToMe token 数线性递减 (197 -> 101)；EViT 第1层后立刻到 K+1 并保持

同样挂钩 timm 1.x Block.forward，不依赖官方 EViT 仓库（其基于旧版 timm
且绑定 deit 蒸馏权重）；实验统一用 deit3 权重（纯 cls）保证公平。
"""

import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def evit_select_and_fuse(x: torch.Tensor, cls_attn: torch.Tensor, keep_num: int):
    """按 CLS 注意力保留 top-K token，其余按注意力加权融合成 1 个 token。

    Args:
        x: (B, N, C) 当前 token 序列
        cls_attn: (B, N) 每个 token 被 CLS 关注的程度（softmax 后跨头平均）
        keep_num: 保留的普通 token 数（不含 cls；输出共 keep_num+1 个）
    Returns:
        (新序列 (B, keep_num+1, C), 实际丢弃的 token 数)
    """
    B, N, C = x.shape
    if N <= keep_num + 1:
        return x, 0
    order = cls_attn[:, 1:].argsort(dim=-1, descending=True)  # 0..N-2, 指向去掉cls的序列
    tokens = x[:, 1:]                                         # (B, N-1, C) 不含cls
    keep_idx = order[:, :keep_num - 1]                        # 保留: 注意力最高的K-1个
    drop_idx = order[:, keep_num - 1:]                        # 丢弃: 其余
    kept = tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, C))
    dropped = tokens.gather(1, drop_idx.unsqueeze(-1).expand(-1, -1, C))
    # 论文式加权融合: 注意力越高的被丢弃token贡献越大(区别于简单均值)
    w = cls_attn[:, 1:].gather(1, drop_idx)                   # (B, n_drop)
    fused = (dropped * w.unsqueeze(-1)).sum(dim=1, keepdim=True) \
        / w.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-9)  # (B,1,1)否则广播错位
    out = torch.cat([x[:, :1], kept, fused], dim=1)           # cls 恒在第0位
    return out, dropped.shape[1]


def _evit_block_forward(self, x: torch.Tensor, attn_mask=None, is_causal=False):
    """替换 timm Block.forward：手动拆开 attention 以拿到 CLS 注意力行。"""
    cfg = self._evit_cfg
    attn_mod = self.attn
    B, N, C = x.shape
    H, d = attn_mod.num_heads, C // attn_mod.num_heads

    norm_x = self.norm1(x)
    # 沿用 timm Attention 内部的 qkv 拆分流程（含可选 qk_norm）
    qkv = attn_mod.qkv(norm_x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if hasattr(attn_mod, "q_norm"):
        q = attn_mod.q_norm(q)
    if hasattr(attn_mod, "k_norm"):
        k = attn_mod.k_norm(k)

    # CLS 行注意力：选择依据（只算一行，开销可忽略）
    # 注意用 0:1 保留单元素维度 (B,H,1,d)，否则 squeeze 后广播会错位
    cls_row = (q[:, :, 0:1] @ k.transpose(-1, -2)) * attn_mod.scale  # (B, H, 1, N)
    cls_attn = cls_row.softmax(dim=-1).squeeze(2).mean(dim=1)        # (B, N)

    # 正常走注意力输出（与 timm 内部实现一致）
    attn_out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=attn_mod.attn_drop.p if self.training else 0.0)
    attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
    attn_out = attn_mod.proj_drop(attn_mod.proj(attn_out))
    x = x + self.drop_path1(self.ls1(attn_out))

    # 论文消融: 前几层的CLS注意力不可靠, 过早剪枝精度断崖下跌, 故从
    # start_layer 层起才做选择（默认第4层, 与论文一致）
    if cfg.keep_num > 0 and cfg.layer_idx >= cfg.start_layer:
        x, removed = evit_select_and_fuse(x, cls_attn, cfg.keep_num)
        cfg.info["removed"].append(removed)

    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def apply_evit(model: torch.nn.Module, keep_num: int, start_layer: int = 4) -> torch.nn.Module:
    """给 timm VisionTransformer 挂上 EViT 选择逻辑。

    Args:
        model: timm 创建的 ViT
        keep_num: 保留的普通 token 数；K=100 时最终序列 101（cls+100+融合），
                  与 ToMe r=8 (197->101) 的算力预算对齐，便于公平对比
        start_layer: 从第几层开始剪枝（论文默认第4层; 设1可复现"过早剪枝"消融）
    """
    for i, blk in enumerate(model.blocks):
        blk._evit_cfg = SimpleNamespace(keep_num=keep_num, layer_idx=i,
                                        start_layer=start_layer,
                                        info={"removed": []})
        blk.forward = types.MethodType(_evit_block_forward, blk)
    model._evit_k = keep_num
    model._evit_start = start_layer
    return model


def evit_removed_counts(model: torch.nn.Module) -> list:
    """返回最近一次前向中每层实际丢弃的 token 数（验证/记录用）。"""
    return [blk._evit_cfg.info["removed"][-1] if blk._evit_cfg.info["removed"] else 0
            for blk in model.blocks]
