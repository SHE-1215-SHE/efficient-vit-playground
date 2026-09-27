"""EViT: 依据 CLS 注意力的 token 选择与融合。

每个启用层在 attention 之后、MLP 之前执行一次选择:
    1. 取该层注意力的 CLS 行: p = softmax(q_cls @ K^T / sqrt(d))，跨头平均，
       得到每个 token 被 CLS 关注的程度 cls_attn (B, N)
    2. CLS 自身不参与排序，保留 cls_attn 最高的 (K-1) 个 patch token
    3. 其余 token 按 cls_attn 加权平均融合为 1 个 token，拼接在序列末尾
    4. 输出序列 [cls(位置 0 不变), 按注意力降序的 K-1 个, fused] 共 K+1 个，
       首个启用层之后序列长度恒定

与 ToMe 的取舍:
    - ToMe 两两合并，信息保留更完整但匹配有额外开销；EViT 丢弃+浓缩，
      计算更省，信息损失更大
    - ToMe token 数逐层线性递减 (197 -> 101)；EViT 首个启用层后立即降到
      K+1 并保持
"""

import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def evit_select_and_fuse(x: torch.Tensor, cls_attn: torch.Tensor, keep_num: int):
    """按 CLS 注意力做 Top-K 选择 + 加权融合。

    Args:
        x: (B, N, C) token 序列，index 0 为 CLS
        cls_attn: (B, N) CLS 对各 token 的跨头平均注意力，与 x 逐位对齐
        keep_num: 保留的 patch token 数 K

    Returns:
        (out, n_removed): out (B, K+1, C) = [cls, top-(K-1) 注意力降序, fused]；
        n_removed = N-K，为被融合吸收的 token 数。N <= K+1 时原样返回、删除 0。

    数学: fused = Σ_i w_i x_i / Σ_i w_i，w_i 为被弃 token 的 cls_attn。
    采用注意力加权而非朴素均值: 判别性弱的 token 贡献被压低；
    分母 clamp_min 防止权重全零。
    """
    B, N, C = x.shape
    if N <= keep_num + 1:
        return x, 0
    order = cls_attn[:, 1:].argsort(dim=-1, descending=True)  # (B, N-1) patch 按注意力降序; CLS 自身不参与
    tokens = x[:, 1:]  # 剔除 CLS 后的 patch
    keep_idx = order[:, :keep_num - 1]  # (B, K-1) 保留集的原序列下标
    drop_idx = order[:, keep_num - 1:]  # (B, N-K) 待融合集的原序列下标
    kept = tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, K-1, C)
    dropped = tokens.gather(1, drop_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, N-K, C)
    w = cls_attn[:, 1:].gather(1, drop_idx)  # (B, N-K) 各待融合 token 的注意力权重
    fused = (dropped * w.unsqueeze(-1)).sum(dim=1, keepdim=True) \
        / w.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-9)  # (B, 1, C) 注意力加权均值
    out = torch.cat([x[:, :1], kept, fused], dim=1)  # [cls 保位 | top-(K-1) | fused]，非原序重排
    return out, dropped.shape[1]


def _evit_block_forward(self, x: torch.Tensor, attn_mask=None, is_causal=False):
    """被替换进 timm ViT block 的前向: 手工展开 attention 以同步获取 CLS 注意力行。

    动机: 选择依据必须与真实参与计算的注意力完全一致，故不复用 attn 模块的
    黑盒前向，而是共用同一份 q/k/v 分两路: CLS 行 softmax(选择依据) 与
    SDPA(实际注意力计算，fused kernel 省显存)。

    数据流: x (B, N, C) -> attn + 残差 -> 选择+融合 (N -> K+1) -> MLP + 残差。
    """
    cfg = self._evit_cfg
    attn_mod = self.attn
    B, N, C = x.shape
    H, d = attn_mod.num_heads, C // attn_mod.num_heads

    norm_x = self.norm1(x)
    qkv = attn_mod.qkv(norm_x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if hasattr(attn_mod, "q_norm"):
        q = attn_mod.q_norm(q)  # 黑盒展开后需手动补 q/k LayerNorm，否则注意力分布失真
    if hasattr(attn_mod, "k_norm"):
        k = attn_mod.k_norm(k)
    cls_row = (q[:, :, 0:1] @ k.transpose(-1, -2)) * attn_mod.scale  # (B, H, 1, N) CLS 行 logits
    cls_attn = cls_row.softmax(dim=-1).squeeze(2).mean(dim=1)  # (B, N) 跨头平均 = 各 token 的 CLS 关注度
    attn_out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=attn_mod.attn_drop.p if self.training else 0.0)  # 实际注意力走 SDPA fused 实现
    attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
    attn_out = attn_mod.proj_drop(attn_mod.proj(attn_out))
    x = x + self.drop_path1(self.ls1(attn_out))
    if cfg.keep_num > 0 and cfg.layer_idx >= cfg.start_layer:
        # start_layer 之前不剪: 早层 CLS 行判别力弱，先让注意力成熟(EViT 论文默认第 4 层起)
        x, removed = evit_select_and_fuse(x, cls_attn, cfg.keep_num)
        cfg.info["removed"].append(removed)

    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def apply_evit(model: torch.nn.Module, keep_num: int, start_layer: int = 4) -> torch.nn.Module:
    """给 timm ViT 打 EViT 补丁: 逐 block 替换前向。

    每个 block 挂独立 cfg(keep_num/layer_idx/start_layer/info)；仅
    layer_idx >= start_layer 的层执行选择-融合。model._evit_k/_evit_start
    供评测脚本读取。
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
    """各 block 最近一次前向实际融合移除的 token 数；未启用层计 0。

    NOTE: info 跨前向累积，取末位即最近一次。
    """
    return [blk._evit_cfg.info["removed"][-1] if blk._evit_cfg.info["removed"] else 0
            for blk in model.blocks]
