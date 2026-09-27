"""MPM: Mutual Pair Merging (arXiv 2604.05718) 匹配后端。

核心思想: 只合并"互为最近邻"的 token 对 (i 的最近邻是 j 且 j 的最近邻是 i),
配对精确但数量数据相关。与 ToMe 二分匹配 (BSM) 同为训练免费合并, 对比的是匹配算法本身。

两种模式:
    budget=True  : 互配对按相似度取前 r 对, 输出恰好 N-r (与 ToMe 同预算可比, 可批推理)。
                   互配对不足 r 时整批回退到 ToMe 二分匹配 (保证输出形状, 回退次数记录在 info)。
    budget=False : 论文原始模式, 合并全部互配对, 输出长度数据相关, 仅支持 batch=1。

互配对天然不相交: 若 (i,j) 互配, 则 j 的最近邻已被 i 占据, 不可能再与 k 互配,
因此无需冲突处理, 直接 scatter 合并即可。
"""

import torch
import torch.nn.functional as F


def mutual_pair_merge(x: torch.Tensor, metric: torch.Tensor, r: int,
                      protect_cls: bool = True, budget: bool = True,
                      min_tokens: int = 8):
    """互最近邻配对合并。

    Args:
        x: (B, N, C) token 序列
        metric: (B, N, C) 相似度度量 (K), 与 x 逐 token 对应
        r: 预算 (budget=True 时生效)
        protect_cls: CLS 永不参与配对
        budget: True=固定预算模式, False=论文原始模式
        min_tokens: 序列长度低于该值时停止合并

    Returns:
        (merged_x, n_merged, n_pairs_found)
        budget 模式下输出长度恒为 N-r; 若整批互配对不足 r, 回退二分匹配 (同样 N-r)。

    数学: nn = argmax_j cos(m_i, m_j); mutual(i) <=> nn(nn(i)) == i,
    即两 token 互为最近邻; 合并取均值, 与 ToMe 的聚合方式一致。
    """
    from .tome import bipartite_merge  # 延迟导入避免循环依赖

    B, N, C = x.shape
    if N <= min_tokens or (budget and r <= 0):
        return x, 0, 0
    if not budget and B > 1:
        raise ValueError("pure-MPM 输出长度数据相关, 仅支持 batch=1")

    with torch.no_grad():
        m = F.normalize(metric, dim=-1)
        off = 1 if protect_cls else 0
        m1 = m[:, off:]                        # (B, M, C) 非 CLS token
        M = m1.shape[1]
        sim = m1 @ m1.transpose(1, 2)          # (B, M, M) 余弦亲和
        sim = sim.masked_fill(
            torch.eye(M, dtype=torch.bool, device=x.device).unsqueeze(0),
            float("-inf"))                     # 屏蔽自身, 否则 argmax 选中自己产生假互配对
        nn = sim.argmax(-1)                    # (B, M) 各自最近邻
        idx = torch.arange(M, device=x.device).unsqueeze(0).expand(B, -1)
        mutual = nn.gather(1, nn) == idx       # (B, M) 互最近邻掩码
        pair_sim = sim.gather(1, nn.unsqueeze(-1)).squeeze(-1)  # (B, M)
        pair_sim = pair_sim.masked_fill(~mutual, float("-inf"))
        n_pairs = int(mutual.sum(dim=1).min())  # 整批最少互配对数; int() 触发一次 GPU->CPU 同步

        if not budget:
            # 论文原始模式: 合并全部互配对 (batch=1)
            src_i = idx[0][mutual[0]]
            dst_j = nn[0][mutual[0]]
            n_merged = int(mutual[0].sum())
            if n_merged == 0:
                return x, 0, 0
            src_f = (src_i + off).unsqueeze(-1).expand(-1, C)
            dst_f = (dst_j + off).unsqueeze(-1).expand(-1, C)
            src_v = x[0].gather(0, src_f)
            x = x.scatter_reduce(1, dst_f.unsqueeze(0), src_v.unsqueeze(0),
                                 reduce="mean", include_self=False)
            out = _drop_tokens(x[0], src_i + off, N - n_merged).unsqueeze(0)
            return out, n_merged, n_merged

        # budget 模式: 互配对不足 r 时整批回退 (保证形状, 记录回退)
        if n_pairs < r:
            xb, merged = bipartite_merge(x, metric, r, protect_cls=protect_cls)
            return xb, merged, n_pairs

        top = pair_sim.topk(r, dim=-1).indices       # (B, r) 全为有效互配对
        src_f = (top + off)                          # (B, r) 源 token 索引
        dst_f = nn.gather(1, top) + off              # (B, r) 目标 token 索引

    src_v = x.gather(1, src_f.unsqueeze(-1).expand(-1, -1, C))
    x = x.scatter_reduce(1, dst_f.unsqueeze(-1).expand(-1, -1, C), src_v,
                         reduce="mean", include_self=False)
    out = _drop_tokens(x, src_f, N - r)              # 源 token 从序列中移除
    return out, r, n_pairs


def _drop_tokens(x: torch.Tensor, src_idx: torch.Tensor, keep_len: int) -> torch.Tensor:
    """从序列中移除 src_idx 指向的 token, 保留其余 (保序)。

    不用 in-place scatter_/布尔索引 (fvcore JIT 追踪不支持), 改用
    "删除标记 + stable argsort + gather" 构造, 对追踪图友好。
    """
    N = x.shape[-2]
    ar = torch.arange(N, device=x.device)
    drop = (src_idx.unsqueeze(-1) == ar).any(dim=-2)     # (..., N) 是否被删除
    order = drop.to(torch.uint8).argsort(dim=-1, stable=True)  # 保留者在前, 保原序
    idx = order[..., :keep_len].unsqueeze(-1)              # (..., keep_len, 1)
    idx = idx.expand(*idx.shape[:-2], -1, x.shape[-1])
    return x.gather(-2, idx)
