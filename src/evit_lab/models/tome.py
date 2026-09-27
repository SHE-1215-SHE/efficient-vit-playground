"""ToMe (Token Merging) 与熵引导调度 —— 项目核心创新点。

基础 ToMe: 每个 block 在 attention 之后、MLP 之前，将相似 token 两两合并
(取均值)，每层减少 r 个 token，训练免费。

本项目扩展 —— 熵引导调度: 用第 3 个 block(index 2) 的 CLS 注意力归一化熵
做一次性决策，从「前重 / 后重」两套逐层倍率模板中选一套，换算成逐层删除数
后广播给后续所有层；决策之后的层纯查表，不再产生新的 GPU->CPU 同步。
见 _cls_entropy / _decide_schedule / _schedule_rs。
"""

import math
import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def bipartite_merge(x: torch.Tensor, metric: torch.Tensor, r: int,
                    protect_cls: bool = True):
    """二分图软匹配合并 (ToMe BSM 简化实现)。

    将 token 按下标奇偶划分成两组(组内不配对、组间全连接): 每个偶位 token
    取其在奇位组中的余弦最近邻作为候选配对，按配对相似度从高到低执行 r 次
    合并，其余偶位 token 原样保留。

    Args:
        x: (B, N, C) token 序列，index 0 为 CLS
        metric: (B, N, C) 相似度度量，与 x 逐 token 对应；内部 L2 归一化，
            点积即余弦相似度
        r: 目标合并数；实际合并数 r' = min(r, N//2 - protect_cls) 可能小于 r
        protect_cls: True 时 CLS 永不作为合并源，保持序列首位

    Returns:
        (merged_x, r')：merged_x (B, N-r', C)，排列为 [未合并偶位 token(保序),
        奇位 token(含被合并更新)] —— 非保序重排，下游不得依赖原始位置序；
        r' 为实际合并数。

    数学: sim = cos(a_i, b_j)；被选中配对的 dst_j <- mean(指派到 j 的全部 src)
    """
    B, N, C = x.shape
    if r <= 0 or N <= 2:
        return x, 0

    with torch.no_grad():
        # 相似度矩阵与索引选择不建图: 索引不可导，且 (B, N/2, N/2) 中间量无需保留梯度
        metric = F.normalize(metric, dim=-1)  # 单位化后点积 = 余弦相似度
        a_m, b_m = metric[:, ::2], metric[:, 1::2]  # 奇偶二分，构成二分图
        scores = a_m @ b_m.transpose(1, 2)  # (B, N/2, N/2) 组间余弦相似度

        if protect_cls:
            scores[:, 0, :] = float("-inf")  # INVARIANT: CLS(index 0) 候选全屏蔽，必落入未合并集合
        node_max, node_idx = scores.max(dim=-1)  # 各偶位 token 的最佳配对相似度与对手下标
        order = node_max.argsort(dim=-1, descending=True)  # 按相似度降序 = 最确信的配对先合并
        r = min(r, order.shape[1] - (1 if protect_cls else 0))  # 保护 CLS 时上限为 |A|-1
        if r <= 0:
            return x, 0

        src_idx = order[..., :r, None]  # (B, r, 1) 待合并的源(偶位)
        unm_idx = order[..., r:, None].sort(dim=-2).values  # 未合并源按原下标升序，保持相对顺序
        dst_idx = node_idx.gather(-1, src_idx.squeeze(-1)).unsqueeze(-1)  # (B, r, 1) 各源的目标(奇位)

    src, dst = x[:, ::2], x[:, 1::2]
    unm = src.gather(1, unm_idx.expand(-1, -1, C))  # 未合并源，已按原序排列
    src_sel = src.gather(1, src_idx.expand(-1, -1, C))  # (B, r, C) 被合并的源
    dst = dst.scatter_reduce(1, dst_idx.expand(-1, -1, C), src_sel,
                             reduce="mean", include_self=False)  # dst <- 指派源的均值(不含 dst 原值)
    out = torch.cat([unm, dst], dim=1)  # [未合并偶位(保序) | 奇位组]，总长 N - r'
    return out, r


def _tome_block_forward(self, x: torch.Tensor, attn_mask=None, is_causal=False):
    """被替换进 timm ViT block 的前向: 在 attention 与 MLP 之间插入合并。

    数据流: x (B, N, C) -> attn + 残差 -> 合并 (N -> N-r_t) -> MLP + 残差
    -> (B, N-r_t, C)。
    合并位置的选择: attention 之后 token 间信息已完成交换，此时合并的信息
    损失最小；MLP 逐 token 作用，合并后直接省去其计算量。
    """
    cfg = self._tome_cfg
    norm_x = self.norm1(x)
    attn_out = self.attn(norm_x, attn_mask=attn_mask, is_causal=is_causal)
    x = x + self.drop_path1(self.ls1(attn_out))

    if cfg.r > 0 and x.shape[1] > cfg.min_tokens:
        # 正常路径下 qkv hook 已缓存本层 attention 内部的 qkv，免于重算
        # (重算会使 qkv 线性层开销近乎翻倍)；仅 hook 未触发时兜底重算
        qkv = getattr(cfg, "_qkv", None)
        if qkv is None:
            qkv = self.attn.qkv(norm_x)
        C = qkv.shape[-1] // 3
        metric = qkv[..., C:2 * C]  # 取 K 作相似度度量(ToMe 消融: K 优于输入 X 与注意力输出)
        if cfg.strength > 0 and not cfg.decided and cfg.later is not None:  # 仅 index 2 决策: 0/1 层 later 为 None 直接跳过, 全模型恰 1 次 GPU->CPU 同步
            _decide_schedule(self, qkv)
        r_t = cfg.plan[0]  # 查表得本层删除数(固定预算下恒为基准 r)
        if getattr(cfg, "matcher", "bipartite") == "mutual":
            from .mutual_pair import mutual_pair_merge
            x, merged, n_pairs = mutual_pair_merge(x, metric, r_t, protect_cls=True,
                                                   min_tokens=cfg.min_tokens,
                                                   budget=getattr(cfg, "budget", True))
            cfg.info.setdefault("pairs", []).append(n_pairs)
        else:
            x, merged = bipartite_merge(x, metric, r_t, protect_cls=True)
        cfg.info["r"].append(merged)
    cfg._qkv = None  # 用毕即清: 防止缓存的大张量滞留显存，也杜绝后续误用过期度量

    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x

# 逐层删除数模板: 相对基准 r 的倍率。FRONT 前重后轻(浅层激进合并)，BACK 为其逆序。
# 长度 10 = 12 层骨干 - 前两层(固定 r): 自 index 2 起逐层生效
_SCHED_FRONT = (1.7, 1.6, 1.5, 1.4, 1.2, 1.0, 0.7, 0.4, 0.3, 0.2)
_SCHED_BACK = tuple(reversed(_SCHED_FRONT))


def _cls_entropy(qkv: torch.Tensor, attn: torch.nn.Module) -> torch.Tensor:
    """CLS 注意力分布的归一化熵(对 batch 取均值，返回标量 tensor)。

    数学: p = mean_h softmax(q_cls K^T / sqrt(d))  (B, N)，去掉 CLS 自身项后重归一
          H = -(Σ_j p_j log p_j) / log(N-1) ∈ [0, 1]

    H 低 <=> CLS 注意力集中于少数 token(前景判别明确)；
    H 高 <=> 接近均匀分布(语义尚未收敛)。

    Args:
        qkv: (B, N, 3C) 本层 qkv 投影(hook 缓存，复用避免重算)
        attn: 本层 attention 模块，提供 num_heads 与 scale

    NOTE: 调用方对返回值 .item() 会触发 GPU->CPU 同步；该代价只发生在未
    decided 的层上(见 _decide_schedule)。
    """
    B, N, C3 = qkv.shape
    C = C3 // 3
    H, d = attn.num_heads, C // attn.num_heads
    qkv3 = qkv.reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
    q, k = qkv3[0], qkv3[1]
    cls_row = (q[:, :, 0:1] @ k.transpose(-1, -2)) * attn.scale  # (B, H, 1, N) CLS 行注意力 logits
    p = cls_row.softmax(dim=-1).squeeze(2).mean(dim=1)  # softmax 后跨头平均 -> (B, N)
    p = p[:, 1:]  # 去掉 CLS 对自身的注意力项，否则该项会系统性压低熵的区分度
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # 重归一化，补回被移除的质量
    h = -(p * (p + 1e-9).log()).sum(-1)  # (B,) 熵(自然对数)
    return (h / math.log(p.shape[-1])).mean()  # 除以 log(N-1) 归一化到 [0,1]，使跨层/跨序列长度的熵可比


def _schedule_rs(template, base_r: int, strength: float) -> list:
    """把逐层倍率模板换算为整数删除数序列。

    数学: blended_i = 1 + strength*(s_i - 1)  (strength=0 退化为每层固定 base_r)
          rs_i = round(Σ_{j<=i} blended_j*base_r) - round(Σ_{j<i} blended_j*base_r)

    采用「累计和取整后作差」而非逐层独立取整: 保证 Σ rs_i == round(Σ blended_i*base_r)，
    总删除量不因舍入而漂移；cum_f 单调不减，其取整序列亦不减，故 rs_i >= 0。

    Returns:
        list[int]，长度与 template 一致，为逐层删除数
    """
    blended = [1.0 + strength * (s - 1.0) for s in template]
    rs, prev_cum, cum_f = [], 0.0, 0.0
    for f in blended:
        cum_f += f * base_r
        rs.append(int(round(cum_f)) - int(round(prev_cum)))
        prev_cum = cum_f
    return rs


def _decide_schedule(blk, qkv: torch.Tensor):
    """熵引导调度的一次性决策: 计算熵 -> 选模板 -> 生成逐层 r 并向下广播。

    设计动机:
    - 决策点固定在 block index 2: 更早的层注意力尚未形成语义，熵判别力不足；
      决策越早，受调度覆盖的层越多。
    - 决策一次、后续层纯查表: 决策后所有层只读 cfg.plan，无额外同步开销。
    - 调用约束: 仅 index 2 (later 非空) 会进入本函数；index 0/1 的 cfg.later
      为 None，在调用方即被跳过，保证全模型恰 1 次 GPU->CPU 同步。

    副作用: 写入 later 各 cfg 的 plan/decided；在 model_ref._tome_schedule
    记录 {"entropy", "name", "rs"} 供事后分析。
    """
    cfg = blk._tome_cfg
    if cfg.force:
        h, name = None, cfg.force  # 指定模板: 跳过熵计算，无 GPU 同步
    else:
        # .item() 触发 GPU->CPU 同步；调用方已保证仅 index 2 进入本函数，
        # 故无 force 时全模型唯一次同步发生在这里
        h = _cls_entropy(qkv, blk.attn).item()
        name = "front" if h < 0.5 else "back"
    template = _SCHED_FRONT if name == "front" else _SCHED_BACK
    rs = _schedule_rs(template, cfg.r, cfg.strength)
    if cfg.later is not None:
        for c, rr in zip(cfg.later, rs):
            c.plan = [rr]
            c.decided = True  # 锁定后继层: 不再进入决策分支，纯查表
        cfg.model_ref._tome_schedule = {"entropy": h, "name": name, "rs": rs}


def apply_tome(model: torch.nn.Module, r: int, min_tokens: int = 8,
               strength: float = 0.0, force_schedule: str = None,
               matcher: str = "bipartite", budget: bool = True) -> torch.nn.Module:
    """给 timm ViT 打 ToMe 补丁: 逐 block 替换前向并挂 qkv 缓存 hook。

    机制:
    - 每个 block 一份 cfg(SimpleNamespace) 记录本层计划与统计；index 2 的
      cfg.later 指向其后全部 cfg，是熵调度的广播入口。
    - qkv forward hook 缓存 attention 内部已算出的 qkv (B, N, 3C)，复用其 K
      作为度量，避免 metric 重算令 qkv 线性层开销近乎翻倍。
    - 实例级 monkey-patch(blk.forward)，不修改 timm 类定义。
    - model._tome_r/_tome_strength/_tome_matcher/_tome_schedule 供评测脚本读取。

    Args:
        r: 每层基准删除数
        min_tokens: 序列长度低于该值时本层跳过合并
        strength: 0=固定预算(每层 r)；>0 启用熵引导调度，值为模板插值强度
        force_schedule: "front"/"back"，指定时跳过熵决策(用于消融)
        matcher: "bipartite"(BSM) 或 "mutual"(互最近邻配对)
        budget: 传入 mutual 匹配器的预算开关；bipartite 恒为固定预算
    """
    blocks = list(model.blocks)
    cfgs = [SimpleNamespace(r=r, min_tokens=min_tokens, strength=strength,
                            info={"r": []}, _qkv=None,
                            plan=[r],  # 默认计划: 每层固定删 r；熵调度生效后按层覆写
                            decided=False,
                            later=None, model_ref=model, force=force_schedule,
                            matcher=matcher, budget=budget)
            for _ in blocks]
    if len(blocks) > 2:
        cfgs[2].later = cfgs[2:]  # 广播链: index 2 决策后覆写自身及全部后继的 plan
    for blk, cfg in zip(blocks, cfgs):
        blk._tome_cfg = cfg
        blk.forward = types.MethodType(_tome_block_forward, blk)
        blk._tome_hook = blk.attn.qkv.register_forward_hook(
            lambda m, i, o, _cfg=cfg: setattr(_cfg, "_qkv", o))  # 以默认参数绑定各自 cfg，规避循环变量晚绑定
    model._tome_r = r
    model._tome_strength = strength
    model._tome_matcher = matcher
    model._tome_schedule = None  # 熵调度结果由 _decide_schedule 在首次前向时写入
    return model


def tome_token_counts(model: torch.nn.Module) -> list:
    """各 block 最近一次前向实际删除的 token 数。

    NOTE: cfg.info["r"] 跨前向累积追加，此处取末位即最近一次；未合并的层计 0。
    """
    return [sum(blk._tome_cfg.info["r"]) and blk._tome_cfg.info["r"][-1]
            if blk._tome_cfg.info["r"] else 0 for blk in model.blocks]
