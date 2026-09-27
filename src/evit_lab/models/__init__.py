"""模型注册入口：基线与各剪枝方法统一经 build_model 创建，保证评测口径一致。"""

from .baseline import TIMM_NAMES, create_baseline
from .tome import apply_tome
from .evit import apply_evit
from .tokenlearner import apply_tokenlearner


def build_model(name: str, num_classes: int = 1000, pretrained: bool = True,
                tome_r: int = 0, evit_k: int = 0, tome_strength: float = 0.0,
                evit_start: int = 4, tokenlearner: int = 0,
                tome_force_schedule: str = None, tome_matcher: str = "bipartite",
                tome_budget: bool = True):
    """统一模型工厂。

    Args:
        name: 模型别名，见 baseline.TIMM_NAMES
        tome_r: >0 时启用 ToMe 合并（每层基准删 tome_r 个 token）
        evit_k: >0 时启用 EViT 选择（保留 evit_k-1 个 patch token + 1 个融合 token，cls 恒保留）
        tome_strength: ToMe 自适应强度（0=固定预算原版；>0=熵引导逐层自适应）
        evit_start: EViT 从第几层开始剪枝（论文默认4, 1=复现过早剪枝消融）
        tokenlearner: >0 时在第 6 个 block 后把 token 浓缩成该数量（路由需训练，仅 PoC）
        三种方法互斥，同时启用多个会抛 ValueError
    """
    methods = sum(v > 0 for v in (tome_r, evit_k, tokenlearner))
    if methods > 1:
        raise ValueError("ToMe / EViT / TokenLearner 互斥，一次只启用一个")
    if name in TIMM_NAMES:
        model = create_baseline(name, num_classes=num_classes, pretrained=pretrained)
        if tome_r > 0:
            apply_tome(model, tome_r, strength=tome_strength,
                       force_schedule=tome_force_schedule, matcher=tome_matcher,
                       budget=tome_budget)
        elif evit_k > 0:
            apply_evit(model, evit_k, start_layer=evit_start)
        elif tokenlearner > 0:
            apply_tokenlearner(model, num_tokens=tokenlearner)
        return model
    raise KeyError(f"未知模型 '{name}'，当前支持: {list(TIMM_NAMES)}")
