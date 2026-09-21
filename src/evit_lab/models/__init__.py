"""模型注册入口：所有模型（基线 + 之后的剪枝方法）统一从 build_model 创建。"""

from .baseline import TIMM_NAMES, create_baseline
from .tome import apply_tome
from .evit import apply_evit


def build_model(name: str, num_classes: int = 1000, pretrained: bool = True,
                tome_r: int = 0, evit_k: int = 0, tome_strength: float = 0.0):
    """统一模型工厂。

    Args:
        name: 模型别名，见 baseline.TIMM_NAMES
        tome_r: >0 时启用 ToMe 合并（每层基准删 tome_r 个 token）
        evit_k: >0 时启用 EViT 选择（每层保留 evit_k 个普通 token）
        tome_strength: ToMe 自适应强度（0=固定预算原版；>0=熵引导逐层自适应）
        ToMe 与 EViT 互斥，同时给值会报错
    """
    if tome_r > 0 and evit_k > 0:
        raise ValueError("ToMe 与 EViT 不能同时启用，只设一个")
    if name in TIMM_NAMES:
        model = create_baseline(name, num_classes=num_classes, pretrained=pretrained)
        if tome_r > 0:
            apply_tome(model, tome_r, strength=tome_strength)
        elif evit_k > 0:
            apply_evit(model, evit_k)
        return model
    raise KeyError(f"未知模型 '{name}'，当前支持: {list(TIMM_NAMES)}")
