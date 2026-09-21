"""模型注册入口：所有模型（基线 + 之后的剪枝方法）统一从 build_model 创建。"""

from .baseline import TIMM_NAMES, create_baseline
from .tome import apply_tome


def build_model(name: str, num_classes: int = 1000, pretrained: bool = True,
                tome_r: int = 0):
    """统一模型工厂。

    Args:
        name: 模型别名，见 baseline.TIMM_NAMES
        tome_r: >0 时在该骨干上启用 ToMe 合并（每层删 tome_r 个 token）
    """
    if name in TIMM_NAMES:
        model = create_baseline(name, num_classes=num_classes, pretrained=pretrained)
        if tome_r > 0:
            apply_tome(model, tome_r)
        return model
    raise KeyError(f"未知模型 '{name}'，当前支持: {list(TIMM_NAMES)}")
