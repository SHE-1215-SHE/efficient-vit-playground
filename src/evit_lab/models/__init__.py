"""模型注册入口：所有模型（基线 + 之后的剪枝方法）统一从 build_model 创建。"""

from .baseline import TIMM_NAMES, create_baseline


def build_model(name: str, num_classes: int = 1000, pretrained: bool = True):
    """统一模型工厂。后续 DynamicViT/EViT/ToMe 会在这里扩展分支。"""
    if name in TIMM_NAMES:
        return create_baseline(name, num_classes=num_classes, pretrained=pretrained)
    raise KeyError(f"未知模型 '{name}'，当前支持: {list(TIMM_NAMES)}")
