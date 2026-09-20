"""DeiT / ViT 基线模型加载。

用 timm 库直接加载facebook官方发布的预训练权重（在ImageNet-1k上训练好的），
我们不做从头训练，只做「加载权重 -> 评测加速效果」，这也是一个月工期内
唯一现实的路线。后续的剪枝方法（DynamicViT/EViT/ToMe）都基于这些骨干网络。
"""

import torch
import timm

# 项目内统一别名 -> timm 模型名
# 之所以用别名：之后剪枝方法也要挂在这批骨干上，统一入口方便对比
TIMM_NAMES = {
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
}


def create_baseline(name: str, num_classes: int = 1000, pretrained: bool = True) -> torch.nn.Module:
    """按别名创建一个(预训练)DeiT/ViT模型。

    Args:
        name: TIMM_NAMES 里的别名，如 "deit_small"
        num_classes: 分类数，ImageNet-1k 是 1000
        pretrained: 是否下载官方预训练权重
    Returns:
        nn.Module，输入 shape 为 (B, 3, 224, 224)
    """
    if name not in TIMM_NAMES:
        raise KeyError(f"未知模型别名 '{name}'，当前支持: {list(TIMM_NAMES)}")
    model = timm.create_model(
        TIMM_NAMES[name],
        pretrained=pretrained,
        num_classes=num_classes,
    )
    model.eval()
    return model
