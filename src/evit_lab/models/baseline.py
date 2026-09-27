"""DeiT / ViT 基线模型加载。

通过 timm 加载官方发布的 ImageNet-1k 预训练权重。本项目不从头训练，
评测流程为「加载权重 -> 应用 token 剪枝 -> 度量精度/速度收益」；
DynamicViT / EViT / ToMe 等方法均以这批骨干网络为对照基准，
保证加速对比中骨干与权重完全一致。
"""

import torch
import timm

# 项目内统一别名 -> timm 模型名。
# 统一入口使各剪枝方法可挂载到同一批骨干上，消融对比时只改变剪枝策略这一个变量。
TIMM_NAMES = {
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    # deit3: DeiT-III 权重，序列为纯 cls + patch（无 distillation token），
    # 架构与 deit 完全相同。CLS 保护逻辑只固定 index 0，dist token 会在
    # 合并/重排中丢失，导致 DeiT 的 cls+dist 双 token 分类头失配，
    # 因此剪枝类实验一律使用 deit3 权重。
    "deit3_small": "deit3_small_patch16_224",
    "deit3_base": "deit3_base_patch16_224",
}


def create_baseline(name: str, num_classes: int = 1000, pretrained: bool = True) -> torch.nn.Module:
    """按别名创建一个(预训练)DeiT/ViT模型。

    Args:
        name: TIMM_NAMES 中的别名，如 "deit_small"
        num_classes: 分类头类别数，ImageNet-1k 为 1000
        pretrained: 是否下载官方预训练权重
    Returns:
        nn.Module：输入 (B, 3, 224, 224)，输出 (B, num_classes)
    Side effects:
        调用 model.eval() 置于推理模式；剪枝加速评测均为前向，不需要训练态
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
