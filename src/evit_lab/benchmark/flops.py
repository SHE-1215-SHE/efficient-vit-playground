"""FLOPs（浮点运算次数）与参数量统计，基于 fvcore。

NOTE: 动态 token 剪枝模型的 FLOPs 依赖输入内容（实际删除的 token 数随图片
变化），静态统计只能得到「不剪枝」或「指定剪枝率」下的近似值。严谨做法是
在真实图片分布上多次采样求均值；本模块先提供单次统计，保证各模型同口径。
"""

import torch
from fvcore.nn import FlopCountAnalysis


def count_flops(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224),
                device: str = "cpu") -> int:
    """统计单张图片前向的 FLOPs。

    Args:
        model: 任意 nn.Module（需处于 eval 模式）
        input_size: 完整输入 shape（含 batch 维），如 (1, 3, 224, 224)
        device: 统计时用的设备
    Returns:
        FLOPs 数值（乘加算一次），如 deit_small 约 4.6G
    """
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
        # timm 的 ViT 有部分算子未被 fvcore 覆盖，关闭告警避免日志冗长；
        # 缺失算子使总数略偏低，但所有模型同样偏低，横向对比仍然公平
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    return int(total)


def count_params(model: torch.nn.Module) -> int:
    """统计可训练参数总量（requires_grad=True）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
