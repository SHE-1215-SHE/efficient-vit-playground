"""FLOPs（浮点运算次数）与参数量统计，基于 fvcore。

注意坑点：动态 token 剪枝模型的 FLOPs 随输入图片内容变化（剪掉的token数
不同），静态统计只能得到「不剪枝」或「指定剪枝率」的近似值。之后做剪枝
方法时，需要用「真实图片分布上采样多次求平均」的方式，这里先支持单次统计。
"""

import torch
from fvcore.nn import FlopCountAnalysis


def count_flops(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224),
                device: str = "cpu") -> int:
    """统计单张图片前向的 FLOPs。

    Args:
        model: 任意 nn.Module（需处于 eval 模式）
        input_size: 输入张量 shape（不含 batch 维度时请自行调整）
        device: 统计时用的设备
    Returns:
        FLOPs 数值（乘加算一次），如 deit_small 约 4.6G
    """
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
        # timm 的 ViT 有一些 fvcore 不认识的算子，关掉警告避免刷屏，
        # 缺失算子会让总数略偏低，但所有模型同样偏低，对比仍然公平
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    return int(total)


def count_params(model: torch.nn.Module) -> int:
    """统计可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
