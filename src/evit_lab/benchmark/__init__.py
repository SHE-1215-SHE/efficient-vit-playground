"""统一评测模块：FLOPs 统计 + 吞吐量/延迟测量。

项目核心卖点是「加速」，所有模型（基线与剪枝方法）必须在同一口径下测量，
精度/速度对比的结论才成立；本模块即该统一口径的实现。
"""

from .flops import count_flops, count_params
from .speed import measure_speed

__all__ = ["count_flops", "count_params", "measure_speed"]
