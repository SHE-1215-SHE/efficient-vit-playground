"""统一评测模块：FLOPs 统计 + 吞吐量/延迟测量。

为什么重要：整个项目的核心卖点是「加速」，所有模型必须在同一把尺子下
测量，面试时数字才站得住脚。这把尺子就是这里。
"""

from .flops import count_flops, count_params
from .speed import measure_speed

__all__ = ["count_flops", "count_params", "measure_speed"]
