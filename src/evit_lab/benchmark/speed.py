"""吞吐量（images/s）与延迟（ms/img）测量。

测量规范：
1. 先 warmup 若干次，排除显存分配、cudnn 自动调优等一次性开销
2. GPU 用 cuda Event 计时（时钟分辨率高于 CPU 计时器），CPU 回退到 perf_counter
3. 每次迭代后 synchronize，保证异步 kernel 真正完成再进入下次计时
4. 报告多组迭代的中位数，抑制共享显卡被抢占导致的抖动
"""

import time
from statistics import median

import torch


@torch.no_grad()
def measure_speed(model: torch.nn.Module, input_size: tuple = (3, 224, 224),
                  batch_size: int = 64, device: str = "cuda",
                  warmup: int = 20, iters: int = 50) -> dict:
    """测量模型在指定 batch 下的延迟与吞吐。

    Returns:
        dict: {
            "latency_ms":      单张图片平均延迟（毫秒）
            "throughput":      每秒处理的图片数
            "iter_times_ms":   每次迭代（一个batch）耗时列表，供分析抖动
        }

    NOTE: latency 与 throughput 均由同一中位迭代耗时导出，二者互为倒数换算，
    不携带独立信息；动态剪枝模型的耗时随输入内容变化，跨方法对比需同数据集。
    """
    model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, *input_size, device=device)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    def one_iter():
        if use_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(dummy)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)  # 毫秒
        t0 = time.perf_counter()
        model(dummy)
        return (time.perf_counter() - t0) * 1000.0

    # warmup 前整体同步一次，防止此前排队的 kernel 混入首批计时
    if use_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        one_iter()

    iter_times = [one_iter() for _ in range(iters)]
    med = median(iter_times)
    return {
        "latency_ms": med / batch_size,
        "throughput": batch_size * 1000.0 / med,
        "iter_times_ms": iter_times,
    }
