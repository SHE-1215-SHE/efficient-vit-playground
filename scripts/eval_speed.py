"""测速脚本：统计 FLOPs / 参数量 / 延迟 / 吞吐量。

用法（本机 CPU 冒烟）:
    python scripts/eval_speed.py --model deit_small --device cpu --batch-size 8 --iters 5
用法（服务器 4090 正式测）:
    python scripts/eval_speed.py --model deit_small --device cuda --batch-size 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from evit_lab.benchmark import count_flops, count_params, measure_speed
from evit_lab.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit_small", help="模型别名，见 models/baseline.py")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true", help="测速不需要权重，跳过下载")
    parser.add_argument("--tome-r", type=int, default=0,
                        help=">0 时启用 ToMe（每层合并掉 r 个 token），如 --tome-r 8")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("! 当前机器无 GPU，自动回退到 cpu（正式数据请在服务器上测）")
        args.device = "cpu"

    print(f"加载模型: {args.model} (pretrained={not args.no_pretrained}, tome_r={args.tome_r})")
    model = build_model(args.model, pretrained=not args.no_pretrained, tome_r=args.tome_r)

    flops = count_flops(model, (1, 3, args.img_size, args.img_size))
    params = count_params(model)
    print(f"参数量 : {params / 1e6:.2f} M")
    if args.tome_r > 0:
        # ToMe 逐层合并导致序列长度动态变化，fvcore 静态追踪算不出真实FLOPs，
        # 这里只打印名义值；加速能力请以实测吞吐为准
        n_layers = len(model.blocks)
        print(f"FLOPs  : {flops / 1e9:.2f} G (名义值, 未反映动态合并)")
        print(f"ToMe   : 每层合并 {args.tome_r} 个 token, 共 {n_layers} 层, "
              f"197 -> {197 - n_layers * args.tome_r} 个")
    else:
        print(f"FLOPs  : {flops / 1e9:.2f} G")

    speed = measure_speed(
        model,
        input_size=(3, args.img_size, args.img_size),
        batch_size=args.batch_size,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(f"设备   : {args.device}  batch={args.batch_size}")
    print(f"延迟   : {speed['latency_ms']:.2f} ms/img")
    print(f"吞吐量 : {speed['throughput']:.1f} img/s")


if __name__ == "__main__":
    main()
