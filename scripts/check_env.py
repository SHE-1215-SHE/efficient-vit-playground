"""环境自检脚本：依次验证 Python 版本、核心依赖、GPU 状态与模型前向链路。

输入: 无命令行参数；检查对象为当前 Python 环境与 CUDA 设备。
输出: 终端打印四项自检结果（Python 版本 / 依赖版本 / GPU 显存占用 / 前向冒烟测试）。
典型用法:
    python scripts/check_env.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    print("=" * 50)
    print("[1/4] Python 版本")
    import platform
    print("  ", platform.python_version())

    print("[2/4] 核心依赖版本")
    import torch
    import torchvision
    import timm
    print(f"   torch      {torch.__version__}")
    print(f"   torchvision{torchvision.__version__}")
    print(f"   timm       {timm.__version__}")

    print("[3/4] GPU 状态")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"   GPU{i}: {props.name}  显存 {free / 1024**3:.1f}/{total / 1024**3:.1f} GB 空闲")
    else:
        print("    ! 无可用 GPU（本机调试属正常，服务器上必须能看到 4090）")

    # NOTE: 冒烟测试不依赖 GPU，CPU 即可完成模型构建与前向链路验证；输出维度
    # 断言 (2, 1000) 用于同时确认分类头结构与 ImageNet 类别数对齐。
    print("[4/4] 模型前向冒烟测试（CPU 即可）")
    from evit_lab.models import build_model
    model = build_model("deit_tiny", pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1000), f"输出维度异常: {out.shape}"
    print("    deit_tiny 前向通过，输出 shape = (2, 1000)")

    print("=" * 50)
    print("环境自检全部通过 ✓")


if __name__ == "__main__":
    main()
