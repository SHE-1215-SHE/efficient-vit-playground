# Efficient-ViT Playground

动态 Token 剪枝 ViT 推理加速：从算法对比到 C++ 部署，再到多模态大模型落地。

## 项目简介

视觉 Transformer（ViT）中大量 token 是冗余的背景块，剪掉它们可以近乎无损地大幅降低计算量。本项目在**统一评测协议**下复现和对比主流动态 Token 剪枝方法（DynamicViT / EViT / ToMe），并延伸到工程化部署（ONNX + C++）与多模态应用（剪枝 ViT 加速 VLM prefill）。

## 已复现结果

所有测速在 **RTX 4090, FP32, batch=64** 下完成；精度在 **ImageNet-V2 (matched-frequency)** 10k 样本上评测。

| 模型 | 参数量 | FLOPs | 吞吐量 (img/s) |
|---|---|---|---|
| DeiT-Tiny | 5.72M | 1.08G | 6558.9 |
| DeiT-Small | 22.05M | 4.25G | 2826.3 |
| DeiT-Base | 86.57M | 16.87G | 953.6 |

基线精度：DeiT-Small 在 ImageNet-V2 上 **Top-1 68.52% / Top-5 88.16%**。

（剪枝方法的对比数据陆续补充中）

## 快速开始

```bash
pip install -r requirements.txt

# 环境自检
python scripts/check_env.py

# 测速（--no-pretrained 跳过权重下载，测速只与结构有关）
python scripts/eval_speed.py --model deit_small --device cuda --batch-size 64 --no-pretrained

# 精度评测（需先下载 ImageNet-V2）
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像
python scripts/download_imagenetv2.py --out data
python scripts/eval_accuracy.py --model deit_small --data data/imagenetv2-matched-frequency-format-val
```

## 目录结构

```
src/evit_lab/
├── models/       # 模型工厂：基线 + 剪枝方法统一入口
└── benchmark/    # 统一评测：FLOPs(fvcore) / 延迟+吞吐(CUDA Event)
scripts/          # 评测与数据准备脚本
results/          # 原始实验数据（CSV）
```

## 路线图

- [x] 阶段①：基线复现与统一评测框架
- [ ] 阶段②：DynamicViT / EViT / ToMe 统一集成，精度-速度帕累托曲线
- [ ] 阶段③：ONNX 导出 + C++ 推理 + INT8 量化
- [ ] 阶段④：剪枝 ViT 加速轻量 VLM 的 prefill
