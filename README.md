# Efficient-ViT Playground

动态 Token 剪枝 ViT 推理加速：在统一评测协议下对比 ToMe / EViT / MPM 三种匹配范式，并提出可跨匹配器复用的**熵引导调度**（training-free）。

## 核心结果

DeiT3-Small（patch16, 224×224, 22.06M），RTX 4090 / FP32 / batch=64，ImageNet-1k 官方 val 5 万张，全部数字见 `results_retest_20260924/`。

| 配置 | 最终 token | 吞吐 (img/s) | 加速比 | Top-1 | ΔTop-1 |
|---|---|---|---|---|---|
| Baseline | 197 | 2812.7 | 1.00× | 81.38 | — |
| ToMe r=4 | 149 | 2971.1 | 1.06× | 81.09 | −0.29 |
| ToMe r=8 | 101 | 3383.0 | 1.20× | 80.52 | −0.86 |
| ToMe r=12 | 53 | 3672.6 | 1.31× | 78.93 | −2.45 |
| **自适应 v2 r=12（本文方法）** | 53 | 3669.6 | 1.30× | **79.62** | **−1.76** |
| EViT k=52（免训练复现） | 53 | 4797.7 | 1.71× | 66.94 | −14.44 |
| MPM r=12（互最近邻） | 53 | 1204.6 | 0.43× | 74.64 | −6.74 |
| MPM + 熵调度（本文方法迁移） | 53 | 1511.4 | 0.54× | 76.47 | −4.91 |

## 核心发现

1. **1/4 token 预算只掉 2.45 点**（ToMe r=12）——ViT 背景区块高度冗余，且"合并"优于"丢弃"：同预算下 ToMe 比 EViT 高 12 点（信息保留 vs 丢弃浓缩的范式 tradeoff），但 EViT 快 1.7 倍（免去匹配合并开销）。
2. **熵引导调度（创新点①）**：第 3 层用 CLS 注意力熵做一次 front/back 调度模板选择，同预算下比固定 ToMe 高 **+0.69** 点（53 token 档）；消融 front(78.07) < uniform(78.93) < 熵自选(79.62) 证明决策本身有效；**迁移到 MPM 匹配器上仍 +1.83**——是跨匹配器可复用的调度机制，不是 ToMe 专属技巧。
3. **工程教训：数据依赖控制流是动态剪枝的通病**——逐层 `.item()` 决策（v1）触发 12 次 GPU 同步，比不剪枝还慢（2004 vs 2813 img/s）；重构为一次决策 + 模板查表（v2）后恢复。且"1 次同步"的声明本身曾被复查推翻（实际 3 次，详见 INTERVIEW.md 工程故事③），修复后自适应三档提速 7%~13%，r=12 档与固定 ToMe 速度持平、调度近乎零成本——同步开销被修复直接量化。
4. **匹配质量 ≠ 匹配开销**：MPM 的互最近邻配对更"精确"但召回低，同预算精度全输 ToMe 且越紧越输（−0.49/−1.91/−4.29）；其全量 N×N 亲和矩阵使吞吐仅为 ToMe 的 40%，全部低于不剪枝基线。无预算约束的 pure 模式更直接崩到 8.20%（诊断记录见 [results_retest_20260924/mpm_pure_diagnosis.md](results_retest_20260924/mpm_pure_diagnosis.md)，注意该行 `--tome-r 12` 参数无效，勿与 mpm_r12 的 74.64 混淆）。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/check_env.py                      # 环境自检

# 测速（--no-pretrained 只测结构吞吐）
python scripts/eval_speed.py --model deit3_small --device cuda --batch-size 64 --no-pretrained

# 精度（需 ImageNet-1k val, 数字类目目录结构 0000/...0999）
python scripts/eval_accuracy.py --model deit3_small --data data/imagenet_val

# ToMe 固定 / 熵自适应
python scripts/eval_speed.py --model deit3_small --device cuda --batch-size 64 \
    --tome-r 12 --tome-strength 0.7              # strength>0 启用熵调度
python scripts/eval_accuracy.py --model deit3_small --data data/imagenet_val \
    --tome-r 12 --tome-strength 0.7

# EViT / MPM
python scripts/eval_accuracy.py --model deit3_small --data data/imagenet_val --evit-k 52
python scripts/eval_accuracy.py --model deit3_small --data data/imagenet_val \
    --tome-r 12 --matcher mutual                 # budget-MPM; 加 --mpm-pure 为论文原味(batch=1)

# 一键复现全部实验
bash scripts/run_retest_main.sh                  # 主对比 13 组
bash scripts/run_mpm_exp.sh                      # MPM 对比
python scripts/plot_pareto.py                    # 帕累托双面板图
```

正确性自检：`pytest tests/`（ToMe 合并数/等价性、EViT 序列稳定性、MPM 预算守恒等 14 项）。

## 目录结构

```
src/evit_lab/
├── models/baseline.py        # timm 官方预训练权重加载
├── models/tome.py            # ToMe 二分匹配 + 熵引导调度(创新点)
├── models/evit.py            # EViT 免训练复现(CLS 注意力 top-K + 浓缩)
├── models/tokenlearner.py    # TokenLearner 可学习浓缩 PoC
├── models/mutual_pair.py     # MPM 互最近邻匹配(budget/pure 双模式)
└── benchmark/                # fvcore FLOPs + CUDA Event 吞吐(中位数抗抖动)
scripts/                      # 评测入口 + 一键复现 + 数据准备
tests/                        # 正确性自检
results/                      # 汇总 Excel(experiment_summary.xlsx) + 帕累托图 + 原始 CSV
results_retest_20260924/      # 2026-09-24 全量重测(13 组复现 + MPM; 09-25 同步修复后自适应三档重测)
```

## 方法说明

- **ToMe**（training-free）：每层 attention 后、MLP 前，用 K 的余弦相似度做双向二分匹配，合并 r 对 token；CLS 永不合并；12 层共删 12r 个。
- **熵引导调度**（本文）：前两层固定 r，第 3 层用 CLS 注意力熵判别"图是否已聚焦"，熵低选 front 模板（浅层多删）、熵高选 back（深层多删），模板系数和恒为 10 保证总预算守恒；每张图仅 1 次 GPU→CPU 同步。
- **EViT**（免训练复现）：第 4 层按 CLS 注意力保留 top-K，其余加权浓缩成 1 个 token。
- **MPM**（arXiv 2604.05718 对比实现）：互为最近邻才配对合并；budget 模式按相似度取 top-r 保证同预算可比。

