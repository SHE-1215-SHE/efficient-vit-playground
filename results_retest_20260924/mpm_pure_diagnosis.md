# pure_mpm_batch1 诊断记录

对应 `mpm_accuracy.csv` 末行 `pure_mpm_batch1,8.20,20.10,variable`。
该行是 **MPM 论文原始模式（pure, 无 budget 约束）的失败上界记录**，诊断性质，
不参与任何正式对比表。

## 产生命令（scripts/run_mpm_exp.sh L52-57，脚本自动 append 至 CSV）

```bash
python scripts/eval_accuracy.py --model deit3_small --data data/imagenet_val \
    --batch-size 1 --tome-r 12 --matcher mutual --mpm-pure
```

- 数据：官方 val 全量 5 万张
- batch=1：pure 模式每张图输出 token 数不同，凑不成 (B,N,C) 批
  （src/evit_lab/models/mutual_pair.py L44-45 对 `not budget and B > 1` 直接 raise）
- `--tome-r 12` 在 pure 模式下**无效**：`budget=False` 路径不使用 r；
  引用本行时切勿与 mpm_r12 的 74.64 混淆为同一档位
- `final_tokens=variable`：每张图剩余 token 数不同，无单一数值可写

## 崩溃至 8.20% 的三级机制（mutual_pair.py L60-73 pure 路径）

1. **数量塌缩**：真实数据每层互配对 18~78 个，每层删几十个 →
   197 → ~150 → ~110 几何式衰减，6~8 层内触及 `min_tokens=8` 地板（L21/L42），
   序列仅剩 8~12 个 token。
2. **质量无底线**：budget 模式按相似度取 top-r；pure 模式来者不拒——
   "互为最近邻"只要求互相同意，不要求相似度高，余弦 0.3 的低质 token
   互为最近邻照样合并。
3. **分布外输入**：免训练协议下，预训练模型后几层从未见过
   "约 10 个 token 且全是反复平均产物"的序列，CLS 输出接近噪声
   → Top-1 8.20% / Top-5 20.10%（随机水平 0.1%，残余信号说明
   CLS 尚存统计偏好，但特征已基本报废）。

## 结论与局限

- 佐证 MPM 原论文为何仅在分割任务 + 精选插入层的设定下使用 pure 模式；
  budget 模式（按相似度截断）才是 MPM 用于分类的正确姿势。
- 局限：pure 运行未逐层记录每张图实际剩余 token 数，
  `variable` 的具体分布（如中位数）未留档；坐实需在 pure 路径
  加 token 计数日志重跑。
