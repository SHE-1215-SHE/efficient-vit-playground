"""注意力热力图可视化：看 CLS 到底在看图里哪里。

两种视角:
    1. last-layer: 最后一层 CLS 对各 patch 的注意力（最直接）
    2. rollout:    Attention Rollout (Abnar & Zuidema 2020)，逐层连乘
       (A+I)/2，考虑了残差连接的信息流通，通常比单层更接近真实归因

实现方式: 在每个 block 的 qkv Linear 上挂 forward hook，从输出手工重算
注意力矩阵（不侵入模型 forward，基线/剪枝模型通用）。
注意: 热力图与空间位置一一对应，只在**基线模型**上做（ToMe/EViT 会打乱
patch 位置，网格对应关系失效——这本身也是个值得讲的点）。

用法:
    python scripts/visualize_attention.py --image my.jpg
    python scripts/visualize_attention.py                # 无图则生成合成测试图
输出: results/attention_<model>.png (原图 | rollout | 最后一层 三联图)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from evit_lab.models import build_model

GRID = 14  # 224/16 patch 网格


def _synthetic_image() -> Image.Image:
    """生成合成测试图: 红色方块(高对比前景) + 噪声背景, 用于开箱验证。"""
    rng = np.random.default_rng(0)
    arr = rng.integers(60, 100, (224, 224, 3), dtype=np.uint8)
    arr[70:150, 80:160] = [220, 40, 40]  # 前景红块
    return Image.fromarray(arr)


@torch.no_grad()
def collect_attention(model, x):
    """在每个 block 的 qkv 上挂 hook 收集该层平均注意力矩阵, 返回 list[(N,N)]。"""
    attn_mats = []
    hooks = [blk.attn.qkv.register_forward_hook(_grab_qkv(blk.attn, attn_mats))
             for blk in model.blocks]
    model(x)
    for h in hooks:
        h.remove()
    return attn_mats


def _grab_qkv(attn, store):
    """返回 hook: 从 qkv 输出重算该层平均注意力矩阵 (1,N,N)。"""
    def hook(module, inp, out):
        qkv = out                                   # (B, N, 3C)
        B, N, C3 = qkv.shape
        C = C3 // 3
        H, d = attn.num_heads, C // attn.num_heads
        q, k, _ = qkv.reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4).unbind(0)
        p = (q @ k.transpose(-1, -2) * attn.scale).softmax(-1)  # (B,H,N,N)
        store.append(p.mean(1)[0].cpu())            # 头平均, 取 batch 0
    return hook


def rollout(attn_mats):
    """Attention Rollout: 逐层 A <- (A+I)/2 连乘, 取 CLS 行。"""
    result = None
    for a in attn_mats:
        n = a.shape[0]
        a = (a + torch.eye(n)) / 2
        result = a if result is None else a @ result
    return result[0]  # CLS 行, (N,)


def to_heatmap(cls_attn, size=224):
    """CLS 行 (197,) -> 去掉cls -> 14x14 -> 上采样 -> [0,1] 热力图数组。"""
    m = cls_attn[1:].reshape(GRID, GRID)
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    m = F.interpolate(m[None, None], size=(size, size), mode="bilinear",
                      align_corners=False)[0, 0].numpy()
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit3_small")
    parser.add_argument("--image", default="", help="图片路径; 留空用合成测试图")
    parser.add_argument("--out", default="", help="输出路径, 默认 results/attention_<model>.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(args.image).convert("RGB") if args.image else _synthetic_image()
    model = build_model(args.model, pretrained=True)
    model.eval()

    from timm.data import create_transform, resolve_model_data_config
    tf = create_transform(**resolve_model_data_config(model), is_training=False)
    x = tf(img).unsqueeze(0)

    attn_mats = collect_attention(model, x)
    last = attn_mats[-1][0]        # 最后一层 CLS 行
    roll = rollout(attn_mats)      # rollout CLS 行

    out_path = args.out or f"results/attention_{args.model}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    resized = img.resize((224, 224))
    for ax, (title, hm) in zip(axes, [
            ("Input", None),
            ("Attention Rollout (CLS)", to_heatmap(roll)),
            ("Last Layer (CLS)", to_heatmap(last))]):
        ax.imshow(resized)
        if hm is not None:
            ax.imshow(hm, cmap="jet", alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"已保存: {out_path}")

    # 数值 sanity: 合成图中红块位于 patch 网格 [4:9, 5:10], 检查注意力对比度
    patch_attn = roll[1:].reshape(GRID, GRID)
    fg = patch_attn[4:9, 5:10].mean().item()
    bg = patch_attn.median().item()
    print(f"块内/背景注意力对比度: {fg / bg:.2f}x (合成红块图应>1; 真实自然图片通常更尖锐)")


if __name__ == "__main__":
    main()
