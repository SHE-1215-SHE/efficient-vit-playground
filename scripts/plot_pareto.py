"""绘制帕累托曲线 v2：双面板展示各剪枝方法的精度-效率权衡（y=Top-1）。

输入: 结果目录（默认 results_retest_20260924/，2026-09-24 同日背靠背重测、
      13 组全部复现）下的 accuracy.csv / speed.csv / mpm_accuracy.csv / mpm_speed.csv。
输出: 双面板 PNG（默认 results/pareto_v2.png）。面板A x=最终 token 数，衡量信息
      保留效率；面板B x=实测吞吐，衡量真实加速收益——EViT 的加速收益仅在面板B可见。
典型用法:
    python scripts/plot_pareto.py [--dir results_retest_20260924] [--out results/pareto_v2.png]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 服务器/CI 环境通常缺少中文字体，中文标签会渲染为方块，图内文字统一使用英文
STYLE = {
    "baseline": dict(color="gray", marker="*", s=300, label="Baseline (DeiT3-S)"),
    "tome": dict(color="tab:blue", marker="o", s=70, label="ToMe (fixed r)"),
    "adaptive": dict(color="tab:red", marker="s", s=70, label="ToMe + entropy-adaptive (ours)"),
    "evit": dict(color="tab:green", marker="^", s=70, label="EViT (training-free repro)"),
    "mpm": dict(color="tab:purple", marker="D", s=60, label="MPM mutual-NN (budget)"),
    "adaptive_mpm": dict(color="violet", marker="P", s=80, label="MPM + entropy-adaptive (ours)"),
}


def group_of(name: str) -> str:
    if name.startswith("tome_r"):
        return "tome"
    if name.startswith("adaptive_mpm"):
        return "adaptive_mpm"
    if name.startswith("adaptive"):
        return "adaptive"
    if name.startswith("evit"):
        return "evit"
    if name.startswith("mpm_r"):
        return "mpm"
    return "baseline"


def load(retest_dir: Path) -> dict:
    acc = {r["name"]: r for r in csv.DictReader(open(retest_dir / "accuracy.csv", encoding="utf-8"))}
    spd = {r["name"]: r for r in csv.DictReader(open(retest_dir / "speed.csv", encoding="utf-8"))}
    macc = {r["name"]: r for r in csv.DictReader(open(retest_dir / "mpm_accuracy.csv", encoding="utf-8"))}
    mspd = {r["name"]: r for r in csv.DictReader(open(retest_dir / "mpm_speed.csv", encoding="utf-8"))}
    merged = {}
    for name, r in {**acc, **macc}.items():
        s = spd.get(name) or mspd.get(name)
        if s is None:
            continue  # 纯 MPM 等未参与测速的行缺少吞吐数据，跳过
        merged[name] = dict(
            method=group_of(name),
            tokens=float(r.get("final_tokens_measured") or r["final_tokens"]),
            top1=float(r["top1"]),
            thr=float(s["throughput_img_s"]),
        )
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_retest_20260924")
    ap.add_argument("--out", default="results/pareto_v2.png")
    args = ap.parse_args()

    pts = load(Path(args.dir))
    grouped = defaultdict(list)
    for p in pts.values():
        grouped[p["method"]].append(p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for ax, xkey, xlabel in (
        (ax1, "tokens", "Final tokens kept (lower = more pruning)"),
        (ax2, "thr", "Throughput (img/s, RTX 4090 fp32, higher = faster)"),
    ):
        for method, ps in grouped.items():
            st = STYLE[method]
            xs = [p[xkey] for p in ps]
            ys = [p["top1"] for p in ps]
            ax.scatter(xs, ys, **st, zorder=3, edgecolors="white", linewidths=0.5)
            if method != "baseline":
                ps2 = sorted(zip(xs, ys))
                ax.plot([x for x, _ in ps2], [y for _, y in ps2],
                        color=st["color"], alpha=0.4, lw=1.5, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ImageNet val Top-1 (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(64, 83)  # y 轴下界放宽至 64，避免 EViT k=52 (66.94) 被裁剪出画

    # 关键结论标注: 面板A标出自适应 r=4 的调度开销位置，面板B标出 EViT 的
    # 速度上限与 MPM 匹配器开销
    a4 = pts.get("adaptive_r4")
    if a4:
        ax1.annotate("adaptive r=4: 0.97x\n(scheduling overhead > saving)",
                     (a4["tokens"], a4["top1"]), textcoords="offset points",
                     xytext=(6, -26), fontsize=7.5, color="tab:red")
    e52 = pts.get("evit_k52")
    if e52:
        ax2.annotate("EViT k=52: 1.71x\nbut -14.4 pts",
                     (e52["thr"], e52["top1"]), textcoords="offset points",
                     xytext=(-88, -2), fontsize=7.5, color="tab:green")
    m12 = pts.get("mpm_r12")
    if m12:
        ax2.annotate("MPM matcher overhead:\nall below baseline",
                     (m12["thr"], m12["top1"]), textcoords="offset points",
                     xytext=(8, -6), fontsize=7.5, color="tab:purple")

    fig.suptitle("Token pruning Pareto curves (DeiT3-Small 384, 197 tokens, retested 2026-09-24)", y=0.99)
    handles = [plt.scatter([], [], **{k: v for k, v in st.items() if k != "s"}, s=st["s"])
               for st in STYLE.values()]
    fig.legend(handles, [st["label"] for st in STYLE.values()],
               loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.085))
    fig.text(0.5, 0.015,
             "TokenLearner (25x condensation, 100-class PoC) and pure-MPM (collapses to 8.2%) "
             "use different protocols, excluded.",
             ha="center", fontsize=7.5, color="gray")
    fig.tight_layout(rect=(0, 0.17, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
