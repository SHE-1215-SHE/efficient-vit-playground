"""从官方 val tar 包整理出数字类目结构的 ImageNet 验证集（服务器一次性准备步骤）。

输入: ILSVRC2012_img_val.tar（50000 张平铺 JPEG，无类别目录）与官方 ground truth
      标签文件 assets/imagenet_2012_validation_synset_labels.txt
      （每行依次对应 ILSVRC2012_val_00000001.JPEG 起递增编号的图片）。
输出: <out>/<0000..0999>/ 数字类目目录，eval_accuracy.py 可直接使用，无需改动。
典型用法（服务器）:
    python scripts/prepare_imagenet_val.py --tar /newdisk/data/ImageNet/ILSVRC2012_img_val.tar --out data/imagenet_val

NOTE: 类目映射——标准 ImageNet 类索引 = wnid 排序序号（torchvision/timm 约定），
wnid 全集恰好可从标签文件取得（每类在 val 中都有 50 张）。
正确性自检——整理完成后用 deit3_small 评测，Top-1 应约 81~83%；
若接近随机（0.1%）则说明映射有误。
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

LABELS_FILE = Path(__file__).resolve().parents[1] / "assets" / "imagenet_2012_validation_synset_labels.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", required=True, help="ILSVRC2012_img_val.tar 路径")
    parser.add_argument("--out", default="data/imagenet_val", help="输出目录（数字类目结构）")
    args = parser.parse_args()

    labels = LABELS_FILE.read_text().split()
    assert len(labels) == 50000, f"标签行数异常: {len(labels)}"
    wnids = sorted(set(labels))
    assert len(wnids) == 1000, f"类别数异常: {len(wnids)}"
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    print(f"标签校验通过: 50000 行 / 1000 类")

    out = Path(args.out)
    for i in range(1000):
        (out / f"{i:04d}").mkdir(parents=True, exist_ok=True)

    count = 0
    with tarfile.open(args.tar) as tar:
        for member in tar:  # NOTE: 流式遍历 tar 边解包边归类，不落地临时目录，避免磁盘同时保存两份数据
            if not member.isfile():
                continue
            name = Path(member.name).name  # ILSVRC2012_val_XXXXXXXX.JPEG
            img_no = int(name.split("_")[-1].split(".")[0])  # 1-indexed 编号，与标签文件行号对齐
            idx = wnid_to_idx[labels[img_no - 1]]
            with tar.extractfile(member) as src, open(out / f"{idx:04d}" / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
            if count % 10000 == 0:
                print(f"  已整理 {count}/50000")
    print(f"完成: {count} 张图片 -> {out}")

    # 自检: 标准 val 划分下每类应恰好 50 张，数量不符说明归类出错
    bad = [d.name for d in out.iterdir() if len(list(d.iterdir())) != 50]
    if bad:
        print(f"! 以下类目数量不是50, 请检查: {bad[:10]}")
    else:
        print("自检通过: 每类恰好 50 张")


if __name__ == "__main__":
    main()
