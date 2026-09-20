"""下载 ImageNet-V2 验证集（matched-frequency 版，约1.2GB）。

国内网络如果 huggingface.co 连不上，脚本会自动切换到 hf-mirror.com 镜像。

用法:
    python scripts/download_imagenetv2.py --out data
下载并解压后得到 data/imagenetv2-matched-frequency-format-val/<0..999>/
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

# 官方 ImageNet-V2 的 HuggingFace 托管地址
HF_URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
MIRROR_URL = "https://hf-mirror.com/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"


def download(url: str, dst: Path):
    print(f"下载中: {url}")
    print("（约1.2GB，请耐心等待）")

    def hook(blocks, block_size, total):
        if total > 0:
            done_mb = blocks * block_size / 1024 / 1024
            total_mb = total / 1024 / 1024
            if blocks % 200 == 0:
                print(f"\r  {done_mb:.0f}/{total_mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dst, reporthook=hook)
    print("\n下载完成")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="存放目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = out_dir / "imagenetv2-matched-frequency-format-val"

    if extracted.exists() and len(list(extracted.iterdir())) >= 1000:
        print(f"已存在且完整，跳过下载: {extracted}")
        return

    tar_path = out_dir / "imagenetv2-matched-frequency.tar.gz"

    try:
        download(HF_URL, tar_path)
    except Exception as e:
        print(f"官方源失败({e})，切换国内镜像重试...")
        download(MIRROR_URL, tar_path)

    print("解压中...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)

    n_dirs = len([d for d in extracted.iterdir() if d.is_dir()])
    assert n_dirs == 1000, f"解压后类别目录数应为1000，实际 {n_dirs}"
    tar_path.unlink()  # 删除压缩包省磁盘
    print(f"完成 ✓  验证集路径: {extracted}")


if __name__ == "__main__":
    main()
