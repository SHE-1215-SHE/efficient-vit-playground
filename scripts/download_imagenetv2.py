"""下载并解压 ImageNet-V2 验证集（matched-frequency 版，约 1.2GB）。

输入: --out 指定的存放目录（默认 data）。
输出: <out>/imagenetv2-matched-frequency-format-val/<0..999>/，共 1000 个数字类目目录。
典型用法:
    python scripts/download_imagenetv2.py --out data

NOTE: 国内网络直连 huggingface.co 经常不可达，官方源请求抛出异常后自动回退到
hf-mirror.com 镜像重试，无需预先配置代理或镜像环境变量。
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

# ImageNet-V2 官方数据的 HuggingFace 托管地址；镜像源内容一致，
# 仅在官方域名网络不可达时作为回退使用
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

    # NOTE: 类目目录数 >= 1000 视为数据完整，跳过重复下载，保证脚本可幂等重跑
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
    tar_path.unlink()  # 校验通过后删除压缩包，释放磁盘空间
    print(f"完成 ✓  验证集路径: {extracted}")


if __name__ == "__main__":
    main()
