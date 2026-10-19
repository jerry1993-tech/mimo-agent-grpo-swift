#!/usr/bin/env python3
"""下载 MiMo-V2.6-RL-oss 的 5 个 parquet「任务环境池」。

数据集（Apache-2.0）：https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss
（ModelScope 同名镜像亦有官方发布，国内下载更快）
本脚本只负责把对应 parquet 拉到本地（默认 data/raw/），并打印行数与 schema。
已存在的文件直接复用，不重复下载。

下载通道（--channel）：
- modelscope（默认，国内快）：经 ModelScope 官方镜像；需 pip install modelscope
- hf：huggingface_hub（尊重 HF_ENDPOINT=https://hf-mirror.com）

用法（工程根目录运行）：
    python3 data/download_dataset.py
    python3 data/download_dataset.py --domains music code --output-dir data/raw
    python3 data/download_dataset.py --channel hf   # HF 镜像通道
"""
from __future__ import annotations

import os
import sys
import argparse

# 项目根 sys.path 兜底
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ID = "XiaomiMiMo/MiMo-V2.6-RL-oss"

# domain -> 仓库内 parquet 相对路径
DOMAIN_PARQUETS = {
    'code': 'code.parquet',
    'webdev': 'webdev.parquet',
    'cyber': 'cyber.parquet',
    'music': 'music.parquet',
    'general': 'general/train.parquet',
}


def _ms_download(rel: str, output_dir: str) -> str:
    """ModelScope 数据集文件直链下载（国内快；官方同步仓）。"""
    import urllib.request
    url = (f'https://modelscope.cn/api/v1/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/'
           f'repo?Revision=master&FilePath={rel}')
    dest = os.path.join(output_dir, rel)
    os.makedirs(os.path.dirname(dest) or output_dir, exist_ok=True)
    tmp = dest + '.tmp'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def _download(rel: str, output_dir: str, channel: str) -> str:
    """按通道下载单个文件，返回本地路径。"""
    if channel == 'modelscope':
        return _ms_download(rel, output_dir)
    # hf 通道
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=REPO_ID, filename=rel, repo_type='dataset',
        local_dir=output_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description='下载 MiMo-V2.6-RL-oss 任务环境池 parquet')
    ap.add_argument('--domains', nargs='*', default=list(DOMAIN_PARQUETS),
                    choices=list(DOMAIN_PARQUETS),
                    help='要下载的领域，默认全部 5 个')
    ap.add_argument('--output-dir', default='data/raw',
                    help='parquet 落地目录（默认 data/raw/）')
    ap.add_argument('--channel', default='modelscope', choices=['modelscope', 'hf'],
                    help='下载通道：modelscope（默认，国内快）| hf（HF/HF-mirror）')
    args = ap.parse_args()

    import pyarrow.parquet as pq

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0
    for domain in args.domains:
        rel = DOMAIN_PARQUETS[domain]
        local_path = os.path.join(args.output_dir, rel)
        if os.path.exists(local_path):
            print(f'[reuse] {domain}: {local_path}')
        else:
            try:
                local_path = _download(rel, args.output_dir, args.channel)
                print(f'[download:{args.channel}] {domain}: {local_path}')
            except Exception as e:
                print(f'[fallback] {domain}: {args.channel} 通道失败'
                      f'（{type(e).__name__}: {str(e)[:120]}），改走 hf')
                from huggingface_hub import hf_hub_download
                local_path = hf_hub_download(
                    repo_id=REPO_ID, filename=rel, repo_type='dataset',
                    local_dir=args.output_dir)

        tbl = pq.read_table(local_path)
        n = tbl.num_rows
        total += n
        print(f'  domain={domain} rows={n}')
    print(f'[done] total rows={total}, output_dir={args.output_dir}')


if __name__ == '__main__':
    main()
