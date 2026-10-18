#!/usr/bin/env python3
"""下载 MiMo-V2.6-RL-oss 的 5 个 parquet「任务环境池」。

数据集（Apache-2.0）：https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss
本脚本只负责把对应 parquet 拉到本地（默认 data/raw/），并打印行数与 schema。
已存在的文件直接复用，不重复下载；huggingface_hub 会自动复用全局 HF cache。

用法（工程根目录运行）：
    python3 data/download_dataset.py
    python3 data/download_dataset.py --domains music code --output-dir data/raw
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


def main() -> None:
    ap = argparse.ArgumentParser(description='下载 MiMo-V2.6-RL-oss 任务环境池 parquet')
    ap.add_argument('--domains', nargs='*', default=list(DOMAIN_PARQUETS),
                    choices=list(DOMAIN_PARQUETS),
                    help='要下载的领域，默认全部 5 个')
    ap.add_argument('--output-dir', default='data/raw',
                    help='parquet 落地目录（默认 data/raw/）')
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0
    for domain in args.domains:
        rel = DOMAIN_PARQUETS[domain]
        local_path = os.path.join(args.output_dir, rel)
        if os.path.exists(local_path):
            print(f'[reuse] {domain}: {local_path}')
        else:
            # local_dir 保留相对路径结构：general/train.parquet 落到 <out>/general/train.parquet
            local_path = hf_hub_download(
                repo_id=REPO_ID, filename=rel, repo_type='dataset',
                local_dir=args.output_dir,
            )
            print(f'[download] {domain}: {local_path}')

        tbl = pq.read_table(local_path)
        n = tbl.num_rows
        total += n
        print(f'  domain={domain} rows={n}')
        print(f'  schema:\n{tbl.schema}')
    print(f'[done] total rows={total}, output_dir={args.output_dir}')


if __name__ == '__main__':
    main()
