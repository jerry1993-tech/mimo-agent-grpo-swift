#!/usr/bin/env python3
"""从 prepare_grpo 产物（grpo_*.jsonl / grpo_train.jsonl）提取镜像清单，
供沙箱服务器预拉取（scripts/prefetch_images.sh）。

用法（GPU 实例，工程根目录）：
    python -m data.gen_image_list data/outputs/grpo_code.jsonl -o grpo_code.list
    # 多文件合并去重：
    python -m data.gen_image_list data/outputs/grpo_*.jsonl -o all.list
    # scp 到服务器后：
    #   bash scripts/prefetch_images.sh grpo_code.list
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description='grpo jsonl -> 镜像预拉取清单')
    ap.add_argument('inputs', nargs='+', help='grpo_*.jsonl（支持 glob）')
    ap.add_argument('-o', '--output', default='images.list', help='输出清单文件')
    args = ap.parse_args()

    images: list = []
    seen = set()
    n_rows = 0
    missing = 0
    for pattern in args.inputs:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cfg = json.loads(line).get('env_config', {})
                    except json.JSONDecodeError:
                        continue
                    n_rows += 1
                    img = cfg.get('dockerhub_image') or cfg.get('docker_image') or ''
                    if not img:            # music 域无沙箱
                        continue
                    if img not in seen:
                        seen.add(img)
                        images.append(img)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(images) + ('\n' if images else ''))
    print(f'[gen_image_list] rows={n_rows} unique_images={len(images)} -> {args.output}')
    if missing:
        print(f'[gen_image_list] {missing} rows had no image (music 域属正常)')


if __name__ == '__main__':
    sys.exit(main())
