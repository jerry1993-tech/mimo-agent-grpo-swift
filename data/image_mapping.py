#!/usr/bin/env python3
"""镜像名映射：parquet 里的 dataset_image -> Docker Hub 的 xiaomimimo/mimo-v2.6-rl-oss:<tag>。

事实（已核实 data/raw/image-mapping.jsonl，3764 行）：
- parquet 的 extra_info.instance_json.docker_image 是本地别名，例如
  ``format-code-task-001457:latest`` / ``arvo-rl:v1-arvo-35858`` /
  ``webdev-rl-opensource:v2`` / ``general-agent-env-0:oss``；
- 真实镜像发布在 Docker Hub 单仓库多 tag：
  ``docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457`` 等（tag 名与别名不同）；
- 映射表是数据集仓的 ``image-mapping.jsonl``（每行 {dataset_image, dockerhub_image}）。

本模块职责：
- ensure_image_mapping(raw_dir)：本地无则下载 image-mapping.jsonl（走 HF_ENDPOINT 镜像）；
- load_image_mapping(raw_dir) -> {dataset_image: dockerhub_image}；
- resolve_dockerhub_image(dataset_image, mapping) -> dockerhub_image 或 None（未收录）。

未收录的镜像名（如自定义/local 后端测试名）原样保留，由 DockerBackend 兜底
（见 sandbox/docker_backend.py 的 _resolve_image）。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_MAPPING_FILE = 'image-mapping.jsonl'


def mapping_path(raw_dir: str) -> str:
    return os.path.join(raw_dir, IMAGE_MAPPING_FILE)


def _ms_download(rel: str, local_dir: str) -> str:
    """ModelScope 数据集文件直链下载（国内快；repo 存在官方同步）。"""
    import urllib.request
    url = (f'https://modelscope.cn/api/v1/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/'
           f'repo?Revision=master&FilePath={rel}')
    dest = os.path.join(local_dir, os.path.basename(rel))
    os.makedirs(local_dir, exist_ok=True)
    tmp = dest + '.tmp'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def ensure_image_mapping(raw_dir: str = 'data/raw') -> str:
    """确保 image-mapping.jsonl 在本地（已存在直接复用；缺失则下载）。

    下载通道：MS_CHANNEL=modelscope（默认，国内快）| hf（HF_ENDPOINT 镜像）。
    """
    path = mapping_path(raw_dir)
    if os.path.exists(path):
        return path
    channel = os.environ.get('MS_CHANNEL', 'modelscope')
    if channel == 'modelscope':
        try:
            return _ms_download(IMAGE_MAPPING_FILE, raw_dir)
        except Exception:
            pass  # 退回 HF
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id='XiaomiMiMo/MiMo-V2.6-RL-oss',
        filename=IMAGE_MAPPING_FILE,
        repo_type='dataset',
        local_dir=raw_dir,
    )


def load_image_mapping(raw_dir: str = 'data/raw') -> dict:
    """读映射表 {dataset_image: dockerhub_image}；文件不存在返回 {}（不抛）。"""
    path = mapping_path(raw_dir)
    if not os.path.exists(path):
        return {}
    mapping: dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ds = rec.get('dataset_image')
            dh = rec.get('dockerhub_image')
            if isinstance(ds, str) and isinstance(dh, str) and ds and dh:
                mapping[ds] = dh
    return mapping


def resolve_dockerhub_image(dataset_image: str, mapping: dict) -> str | None:
    """把 parquet 里的本地镜像名解析为 Docker Hub 全名；未收录返回 None。

    若 dataset_image 本身已是 xiaomimimo/mimo-v2.6-rl-oss（用户 retag 过或手写全名），
    直接原样返回。
    """
    if not dataset_image:
        return None
    if 'xiaomimimo/mimo-v2.6-rl-oss' in dataset_image:
        return dataset_image
    return mapping.get(dataset_image)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='下载/校验 image-mapping.jsonl')
    ap.add_argument('--raw-dir', default='data/raw')
    args = ap.parse_args()
    p = ensure_image_mapping(args.raw_dir)
    m = load_image_mapping(args.raw_dir)
    print(f'[image-mapping] {p} entries={len(m)}')
