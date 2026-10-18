#!/usr/bin/env python3
"""下载 general 域任务环境目录（general/envs/<task_id>/，共 925 个任务目录）。

事实：general 域 989 行中 925 行（general_agent 子集）的 instance_json 携带
env_task_dir（如 ``envs/s3k_0000_...``），指向 HF 数据集仓里的
``general/envs/<task_id>/`` 目录（manifest.json / verifier_meta.json / verify.py /
mcp_bridge.py / tools/ / system/ / workspace/ 等）。训练时沙箱需把这些文件
上传进容器（对齐上游 mimoagent DatasetEnvActor 的 copy_to 链路）。

用法（工程根目录运行）：
    # 下载全部 925 个任务目录（约数 GB，含 workspace 二进制）
    python -m data.download_general_envs
    # 只下载若干任务（配合 prepare_grpo --limit 取样训练）
    python -m data.download_general_envs --tasks s3k_0000_accounting_audit_tax_en_t1_rl_008
    # 按已生成的 grpo_general.jsonl 只取需要的任务
    python -m data.download_general_envs --from-jsonl data/outputs/grpo_general.jsonl --limit 50

说明：
- 下载走 huggingface_hub（尊重 HF_ENDPOINT=https://hf-mirror.com）；
- 文件级去重：已存在的文件跳过（resume 友好）；
- workspace/ 里的二进制（pdf/xlsx/docx）较大，全量下载前请确认磁盘。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ID = "XiaomiMiMo/MiMo-V2.6-RL-oss"
ENVS_PREFIX = "general/envs/"


def list_env_files(repo_files=None) -> "list[str]":
    """列出 general/envs/ 下全部文件（优先用本地缓存清单，其次 API）。"""
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'raw', 'general_env_files.txt')
    if repo_files is None:
        if os.path.exists(cache):
            with open(cache, encoding='utf-8') as f:
                repo_files = [ln.strip() for ln in f if ln.strip().startswith(ENVS_PREFIX)]
        else:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_files = api.list_repo_files(REPO_ID, repo_type="dataset")
            try:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                with open(cache, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(repo_files))
            except OSError:
                pass
    return [f for f in repo_files if f.startswith(ENVS_PREFIX)]


def task_dirs_of(files: "list[str]") -> "set[str]":
    """env 文件清单 -> 任务目录名集合（s3k_xxx）。"""
    dirs = set()
    for f in files:
        rest = f[len(ENVS_PREFIX):]
        if '/' in rest:
            dirs.add(rest.split('/', 1)[0])
    return dirs


def download_task_dir(task: str, output_dir: str, files: "list[str]") -> int:
    """下载单个任务目录全部文件（已存在跳过）。返回新下载文件数。"""
    from huggingface_hub import hf_hub_download
    prefix = f"{ENVS_PREFIX}{task}/"
    n = 0
    for rel in files:
        if not rel.startswith(prefix):
            continue
        local = os.path.join(output_dir, rel)
        if os.path.exists(local):
            continue
        try:
            hf_hub_download(repo_id=REPO_ID, filename=rel, repo_type='dataset',
                            local_dir=output_dir)
            n += 1
        except Exception as e:
            print(f'  [warn] {rel}: {type(e).__name__}: {str(e)[:150]}', file=sys.stderr)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description='下载 general/envs 任务环境目录')
    ap.add_argument('--output-dir', default='data/raw', help='落地根目录（general/envs/ 会建在其下）')
    ap.add_argument('--tasks', nargs='*', default=None,
                    help='只下载指定任务目录（s3k_xxx 名字，不含 general/envs/ 前缀）')
    ap.add_argument('--from-jsonl', default=None,
                    help='从 prepare_grpo 生成的 grpo_general.jsonl 读 env_task_dir 决定要下哪些')
    ap.add_argument('--limit', type=int, default=None, help='最多下载多少个任务目录')
    args = ap.parse_args()

    print(f'[1/3] 枚举 {ENVS_PREFIX} 文件清单 ...')
    files = list_env_files()
    all_tasks = sorted(task_dirs_of(files))
    print(f'  共 {len(all_tasks)} 个任务目录，{len(files)} 个文件')

    if args.tasks:
        want = set(args.tasks)
        missing = want - set(all_tasks)
        if missing:
            print(f'  [warn] 清单中不存在的任务: {sorted(missing)[:5]}')
        tasks = [t for t in all_tasks if t in want]
    elif args.from_jsonl:
        want = set()
        with open(args.from_jsonl, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cfg = json.loads(line).get('env_config', {})
                except json.JSONDecodeError:
                    continue
                d = str(cfg.get('env_task_dir', '')).strip()
                if d:
                    want.add(os.path.normpath(d).lstrip('./').split('/', 1)[-1]
                             if d.startswith('envs/') else d)
        tasks = [t for t in all_tasks if t in want]
    else:
        tasks = list(all_tasks)

    if args.limit:
        tasks = tasks[:args.limit]
    print(f'[2/3] 计划下载 {len(tasks)} 个任务目录 -> {args.output_dir}')

    t0 = time.time()
    total = 0
    for i, task in enumerate(tasks, 1):
        n = download_task_dir(task, args.output_dir, files)
        total += n
        if i % 20 == 0 or i == len(tasks):
            print(f'  [{i}/{len(tasks)}] {task}: +{n} files (cum {total}, {time.time()-t0:.0f}s)')
    print(f'[3/3] done: {total} new files, {len(tasks)} task dirs, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
