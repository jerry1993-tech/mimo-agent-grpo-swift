#!/usr/bin/env python3
"""GRPO 数据准备：把 parquet「任务环境池」归一化为 GRPO jsonl。

每行输出：
    {"messages": [{"role": "user", "content": spec.problem_statement}],
     "env_config": spec.to_env_config(runtime_overrides)}

- 每行用 TaskSpec.from_parquet_row 归一化（契约：swift_agent/spec.py）。
- runtime_overrides 由 CLI 注入后端/评判器/超时等运行期开关，写入 env_config 顶层。
- 同时输出合并文件（默认 data/outputs/grpo_train.jsonl）与按域拆分文件
  grpo_{domain}.jsonl。
- 坏行跳过并计数、打印原因；结尾打印各域条数统计。

用法（工程根目录运行）：
    python3 data/prepare_grpo.py --domains music code --limit 5 --backend local
"""
from __future__ import annotations

import os
import sys
import json
import random
import argparse
from typing import Any, Dict, List, Optional

# 项目根 sys.path 兜底
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swift_agent.spec import TaskSpec  # noqa: E402  (框架无关)

REPO_ID = "XiaomiMiMo/MiMo-V2.6-RL-oss"

DOMAIN_PARQUETS = {
    'code': 'code.parquet',
    'webdev': 'webdev.parquet',
    'cyber': 'cyber.parquet',
    'music': 'music.parquet',
    'general': 'general/train.parquet',
}

# 已缓存的 HF snapshot（直接复用，勿重复下载）
_CACHE_SNAPSHOT = (
    "/runtime/cache/user/huggingface/hub/datasets--XiaomiMiMo--MiMo-V2.6-RL-oss/"
    "snapshots/639865fd3374018d6cb29b9fb82dd531406fcf5f"
)


def resolve_parquet(domain: str, raw_dir: str) -> str:
    """定位某域的 parquet：优先 raw_dir，其次 HF 全局缓存 snapshot，最后联网下载。"""
    rel = DOMAIN_PARQUETS[domain]
    candidates = [
        os.path.join(raw_dir, rel),
        os.path.join(_CACHE_SNAPSHOT, rel),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 兜底：按 repo 下载（hf_hub_download 会复用全局 cache）
    from huggingface_hub import hf_hub_download
    os.makedirs(raw_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=REPO_ID, filename=rel, repo_type='dataset', local_dir=raw_dir,
    )


def build_runtime_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """把 CLI 参数拼成 to_env_config 的 runtime_overrides（None 不注入）。"""
    ov: Dict[str, Any] = {}
    if args.backend is not None:
        ov['backend'] = args.backend
    if args.sandbox_url is not None:
        ov['sandbox_url'] = args.sandbox_url
    if args.judge_base_url is not None:
        ov['judge_base_url'] = args.judge_base_url
    if args.judge_model is not None:
        ov['judge_model'] = args.judge_model
    if args.judge_api_key is not None:
        ov['judge_api_key'] = args.judge_api_key
    if args.tool_timeout is not None:
        ov['tool_timeout_sec'] = float(args.tool_timeout)
    if args.max_tool_output_chars is not None:
        ov['max_tool_output_chars'] = int(args.max_tool_output_chars)
    # network 仅在显式传入时覆盖任务级 allow_internet，避免误关 terminal_bench
    if args.network:
        ov['network'] = True
    return ov


def process_domain(domain: str, raw_dir: str, overrides: Dict[str, Any],
                   limit: Optional[int], seed: int) -> (List[Dict[str, Any]], int, int):
    """读取单域 parquet，返回 (records, kept, bad)。"""
    import pyarrow.parquet as pq

    path = resolve_parquet(domain, raw_dir)
    tbl = pq.read_table(path)
    df = tbl.to_pandas()
    n = len(df)

    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    if limit is not None and limit > 0:
        order = order[:limit]

    records: List[Dict[str, Any]] = []
    bad = 0
    for i in order:
        row = df.iloc[i].to_dict()
        try:
            spec: TaskSpec = TaskSpec.from_parquet_row(row)
            if not spec.problem_statement:
                raise ValueError('empty problem_statement')
            rec = {
                'messages': [{'role': 'user', 'content': spec.problem_statement}],
                'env_config': spec.to_env_config(overrides),
            }
            records.append(rec)
        except Exception as e:  # 坏行：跳过并记录原因
            bad += 1
            print(f'  [bad row] domain={domain} row_index={i} reason={type(e).__name__}: {e}')
    return records, len(records), bad


def main() -> None:
    ap = argparse.ArgumentParser(description='parquet -> GRPO jsonl')
    ap.add_argument('--domains', nargs='*', default=list(DOMAIN_PARQUETS),
                    choices=list(DOMAIN_PARQUETS), help='要处理的领域，默认全部')
    ap.add_argument('--raw-dir', default='data/raw', help='parquet 所在目录')
    ap.add_argument('--limit', type=int, default=None,
                    help='每域取样上限（先按 --seed 洗牌再截断）')
    ap.add_argument('--seed', type=int, default=0, help='取样随机种子')
    ap.add_argument('--output-path', default='data/outputs/grpo_train.jsonl',
                    help='合并输出 jsonl 路径')
    # 运行期开关 -> env_config 顶层
    ap.add_argument('--backend', default='docker', choices=['docker', 'local', 'remote'],
                    help='沙箱后端（默认 docker）')
    ap.add_argument('--sandbox-url', default=None, help='remote 沙箱地址')
    ap.add_argument('--judge-base-url', default=None, help='LLM-as-judge base url')
    ap.add_argument('--judge-model', default=None, help='LLM-as-judge 模型名')
    ap.add_argument('--judge-api-key', default=None, help='LLM-as-judge api key')
    ap.add_argument('--tool-timeout', type=float, default=None, help='单工具调用超时秒')
    ap.add_argument('--max-tool-output-chars', type=int, default=None,
                    help='工具输出截断字符数')
    ap.add_argument('--network', action='store_true', help='允许联网（默认不允许）')
    args = ap.parse_args()

    overrides = build_runtime_overrides(args)
    out_dir = os.path.dirname(args.output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # 合并文件句柄
    total_kept = 0
    total_bad = 0
    stats: Dict[str, int] = {}
    with open(args.output_path, 'w', encoding='utf-8') as f_all:
        for domain in args.domains:
            records, kept, bad = process_domain(
                domain, args.raw_dir, overrides, args.limit, args.seed)
            total_kept += kept
            total_bad += bad
            stats[domain] = kept

            # 按域拆分输出
            domain_path = os.path.join(out_dir, f'grpo_{domain}.jsonl')
            with open(domain_path, 'w', encoding='utf-8') as f_dom:
                for rec in records:
                    f_dom.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    f_all.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f'[domain={domain}] kept={kept} bad={bad} -> {domain_path}')

    print('-' * 48)
    print('[stats] 各域条数:')
    for d, c in stats.items():
        print(f'  {d:8s} {c}')
    print(f'[stats] total_kept={total_kept} total_bad={total_bad}')
    print(f'[output] merged -> {args.output_path}')


if __name__ == '__main__':
    main()
