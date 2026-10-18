#!/usr/bin/env python3
"""IL 预热：用独立 agent runner 采集成功轨迹。

从 GRPO jsonl 读任务（TaskSpec.from_env_config 还原），并发跑 episode，
把每条轨迹（messages + reward/is_infra）写到 data/outputs/rollout_dump.jsonl，
供 prepare_sft.py 进一步过滤成 SFT 数据。

依赖集成方保证存在的模块（不在本仓库时仅保证 py_compile 通过、不实际运行）：
    from swift_agent.agent_runner import run_episode, OpenAIPolicyClient, EpisodeResult

用法（工程根目录运行，需提供 policy 端点）：
    OPENAI_BASE_URL=... OPENAI_API_KEY=... MODEL_NAME=qwen... \
        python3 data/collect_il_demo.py --input data/outputs/grpo_train.jsonl \
            --max-tasks 20 --concurrency 4
"""
from __future__ import annotations

import os
import sys
import json
import asyncio
import argparse
from typing import Any, Dict, List

# 项目根 sys.path 兜底
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swift_agent.spec import TaskSpec  # noqa: E402  (框架无关)
# 集成方保证存在；此处顶层 import 仅在真正运行时解析，py_compile 不受影响
from swift_agent.agent_runner import (  # noqa: E402
    run_episode, OpenAIPolicyClient, EpisodeResult,
)


def load_tasks(input_path: str, max_tasks: int | None) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            spec = TaskSpec.from_env_config(rec['env_config'])
            tasks.append(spec)
            if max_tasks is not None and len(tasks) >= max_tasks:
                break
    return tasks


async def run_one(spec: TaskSpec, policy: OpenAIPolicyClient,
                  max_turns: int | None, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        result: EpisodeResult = await run_episode(
            spec, policy, max_turns=max_turns, tool_dispatcher=None)
        return {
            'messages': result.messages,
            'reward': result.reward,
            'is_infra': result.is_infra,
            'num_turns': getattr(result, 'num_turns', None),
            'stop_reason': getattr(result, 'stop_reason', None),
            'domain': spec.domain,
            'instance_id': spec.instance_id,
        }


async def run_async(args: argparse.Namespace) -> None:
    tasks = load_tasks(args.input, args.max_tasks)
    print(f'[load] {len(tasks)} tasks from {args.input}')

    policy = OpenAIPolicyClient(
        base_url=args.policy_base_url,
        api_key=args.policy_api_key,
        model=args.policy_model,
    )
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *[run_one(t, policy, args.max_turns, sem) for t in tasks],
        return_exceptions=True,
    )

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    ok = fail = 0
    with open(args.output, 'w', encoding='utf-8') as fout:
        for res in results:
            if isinstance(res, Exception):
                fail += 1
                print(f'[episode failed] {type(res).__name__}: {res}')
                continue
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
            ok += 1
    print(f'[done] ok={ok} fail={fail} -> {args.output}')


def main() -> None:
    ap = argparse.ArgumentParser(description='用独立 agent runner 采集成功轨迹（IL 预热）')
    ap.add_argument('--input', default='data/outputs/grpo_train.jsonl',
                    help='GRPO jsonl 输入')
    ap.add_argument('--output', default='data/outputs/rollout_dump.jsonl',
                    help='轨迹 dump 输出路径')
    ap.add_argument('--max-tasks', type=int, default=None, help='最多跑多少任务')
    ap.add_argument('--max-turns', type=int, default=None, help='episode 最大轮次')
    ap.add_argument('--concurrency', type=int, default=4, help='并发数（默认 4）')
    ap.add_argument('--policy-base-url', default=os.environ.get('OPENAI_BASE_URL'))
    ap.add_argument('--policy-api-key', default=os.environ.get('OPENAI_API_KEY'))
    ap.add_argument('--policy-model', default=os.environ.get('MODEL_NAME'))
    args = ap.parse_args()

    if not args.policy_base_url or not args.policy_model:
        print('[warn] 未提供 policy-base-url / policy-model（可用环境变量 '
              'OPENAI_BASE_URL / MODEL_NAME）；无法实际运行。仅完成编译自检。')
        return
    asyncio.run(run_async(args))


if __name__ == '__main__':
    main()
