#!/usr/bin/env python3
"""SFT 数据准备：把「rollout 轨迹 dump」转成 SFT messages jsonl。

输入：rollout dump jsonl，每行形如
    {"messages": [...对话...], "reward": float, "is_infra": bool, ...}
输出：ms-swift 允许的 messages jsonl（role 仅 system/user/assistant/tool_call/tool_response）。

默认只保留 reward >= 阈值 且 is_infra=False 的成功轨迹（数据集本身无轨迹，
SFT 目标来自成功 rollout）。并把轨迹里内联的
    assistant 文本中的 <tool_call>{...}</tool_call>
    user 文本中的 <tool_result>...</tool_result>
规范化为 ms-swift 原生 role：
    {"role": "tool_call", "content": {"name": ..., "arguments": {...}}}
    {"role": "tool_response", "content": "<observation 文本>"}
参考 research/03_msswift_grpo_api.md §5.2。

用法（工程根目录运行）：
    python3 data/prepare_sft.py --input data/outputs/rollout_dump.jsonl --reward-threshold 1.0
"""
from __future__ import annotations

import os
import sys
import json
import re
import argparse
from typing import Any, Dict, List

# 项目根 sys.path 兜底
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swift_agent.tool_parser import (  # noqa: E402
    parse_tool_calls, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE,
)

# 剥离 <tool_call>...</tool_call> 块（含 ```tool_call fenced）
_TOOL_CALL_BLOCK_RE = re.compile(
    r'(?:<tool_call>|```tool_call)\s*.*?\s*(?:</tool_call>|```)', re.DOTALL)
# 剥离 <tool_result ...> ... </tool_result> 标签本身，保留内部 observation
_TOOL_RESULT_OPEN_RE = re.compile(
    re.escape(TOOL_RESULT_OPEN) + r'[^\n>]*\n?', re.DOTALL)
_TOOL_RESULT_CLOSE_RE = re.compile(r'\n?' + re.escape(TOOL_RESULT_CLOSE), re.DOTALL)


def _strip_tool_result_tags(content: str) -> str:
    inner = _TOOL_RESULT_OPEN_RE.sub('', content, count=1)
    inner = _TOOL_RESULT_CLOSE_RE.sub('', inner, count=1)
    return inner.strip()


def _is_tool_result_text(content: str) -> bool:
    return isinstance(content, str) and content.lstrip().startswith(TOOL_RESULT_OPEN)


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把内联标签轨迹规范化为 ms-swift 原生 role 序列。"""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if not isinstance(content, str):
            content = '' if content is None else str(content)

        # observation 回灌（user 文本里包着 <tool_result>）-> tool_response
        if role == 'user' and _is_tool_result_text(content):
            out.append({'role': 'tool_response',
                        'content': _strip_tool_result_tags(content)})
            continue

        if role == 'assistant':
            calls = parse_tool_calls(content)
            if calls:
                # 标签之外的散文思考保留为 assistant
                prose = _TOOL_CALL_BLOCK_RE.sub('', content).strip()
                if prose:
                    out.append({'role': 'assistant', 'content': prose})
                for c in calls:
                    out.append({
                        'role': 'tool_call',
                        'content': {'name': c.name, 'arguments': c.arguments},
                    })
                continue

        # system / user(任务) / 普通 assistant 原样保留
        out.append({'role': role, 'content': content})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='rollout dump -> SFT messages jsonl')
    ap.add_argument('--input', action='append', required=True,
                    help='rollout dump jsonl（可多次传入，如 rollout_*.jsonl）')
    ap.add_argument('--output', default='data/outputs/sft.jsonl',
                    help='输出 SFT jsonl 路径')
    ap.add_argument('--reward-threshold', type=float, default=1.0,
                    help='保留 reward >= 该阈值的轨迹（默认 1.0）')
    args = ap.parse_args()

    scanned = kept = drop_low = drop_infra = drop_bad = 0
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fout:
        for path in args.input:
            with open(path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    scanned += 1
                    try:
                        rec = json.loads(line)
                        reward = float(rec.get('reward', 0.0))
                        is_infra = bool(rec.get('is_infra', False))
                        messages = rec.get('messages', [])
                    except Exception as e:
                        drop_bad += 1
                        print(f'[bad record] {path}: {e}')
                        continue
                    if is_infra:
                        drop_infra += 1
                        continue
                    if reward < args.reward_threshold:
                        drop_low += 1
                        continue
                    norm = normalize_messages(messages)
                    fout.write(json.dumps({'messages': norm}, ensure_ascii=False) + '\n')
                    kept += 1

    print('-' * 48)
    print(f'[stats] scanned={scanned} kept={kept} '
          f'drop_low_reward={drop_low} drop_infra={drop_infra} drop_bad={drop_bad}')
    print(f'[output] {args.output}')


if __name__ == '__main__':
    main()
