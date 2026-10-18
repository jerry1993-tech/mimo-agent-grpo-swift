"""独立 Agent episode 运行器（框架无关，不依赖 ms-swift 训练循环）。

用途：
1. data/collect_il_demo.py 用它批量跑任务、采集成功（reward==1）轨迹做 IL/SFT 预热；
2. tests/ 用它做离线闭环 smoke test（policy 可用 mock 实现）。

它复用与训练时完全相同的 turn_executor / verifier，因此 IL 采集与 RL rollout 的交互逻辑一致。
注意：训练（on-policy）走 swift_agent/mimo_env.py 的 gym Env，而不是这里；
这里仅用于“给定一个外部 policy（OpenAI 兼容）驱动若干 episode”。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import httpx  # noqa: E402

from swift_agent.spec import TaskSpec  # noqa: E402
from swift_agent.tool_parser import build_system_prompt, parse_tool_calls  # noqa: E402
from swift_agent.turn_executor import execute_calls, finalize  # noqa: E402
from sandbox.base import get_backend  # noqa: E402

_NO_TOOL_HINT = (
    'No <tool_call> detected. Emit a <tool_call> block to act, '
    'or call "submit" to finish.'
)


@dataclass
class EpisodeResult:
    messages: List[Dict[str, Any]]
    reward: float = 0.0
    is_infra: bool = False
    num_turns: int = 0
    components: Dict[str, float] = field(default_factory=dict)
    stop_reason: str = ''


class PolicyClient:
    """Policy 接口：给定 messages 返回 assistant 文本。"""

    async def chat(self, messages: List[Dict[str, Any]], tools=None) -> str:
        raise NotImplementedError


class OpenAIPolicyClient(PolicyClient):
    """OpenAI 兼容 chat/completions 客户端（可指向 vLLM/SGLang/任意网关）。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 1.0, top_p: float = 0.95,
                 max_tokens: int = 2048, timeout: float = 120.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or 'EMPTY'
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def chat(self, messages, tools=None) -> str:
        payload: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens,
        }
        if tools:
            payload['tools'] = tools
        headers = {'Authorization': f'Bearer {self.api_key}'}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f'{self.base_url}/chat/completions',
                                  json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data['choices'][0]['message'].get('content') or ''


class ScriptedPolicyClient(PolicyClient):
    """测试用：按预置的多轮脚本依次返回（每条是一轮 assistant 文本）。"""

    def __init__(self, scripted_turns: List[str]):
        self._turns = list(scripted_turns)
        self._i = 0

    async def chat(self, messages, tools=None) -> str:
        if self._i < len(self._turns):
            text = self._turns[self._i]
            self._i += 1
            return text
        return '<tool_call>\n{"name": "submit", "arguments": {}}\n</tool_call>'


async def run_episode(spec: TaskSpec, policy_client: PolicyClient,
                      max_turns: Optional[int] = None,
                      tool_dispatcher=None,
                      env_setup_hook=None) -> EpisodeResult:
    """驱动一条任务到结束，返回完整 messages 与最终 outcome。

    env_setup_hook: 可选 async (backend)->None，在环境启动后、agent 行动前调用，
    用于模拟“任务镜像内预置的基线环境”（如 code 域预置的 git 仓库基线）。
    """
    max_turns = int(max_turns or spec.num_turns or 10)

    # music：无沙箱，单轮生成 ABC 后直接规则打分
    if spec.domain == 'music':
        system = build_system_prompt(spec)
        messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': spec.problem_statement},
        ]
        text = await policy_client.chat(messages)
        messages.append({'role': 'assistant', 'content': text})
        from verifier.music_verifier import MusicVerifier
        res = MusicVerifier(spec).score_text(text)
        return EpisodeResult(messages=messages, reward=float(res.score),
                             is_infra=res.is_infra, num_turns=1,
                             components=res.components, stop_reason='music')

    async with get_backend(spec) as backend:
        if tool_dispatcher is None:
            try:
                from sandbox.mcp_client import mcp_dispatcher_factory
                tool_dispatcher = await mcp_dispatcher_factory(spec)
            except Exception:
                tool_dispatcher = None

        system = build_system_prompt(spec, spec.tools)
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': spec.problem_statement},
        ]
        # 模拟任务镜像预置的基线环境（如 code 域 git 基线）
        if env_setup_hook is not None:
            await env_setup_hook(backend)

        reward, is_infra, components, stop = 0.0, False, {}, 'max_turns'
        turn = 0
        for turn in range(1, max_turns + 1):
            text = await policy_client.chat(messages)
            messages.append({'role': 'assistant', 'content': text})

            calls = parse_tool_calls(text)
            if not calls:
                messages.append({'role': 'user', 'content': _NO_TOOL_HINT})
                continue

            execution = await execute_calls(backend, spec, calls, tool_dispatcher)
            messages.append({'role': 'user', 'content': execution.observation})

            if execution.is_infra:
                is_infra, stop = True, 'infra'
                break
            if execution.submitted:
                if execution.submit_answer:
                    spec.extra['answer'] = execution.submit_answer
                    try:
                        await backend.write_file('answer.md', execution.submit_answer)
                    except Exception:
                        pass
                vres = await finalize(spec, backend)
                reward = float(vres.score)
                components = vres.components
                is_infra = vres.is_infra
                stop = 'submitted'
                break

        return EpisodeResult(messages=messages, reward=reward, is_infra=is_infra,
                             num_turns=turn, components=components, stop_reason=stop)
