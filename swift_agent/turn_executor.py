"""单轮工具执行器（框架无关）。

把“模型最新 assistant 文本里的 tool_call -> 在后端真实执行 -> observation / 终局验证”
封装为可复用函数，供两处调用：
- swift_agent/mimo_env.py 的 gym Env.step（ms-swift 训练时，on-policy）；
- swift_agent/agent_runner.py 的独立 episode（IL 成功轨迹采集 / 离线 smoke test）。

对齐小米：逐轮 generate + observation 回灌；hidden 测试仅在 submit 后的 finalize() 注入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from swift_agent.spec import TaskSpec
from swift_agent.tool_parser import (
    ToolCall, render_observation, render_error,
)
from sandbox.base import SandboxBackend, ExecResult, truncate_output


# 工具分发器签名：async (spec, call) -> (observation_str, is_error)
ToolDispatcher = Callable[[TaskSpec, ToolCall], Any]


@dataclass
class TurnExecution:
    observation: str = ''
    submitted: bool = False
    submit_answer: str = ''
    is_infra: bool = False
    tool_error_count: int = 0
    executed: int = 0


async def _exec_bash(backend: SandboxBackend, spec: TaskSpec, call: ToolCall) -> tuple[str, bool]:
    command = call.arguments.get('command', '')
    if not isinstance(command, str) or not command.strip():
        return render_error('bash', 'Missing required argument "command".'), True
    timeout = call.arguments.get('timeout', spec.tool_timeout_sec)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = spec.tool_timeout_sec
    res: ExecResult = await backend.exec(command, timeout=timeout)
    text, was_truncated = truncate_output(res.output or '', spec.max_tool_output_chars)
    if res.reason == 'timeout':
        text = (text + f'\n[command timed out after {timeout}s]').strip()
    prefix = f'(returncode={res.returncode}, reason={res.reason})'
    body = text if text else '(no output)'
    is_error = res.returncode != 0
    return render_observation('bash', f'{prefix}\n{body}', is_error=is_error), is_error


async def _exec_read(backend: SandboxBackend, spec: TaskSpec, call: ToolCall) -> tuple[str, bool]:
    path = call.arguments.get('path', '')
    if not path:
        return render_error('read_file', 'Missing required argument "path".'), True
    try:
        content = await backend.read_file(path)
    except FileNotFoundError:
        return render_error('read_file', f'File not found: {path}'), True
    except Exception as e:  # transport / backend error
        return render_error('read_file', f'Error reading {path}: {e}'), True
    text, _ = truncate_output(content, spec.max_tool_output_chars)
    return render_observation('read_file', text), False


async def _exec_write(backend: SandboxBackend, spec: TaskSpec, call: ToolCall) -> tuple[str, bool]:
    path = call.arguments.get('path', '')
    content = call.arguments.get('content', '')
    if not path:
        return render_error('write_file', 'Missing required argument "path".'), True
    if not isinstance(content, str):
        content = str(content)
    try:
        await backend.write_file(path, content)
    except Exception as e:
        return render_error('write_file', f'Error writing {path}: {e}'), True
    return render_observation('write_file', f'Wrote {len(content)} chars to {path}.'), False


async def execute_calls(backend: SandboxBackend, spec: TaskSpec, calls: List[ToolCall],
                        tool_dispatcher: Optional[ToolDispatcher] = None) -> TurnExecution:
    """顺序执行一条 assistant 消息里的全部工具调用，汇总为一段 observation。"""
    result = TurnExecution()
    obs_parts: List[str] = []

    for call in calls:
        if result.submitted:
            break
        name = call.name
        try:
            if name == 'bash':
                obs, is_err = await _exec_bash(backend, spec, call)
            elif name == 'read_file':
                obs, is_err = await _exec_read(backend, spec, call)
            elif name == 'write_file':
                obs, is_err = await _exec_write(backend, spec, call)
            elif name == 'submit':
                result.submitted = True
                result.submit_answer = str(call.arguments.get('answer', '') or '')
                obs = render_observation('submit', 'Task submitted; running hidden verifier...')
                is_err = False
            elif tool_dispatcher is not None:
                # general 域 MCP 工具
                out = await tool_dispatcher(spec, call)
                if isinstance(out, tuple):
                    obs, is_err = out
                else:
                    obs, is_err = render_observation(name, str(out)), False
            else:
                obs = render_error(name, f'Unknown tool "{name}". '
                                         f'Available: bash, read_file, write_file, submit.')
                is_err = True
        except Exception as e:
            # 后端/传输异常 -> 记为基建问题，避免被当成模型失败
            result.is_infra = True
            obs = render_error(name, f'[sandbox backend error] {e}')
            is_err = True

        result.executed += 1
        if is_err:
            result.tool_error_count += 1
        obs_parts.append(obs)

    result.observation = '\n'.join(obs_parts)
    return result


async def finalize(spec: TaskSpec, backend: SandboxBackend) -> Any:
    """agent 结束后运行领域 verifier（真实执行 hidden 测试/rubric/规则）。"""
    from verifier.base import get_verifier, VerificationResult
    verifier = get_verifier(spec)
    return await verifier.verify(backend)
