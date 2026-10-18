"""工具调用文本协议（框架无关，禁止 import ms-swift）。

复刻小米 ``qwen3_coder`` 工具解析的语义，但采用更易解析/测试的显式文本块：

    <tool_call>
    {"name": "bash", "arguments": {"command": "ls -la"}}
    </tool_call>

- 一条 assistant 文本可含 0..N 个 <tool_call> 块（也兼容 ```tool_call fenced）。
- 内置工具：bash / read_file / write_file / submit；general 域的 MCP 工具名透传。
- 工具结果以 <tool_result> ... </tool_result> 回灌。

参考（小米多轮工具解析）：
- https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/arvo/agent_loop.py  (ToolParser)
- https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rollout/gym_env.py
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swift_agent.spec import _safe_json_loads  # 复用宽容 JSON 解析

TOOL_CALL_OPEN = '<tool_call>'
TOOL_CALL_CLOSE = '</tool_call>'
TOOL_RESULT_OPEN = '<tool_result>'
TOOL_RESULT_CLOSE = '</tool_result>'

# 同时匹配 <tool_call>...</tool_call> 与 ```tool_call ... ```
_TOOL_CALL_RE = re.compile(
    r'(?:<tool_call>|```tool_call)\s*(.*?)\s*(?:</tool_call>|```)',
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ''
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'arguments': self.arguments}


# ----------------------------------------------------------------------
# 内置工具目录（OpenAI function-calling JSON Schema 风格）
# ----------------------------------------------------------------------
BUILTIN_TOOLS: List[Dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'bash',
            'description': 'Execute a shell command inside the task container and return its stdout/stderr.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {'type': 'string', 'description': 'Shell command to run via bash -lc.'},
                    'timeout': {'type': 'number', 'description': 'Optional per-command timeout in seconds.'},
                },
                'required': ['command'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_file',
            'description': 'Read a UTF-8 text file from the task container.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_file',
            'description': 'Write (overwrite) a UTF-8 text file in the task container.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'},
                },
                'required': ['path', 'content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'submit',
            'description': 'Finish the task and submit the final answer/patch for verification.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {'type': 'string', 'description': 'Optional final answer text.'},
                },
            },
        },
    },
]

BUILTIN_TOOL_NAMES = {t['function']['name'] for t in BUILTIN_TOOLS}


def parse_tool_calls(text: str) -> List[ToolCall]:
    """从 assistant 文本中抽取全部工具调用。非法块跳过（不抛异常）。"""
    calls: List[ToolCall] = []
    if not text:
        return calls
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        body = m.group(1).strip()
        obj = _safe_json_loads(body, default=None)
        if not isinstance(obj, dict):
            continue
        name = obj.get('name')
        if not name or not isinstance(name, str):
            continue
        args = obj.get('arguments', {})
        if isinstance(args, str):
            args = _safe_json_loads(args, default={}) or {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name=name, arguments=args, raw=body, index=i))
    return calls


def has_tool_call(text: str) -> bool:
    return bool(parse_tool_calls(text))


def is_submit(calls: List[ToolCall]) -> bool:
    return any(c.name == 'submit' for c in calls)


def render_observation(name: str, observation: str, is_error: bool = False) -> str:
    """把一次工具执行结果渲染成回灌文本。"""
    tag = 'error' if is_error else 'result'
    obs = observation if observation is not None else ''
    return f'{TOOL_RESULT_OPEN} name="{name}" status="{tag}"\n{obs}\n{TOOL_RESULT_CLOSE}'


def render_error(name: str, message: str) -> str:
    return render_observation(name, message, is_error=True)


# ----------------------------------------------------------------------
# 系统提示
# ----------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You are MiMo, an expert autonomous software & knowledge agent. \
You solve tasks by interacting with an isolated environment through tools.

For every action, output ONE tool call using this exact format (JSON only inside the tags):
<tool_call>
{{"name": "<tool_name>", "arguments": {{...}}}}
</tool_call>

Available tools:
- bash(command, timeout?): run a shell command in the task container.
- read_file(path): read a text file.
- write_file(path, content): create/overwrite a text file.
- submit(answer?): finish the task; the environment then runs the hidden verifier.

Rules:
1. Think step by step, but a <tool_call> block must contain valid JSON and nothing else.
2. You may inspect and modify files freely; the hidden tests are NOT present until you submit.
3. When the task is complete, call submit; do not keep acting after submitting.
4. Tool outputs are wrapped in <tool_result> tags; read them before the next action."""


def build_system_prompt(spec, extra_tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """构造系统提示；general 域可追加 MCP 工具清单说明。"""
    prompt = DEFAULT_SYSTEM_PROMPT
    if extra_tools:
        names = []
        for t in extra_tools:
            fn = t.get('function', t)
            if isinstance(fn, dict) and fn.get('name'):
                names.append(fn['name'])
        if names:
            prompt += '\n\nAdditional domain tools (MCP): ' + ', '.join(names) + '.'
    if spec.domain == 'music':
        # music 不走沙箱：直接产出 ABC 记谱
        prompt = (
            'You are a music composition expert. Output the composition in ABC notation '
            'inside a single ```abc fenced code block. Follow the requested style, meter, '
            'tempo (BPM) and number of voices. Output only the ABC notation.'
        )
    return prompt
