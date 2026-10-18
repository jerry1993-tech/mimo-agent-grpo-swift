"""MCP HTTP 客户端（general 域 sidecar）。

general 域的工具不在内置 bash/read/write/submit 内，而是透传到外接 MCP server
（MCP over HTTP / Streamable HTTP，JSON-RPC）。本模块提供一个工厂：

    dispatcher = await mcp_dispatcher_factory(spec)
    # dispatcher: async (spec, ToolCall) -> (obs_str, is_error)

MCP server URL 来源：
- ``spec.extra['mcp_servers']``：dict，两种形态都支持：
    { "<tool_or_server_name>": "https://.../mcp" }                 # 直连 URL
    { "<server_name>": {"url": "https://.../mcp", "tools": ["t1","t2"]} }
- 读不到任何配置 -> 工厂优雅返回 None（turn_executor 走内置工具分支）。

MCP 协议：先 ``initialize``（拿 Mcp-Session-Id）-> ``notifications/initialized``
-> ``tools/call``。无配置或初始化失败不抛，返回错误 observation。
依赖：httpx。
"""
from __future__ import annotations

# ---- 项目根 sys.path 兜底 ----
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
from typing import Any, Callable, Dict, Optional

import httpx

from swift_agent.spec import TaskSpec
from swift_agent.tool_parser import ToolCall


def _content_text(result: Dict[str, Any]) -> str:
    """把 MCP tools/call 结果的 content 块拼成文本。"""
    parts = []
    for c in result.get('content', []) or []:
        if isinstance(c, dict) and c.get('type') == 'text':
            parts.append(str(c.get('text', '')))
        else:
            parts.append(str(c))
    if parts:
        return '\n'.join(parts)
    return json.dumps(result, ensure_ascii=False)


class _MCPSession:
    """单个 MCP server 的流式 HTTP 会话（懒初始化握手）。"""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={'Accept': 'application/json, text/event-stream'},
        )
        self.session_id: Optional[str] = None
        self.inited = False
        self._id = 0

    async def _rpc(self, method: str, params: Optional[Dict[str, Any]] = None,
                   notify: bool = False) -> Dict[str, Any]:
        self._id += 1
        body: Dict[str, Any] = {'jsonrpc': '2.0', 'method': method}
        if not notify:
            body['id'] = self._id
        if params is not None:
            body['params'] = params
        headers = {}
        if self.session_id:
            headers['mcp-session-id'] = self.session_id
        resp = await self.client.post(self.url, json=body, headers=headers)
        if 'mcp-session-id' in resp.headers and not self.session_id:
            self.session_id = resp.headers['mcp-session-id']
        text = resp.text or ''
        # Streamable HTTP 可能返回 SSE：抽 data: 行
        ctype = resp.headers.get('content-type', '')
        if 'event-stream' in ctype:
            for line in text.splitlines():
                if line.startswith('data:'):
                    text = line[5:].strip()
                    break
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, list):
            data = data[0] if data else {}
        return data if isinstance(data, dict) else {}

    async def initialize(self) -> None:
        if self.inited:
            return
        await self._rpc('initialize', {
            'protocolVersion': '2024-11-05',
            'capabilities': {},
            'clientInfo': {'name': 'mimo-sandbox', 'version': '1.0'},
        })
        try:
            await self._rpc('notifications/initialized', notify=True)
        except Exception:
            pass
        self.inited = True

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            await self.initialize()
        except Exception:
            # 握手失败：跳过握手，直接 tools/call（兼容无状态 JSON-RPC 网关）
            self.inited = True
        data = await self._rpc('tools/call', {'name': name, 'arguments': arguments})
        if 'error' in data:
            raise RuntimeError(json.dumps(data['error'], ensure_ascii=False))
        result = data.get('result', {}) or {}
        if result.get('isError'):
            raise RuntimeError(_content_text(result))
        return _content_text(result)

    async def aclose(self) -> None:
        try:
            await self.client.aclose()
        except Exception:
            pass


def _build_tool_map(mcp_servers: Dict[str, Any]) -> Dict[str, str]:
    """把 spec.extra['mcp_servers'] 展开成 tool_name -> server_url。"""
    tool_url: Dict[str, str] = {}
    for key, val in (mcp_servers or {}).items():
        if isinstance(val, str) and val.startswith(('http://', 'https://')):
            # {tool_or_server_name: url}
            tool_url[key] = val
        elif isinstance(val, dict):
            url = val.get('url')
            if not url:
                continue
            for t in (val.get('tools') or []):
                tname = t.get('name') if isinstance(t, dict) else t
                if isinstance(tname, str):
                    tool_url[tname] = url
            # 也允许直接以 server key 调用
            tool_url.setdefault(key, url)
    return tool_url


async def mcp_dispatcher_factory(spec: TaskSpec) -> Optional[Callable]:
    """返回一个 async (spec, ToolCall) -> (obs_str, is_error) 的 MCP 分发器。

    无任何 MCP 配置时优雅返回 None（调用方回退到内置工具分支）。
    """
    mcp_servers = (spec.extra or {}).get('mcp_servers')
    if not mcp_servers and spec.tools:
        # 兜底：从 tools manifest 里找带 server 标注的工具
        mcp_servers = (spec.extra or {}).get('mcp_servers')
    tool_url = _build_tool_map(mcp_servers or {})
    if not tool_url:
        return None

    sessions: Dict[str, _MCPSession] = {}

    async def _dispatcher(s: TaskSpec, call: ToolCall):
        url = tool_url.get(call.name)
        if not url:
            return (f'[mcp] no server configured for tool "{call.name}"', True)
        sess = sessions.get(url)
        if sess is None:
            sess = _MCPSession(url)
            sessions[url] = sess
        try:
            out = await sess.call_tool(call.name, dict(call.arguments or {}))
            return (out or '(empty mcp result)', False)
        except Exception as e:
            return (f'[mcp error] {call.name}: {e}', True)

    return _dispatcher
