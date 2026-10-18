"""Remote 沙箱后端：HTTP 客户端，对接独立部署的 sandbox/server.py（外接沙箱）。

一任务一 session：
- start()  POST /sessions            body=env_config -> {session_id}
- exec()   POST /sessions/{id}/exec  body={command,timeout?,cwd?} -> ExecResult dict
- write()  POST /sessions/{id}/files body={path,content}
- read()   GET  /sessions/{id}/files?path=... -> {content}
- stop()   DELETE /sessions/{id}

base_url 取 ``TaskSpec.sandbox_url``。传输/超时错误映射到 ExecResult.reason='transport_error'。
依赖：httpx。
"""
from __future__ import annotations

# ---- 项目根 sys.path 兜底 ----
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from typing import Optional

import httpx

from sandbox.base import SandboxBackend, ExecResult, register_backend
from swift_agent.spec import TaskSpec


@register_backend('remote')
class RemoteBackend(SandboxBackend):
    name = 'remote'

    def __init__(self, spec: TaskSpec, **opts):
        super().__init__(spec, **opts)
        if not spec.sandbox_url:
            raise ValueError('RemoteBackend requires TaskSpec.sandbox_url')
        self.base_url = spec.sandbox_url.rstrip('/')
        self.session_id: Optional[str] = None
        self._timeout = float(opts.get('http_timeout',
                               _os.environ.get('SANDBOX_HTTP_TIMEOUT', '30.0')))
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)

    # ------------------------------------------------------------------
    async def start(self) -> None:
        env_config = self.spec.to_env_config()
        try:
            resp = await self.client.post('/sessions', json=env_config)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            await self._close_silent()
            raise RuntimeError(f'create remote session failed: {e}') from e
        self.session_id = data.get('session_id')
        if not self.session_id:
            await self._close_silent()
            raise RuntimeError(f'create session returned no session_id: {data}')
        self._started = True

    async def stop(self) -> None:
        try:
            if self.session_id:
                await self.client.delete(f'/sessions/{self.session_id}')
        except Exception:
            pass  # 幂等吞异常
        finally:
            self.session_id = None
            await self._close_silent()
            self._started = False

    async def _close_silent(self) -> None:
        try:
            await self.client.aclose()
        except Exception:
            pass

    # ------------------------------------------------------------------
    async def exec(self, command: str, timeout: Optional[float] = None,
                   cwd: Optional[str] = None) -> ExecResult:
        if not self.session_id:
            return ExecResult(output='remote session not started', returncode=-1,
                              reason='backend_error')
        body = {'command': command, 'timeout': timeout, 'cwd': cwd}
        try:
            resp = await self.client.post(f'/sessions/{self.session_id}/exec', json=body)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return ExecResult(output='remote exec timed out (transport)', returncode=-1,
                              reason='transport_error')
        except Exception as e:
            return ExecResult(output=f'remote exec transport error: {e}', returncode=-1,
                              reason='transport_error')
        return ExecResult(
            output=data.get('output', ''),
            returncode=int(data.get('returncode', -1)),
            reason=data.get('reason', 'backend_error'),
            truncated=bool(data.get('truncated', False)),
        )

    # ------------------------------------------------------------------
    async def write_file(self, path: str, content: str) -> None:
        if not self.session_id:
            raise RuntimeError('remote session not started')
        resp = await self.client.post(f'/sessions/{self.session_id}/files',
                                      json={'path': path, 'content': content})
        resp.raise_for_status()

    async def read_file(self, path: str) -> str:
        if not self.session_id:
            raise RuntimeError('remote session not started')
        resp = await self.client.get(f'/sessions/{self.session_id}/files',
                                     params={'path': path})
        if resp.status_code == 404:
            raise FileNotFoundError(path)
        resp.raise_for_status()
        return resp.json().get('content', '')
