"""Remote 沙箱后端：HTTP 客户端，对接独立部署的 sandbox/server.py（外接沙箱）。

典型部署：GPU 训练实例本进程 + 远端 CPU 服务器（SANDBOX_BACKEND=docker）跑 per-task 容器。

一任务一 session：
- start()  POST /sessions               body=env_config -> {session_id}
- exec()   POST /sessions/{id}/exec     body={command,timeout?,cwd?} -> ExecResult dict
- write()  POST /sessions/{id}/files    body={path,content}
- read()   GET  /sessions/{id}/files?path=... -> {content}
- upload_dir() POST /sessions/{id}/upload_dir (multipart tar) -> {files:n}
- stop()   DELETE /sessions/{id}

要点：
- base_url 取 ``TaskSpec.sandbox_url``；
- 鉴权：若设置了环境变量 ``SANDBOX_TOKEN``（或 opts['token']），所有请求带
  ``X-Sandbox-Token`` 头；token 只走运行时环境变量，**不写入 env_config/训练数据**；
- exec 的 HTTP 超时随命令超时放大（默认传输超时 30s，命令 timeout>30s 时取
  timeout+60s），避免 verifier/长命令被传输层误杀；
- 传输/超时错误映射到 ExecResult.reason='transport_error'。
依赖：httpx。
"""
from __future__ import annotations

# ---- 项目根 sys.path 兜底 ----
import io as _io
import os as _os
import sys as _sys
import tarfile as _tarfile
from typing import Optional

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

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
        self._http_timeout = float(opts.get('http_timeout',
                                _os.environ.get('SANDBOX_HTTP_TIMEOUT', '30.0')))
        # 鉴权 token：opts > 环境变量 SANDBOX_TOKEN（不落训练数据）
        self._token: str = str(opts.get('token') or _os.environ.get('SANDBOX_TOKEN', ''))
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self._http_timeout,
            headers=self._auth_headers(),
        )

    def _auth_headers(self) -> dict:
        return {'X-Sandbox-Token': self._token} if self._token else {}

    # ------------------------------------------------------------------
    async def start(self) -> None:
        env_config = self.spec.to_env_config()
        # 会话创建可能触发服务端 docker pull（任务镜像 1~2GB，公共加速器下更久）：
        # 总超时取 SANDBOX_CREATE_TIMEOUT（默认 1800s，与服务端 pull 墙钟一致）。
        create_timeout = float(_os.environ.get('SANDBOX_CREATE_TIMEOUT', '1800'))
        try:
            resp = await self.client.post(
                '/sessions', json=env_config,
                timeout=max(self._http_timeout, create_timeout))
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
        # HTTP 超时随命令超时放大：默认传输 30s；命令 timeout 更长时取 timeout+60s
        req_timeout = self._http_timeout
        if timeout is not None and float(timeout) + 60.0 > req_timeout:
            req_timeout = float(timeout) + 60.0
        try:
            resp = await self.client.post(f'/sessions/{self.session_id}/exec',
                                          json=body, timeout=req_timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return ExecResult(output=f'remote exec timed out (transport, {req_timeout:.0f}s)',
                              returncode=-1, reason='transport_error')
        except Exception as e:
            return ExecResult(output=f'remote exec transport error: {e}',
                              returncode=-1, reason='transport_error')
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
                                      json={'path': path, 'content': content},
                                      timeout=max(self._http_timeout, 120.0))
        resp.raise_for_status()

    async def read_file(self, path: str) -> str:
        if not self.session_id:
            raise RuntimeError('remote session not started')
        resp = await self.client.get(f'/sessions/{self.session_id}/files',
                                     params={'path': path},
                                     timeout=max(self._http_timeout, 120.0))
        if resp.status_code == 404:
            raise FileNotFoundError(path)
        resp.raise_for_status()
        return resp.json().get('content', '')

    # ------------------------------------------------------------------
    async def upload_dir(self, local_dir: str, remote_dir: str) -> int:
        """本地目录打 tar -> multipart 上传 -> 服务端解包进容器。

        （general 域 env_task_dir：文件在训练实例本地，经 HTTP 传进远端容器。）
        """
        if not self.session_id:
            raise RuntimeError('remote session not started')
        local_dir = _os.path.abspath(local_dir)
        if not _os.path.isdir(local_dir):
            raise FileNotFoundError(f'local dir not found: {local_dir}')
        buf = _io.BytesIO()
        n = 0
        with _tarfile.open(fileobj=buf, mode='w') as tf:
            for root, _dirs, files in _os.walk(local_dir):
                for fname in files:
                    fpath = _os.path.join(root, fname)
                    tf.add(fpath, arcname=_os.path.relpath(fpath, local_dir))
                    n += 1
        if n == 0:
            return 0
        resp = await self.client.post(
            f'/sessions/{self.session_id}/upload_dir',
            files={'tarball': ('dir.tar', buf.getvalue(), 'application/x-tar')},
            data={'remote_dir': remote_dir},
            timeout=max(self._http_timeout, 600.0),
        )
        resp.raise_for_status()
        return int(resp.json().get('files', n))
