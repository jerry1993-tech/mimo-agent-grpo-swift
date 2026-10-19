"""外接沙箱 HTTP 服务（FastAPI）——管理多个 per-task session。

典型部署：GPU 训练实例（ms-swift GRPO）经 RemoteBackend 连本服务；
本服务在 CPU/容器宿主机上运行，每个 session 持有一个 SandboxBackend
（默认 DockerBackend，一任务一容器）。环境变量 ``SANDBOX_BACKEND=local``
可切到 LocalBackend 便于本机演示（无隔离，勿用于生产）。

端点（OpenAPI 一致，pydantic 模型）：
- POST   /sessions                  body=env_config(dict) -> {session_id}
- POST   /sessions/{id}/exec         body={command,timeout?,cwd?} -> ExecResult dict
- POST   /sessions/{id}/files        body={path,content}
- GET    /sessions/{id}/files?path=  -> {content}
- POST   /sessions/{id}/upload_dir   multipart(tarball, remote_dir) -> {files:n}
- DELETE /sessions/{id}
- GET    /healthz                    （免鉴权，供探活）

环境变量：
- SANDBOX_HOST / SANDBOX_PORT        监听地址（默认 0.0.0.0:8000）
- SANDBOX_BACKEND                    docker | local（默认 docker）
- SANDBOX_TOKEN                      共享密钥；设置后所有业务端点要求
                                      请求头 X-Sandbox-Token 匹配（强烈建议
                                      公网部署时设置，并配合安全组白名单）
- SANDBOX_MAX_CONCURRENT_SESSIONS    并发 session 上限（0=不限）。超出时
                                      create 排队等待（适配小规格宿主机）。
- SANDBOX_SESSION_IDLE_TTL           session 空闲回收秒数（默认 900）。训练侧
                                      max_turns 到顶的轨迹不会调用 close（swift
                                      4.5.2 GYMScheduler 只在 done 时关闭），
                                      靠空闲回收兜底，防止会话/容器泄漏耗尽并发。
- SANDBOX_CREATE_WAIT                 create 排队最长等待秒数（默认 600），
                                      超时返回 503 而非无限阻塞（避免训练死等）。

启动：``python3 -m sandbox.server``。
依赖：fastapi, uvicorn, httpx, pydantic, python-multipart。
"""
from __future__ import annotations

# ---- 项目根 sys.path 兜底 ----
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import asyncio
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sandbox.base import SandboxBackend, get_backend
from swift_agent.spec import TaskSpec

# ----------------------------------------------------------------------
# Pydantic 请求/响应模型
# ----------------------------------------------------------------------
class ExecRequest(BaseModel):
    command: str
    timeout: Optional[float] = None
    cwd: Optional[str] = None


class WriteFileRequest(BaseModel):
    path: str
    content: str = ''


class CreateSessionResponse(BaseModel):
    session_id: str


class ReadFileResponse(BaseModel):
    content: str


class UploadDirResponse(BaseModel):
    files: int


class HealthResponse(BaseModel):
    status: str = 'ok'
    sessions: int = 0
    max_sessions: int = 0


# ----------------------------------------------------------------------
# App + session 表 + 并发闸门
# ----------------------------------------------------------------------
app = FastAPI(title='MiMo Sandbox Server', version='1.1')
SESSIONS: Dict[str, SandboxBackend] = {}

_MAX_SESSIONS = int(_os.environ.get('SANDBOX_MAX_CONCURRENT_SESSIONS', '0') or 0)
_SEMAPHORE: Optional[asyncio.Semaphore] = (
    asyncio.Semaphore(_MAX_SESSIONS) if _MAX_SESSIONS > 0 else None
)
_IDLE_TTL = float(_os.environ.get('SANDBOX_SESSION_IDLE_TTL', '900'))
_CREATE_WAIT = float(_os.environ.get('SANDBOX_CREATE_WAIT', '600'))
# session 最近活跃时间（unix 秒）；用于空闲回收
LAST_USED: Dict[str, float] = {}
_reaper_task: Optional[asyncio.Task] = None


def _touch(sid: str) -> None:
    if sid in LAST_USED:
        LAST_USED[sid] = time.time()


async def _reaper() -> None:
    """周期性回收空闲 session（兜底 max_turns 到顶未 close 的泄漏）。"""
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for sid in list(SESSIONS.keys()):
            if now - LAST_USED.get(sid, now) < _IDLE_TTL:
                continue
            be = SESSIONS.pop(sid, None)
            LAST_USED.pop(sid, None)
            if be is not None:
                try:
                    await be.stop()
                except Exception:
                    pass
            if _SEMAPHORE is not None:
                _SEMAPHORE.release()
            print(f'[reaper] closed idle session {sid} (>{_IDLE_TTL:.0f}s)')


def require_token(x_sandbox_token: str = Header(default='')) -> None:
    """共享密钥鉴权：SANDBOX_TOKEN 设置后，请求头必须带相同值。"""
    expected = _os.environ.get('SANDBOX_TOKEN', '')
    if expected and x_sandbox_token != expected:
        raise HTTPException(status_code=401, detail='invalid or missing X-Sandbox-Token')


def _backend_override(spec: TaskSpec) -> TaskSpec:
    """SANDBOX_BACKEND 环境变量可全局切后端（本机演示用 local）。"""
    forced = _os.environ.get('SANDBOX_BACKEND')
    if forced in ('docker', 'local', 'remote'):
        spec.backend = forced
    return spec


@app.on_event('startup')
async def _startup() -> None:
    # 触发内置后端注册（惰性导入）
    try:
        from sandbox import docker_backend, local_backend, remote_backend  # noqa: F401
    except Exception:
        pass
    global _reaper_task
    if _reaper_task is None:
        _reaper_task = asyncio.create_task(_reaper())


@app.on_event('shutdown')
async def _shutdown() -> None:
    # 优雅关闭：销毁所有残留 session
    for sid in list(SESSIONS.keys()):
        be = SESSIONS.pop(sid, None)
        if be is not None:
            try:
                await be.stop()
            except Exception:
                pass
        if _SEMAPHORE is not None:
            _SEMAPHORE.release()


@app.get('/healthz', response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status='ok', sessions=len(SESSIONS), max_sessions=_MAX_SESSIONS)


@app.post('/sessions', response_model=CreateSessionResponse,
          dependencies=[Depends(require_token)])
async def create_session(env_config: Dict[str, Any]) -> CreateSessionResponse:
    # 并发闸门：达到上限时排队，但有界（SANDBOX_CREATE_WAIT），
    # 超时返回 503 -> 训练侧按 infra 处理，不会无限阻塞 on_trajectory_start。
    if _SEMAPHORE is not None:
        try:
            await asyncio.wait_for(_SEMAPHORE.acquire(), timeout=_CREATE_WAIT)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail=f'sandbox at capacity (>{_MAX_SESSIONS} sessions busy for '
                       f'{_CREATE_WAIT:.0f}s); retry later or raise '
                       f'SANDBOX_MAX_CONCURRENT_SESSIONS / lower training concurrency')
    try:
        spec = TaskSpec.from_env_config(env_config)
        spec = _backend_override(spec)
        backend = get_backend(spec)
        await backend.start()
    except FileNotFoundError as e:
        # 例如本机无 docker -> 503 基建错误
        if _SEMAPHORE is not None:
            _SEMAPHORE.release()
        raise HTTPException(status_code=503, detail=f'sandbox backend unavailable: {e}')
    except Exception as e:
        if _SEMAPHORE is not None:
            _SEMAPHORE.release()
        raise HTTPException(status_code=500, detail=f'failed to start session: {e}')
    sid = uuid.uuid4().hex
    SESSIONS[sid] = backend
    LAST_USED[sid] = time.time()
    return CreateSessionResponse(session_id=sid)


def _get(sid: str) -> SandboxBackend:
    be = SESSIONS.get(sid)
    if be is None:
        raise HTTPException(status_code=404, detail=f'session not found: {sid}')
    return be


@app.post('/sessions/{sid}/exec', dependencies=[Depends(require_token)])
async def exec_command(sid: str, req: ExecRequest) -> JSONResponse:
    be = _get(sid)
    _touch(sid)
    try:
        res = await be.exec(req.command, timeout=req.timeout, cwd=req.cwd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'exec failed: {e}')
    return JSONResponse(res.to_dict())


@app.post('/sessions/{sid}/files', status_code=204,
          dependencies=[Depends(require_token)])
async def write_file(sid: str, req: WriteFileRequest) -> JSONResponse:
    be = _get(sid)
    _touch(sid)
    try:
        await be.write_file(req.path, req.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'write_file failed: {e}')
    return JSONResponse({'ok': True})


@app.get('/sessions/{sid}/files', response_model=ReadFileResponse,
         dependencies=[Depends(require_token)])
async def read_file(sid: str, path: str) -> ReadFileResponse:
    be = _get(sid)
    _touch(sid)
    try:
        content = await be.read_file(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'file not found: {path}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'read_file failed: {e}')
    return ReadFileResponse(content=content)


@app.post('/sessions/{sid}/upload_dir', response_model=UploadDirResponse,
          dependencies=[Depends(require_token)])
async def upload_dir(sid: str, remote_dir: str = Form(...),
                     tarball: UploadFile = File(...)) -> UploadDirResponse:
    """接收 tar 归档（multipart），服务端解包进该 session 的容器 remote_dir。"""
    be = _get(sid)
    _touch(sid)
    try:
        tar_bytes = await tarball.read()
        n = await be.upload_tarball(tar_bytes, remote_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'upload_dir failed: {e}')
    return UploadDirResponse(files=n)


@app.delete('/sessions/{sid}', status_code=204,
            dependencies=[Depends(require_token)])
async def delete_session(sid: str) -> JSONResponse:
    be = SESSIONS.pop(sid, None)
    LAST_USED.pop(sid, None)
    if be is not None:
        try:
            await be.stop()
        except Exception:
            pass  # 幂等吞异常
    if _SEMAPHORE is not None:
        _SEMAPHORE.release()
    return JSONResponse({'deleted': True})


# ----------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    host = _os.environ.get('SANDBOX_HOST', '0.0.0.0')
    port = int(_os.environ.get('SANDBOX_PORT', '8000'))
    uvicorn.run(app, host=host, port=port)
