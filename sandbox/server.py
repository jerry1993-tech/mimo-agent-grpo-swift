"""外接沙箱 HTTP 服务（FastAPI）——管理多个 per-task session。

每个 session 在服务端持有一个 SandboxBackend（默认 DockerBackend，一任务一容器）；
环境变量 ``SANDBOX_BACKEND=local`` 可切到 LocalBackend 便于本机演示（无隔离）。

端点（OpenAPI 一致，pydantic 模型）：
- POST   /sessions                  body=env_config(dict) -> {session_id}
- POST   /sessions/{id}/exec         body={command,timeout?,cwd?} -> ExecResult dict
- POST   /sessions/{id}/files        body={path,content}
- GET    /sessions/{id}/files?path=  -> {content}
- DELETE /sessions/{id}
- GET    /healthz

启动：``python3 -m sandbox.server``（host/port 取 SANDBOX_HOST / SANDBOX_PORT）。
依赖：fastapi, uvicorn, httpx, pydantic。
"""
from __future__ import annotations

# ---- 项目根 sys.path 兜底 ----
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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


class HealthResponse(BaseModel):
    status: str = 'ok'
    sessions: int = 0


# ----------------------------------------------------------------------
# App + session 表
# ----------------------------------------------------------------------
app = FastAPI(title='MiMo Sandbox Server', version='1.0')
SESSIONS: Dict[str, SandboxBackend] = {}


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


@app.get('/healthz', response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status='ok', sessions=len(SESSIONS))


@app.post('/sessions', response_model=CreateSessionResponse)
async def create_session(env_config: Dict[str, Any]) -> CreateSessionResponse:
    try:
        spec = TaskSpec.from_env_config(env_config)
        spec = _backend_override(spec)
        backend = get_backend(spec)
        await backend.start()
    except FileNotFoundError as e:
        # 例如本机无 docker -> 503 基建错误
        raise HTTPException(status_code=503, detail=f'sandbox backend unavailable: {e}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'failed to start session: {e}')
    sid = uuid.uuid4().hex
    SESSIONS[sid] = backend
    return CreateSessionResponse(session_id=sid)


def _get(sid: str) -> SandboxBackend:
    be = SESSIONS.get(sid)
    if be is None:
        raise HTTPException(status_code=404, detail=f'session not found: {sid}')
    return be


@app.post('/sessions/{sid}/exec')
async def exec_command(sid: str, req: ExecRequest) -> JSONResponse:
    be = _get(sid)
    try:
        res = await be.exec(req.command, timeout=req.timeout, cwd=req.cwd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'exec failed: {e}')
    return JSONResponse(res.to_dict())


@app.post('/sessions/{sid}/files', status_code=204)
async def write_file(sid: str, req: WriteFileRequest) -> JSONResponse:
    be = _get(sid)
    try:
        await be.write_file(req.path, req.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'write_file failed: {e}')
    return JSONResponse({'ok': True})


@app.get('/sessions/{sid}/files', response_model=ReadFileResponse)
async def read_file(sid: str, path: str) -> ReadFileResponse:
    be = _get(sid)
    try:
        content = await be.read_file(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'file not found: {path}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'read_file failed: {e}')
    return ReadFileResponse(content=content)


@app.delete('/sessions/{sid}', status_code=204)
async def delete_session(sid: str) -> JSONResponse:
    be = SESSIONS.pop(sid, None)
    if be is not None:
        try:
            await be.stop()
        except Exception:
            pass  # 幂等吞异常
    return JSONResponse({'deleted': True})


# ----------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    host = _os.environ.get('SANDBOX_HOST', '0.0.0.0')
    port = int(_os.environ.get('SANDBOX_PORT', '8000'))
    uvicorn.run(app, host=host, port=port)
