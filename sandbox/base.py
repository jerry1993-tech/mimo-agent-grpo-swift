"""沙箱后端抽象层（框架无关，禁止 import ms-swift）。

对齐小米实现：每个任务一个隔离容器/Pod，命令经 ``timeout N bash -lc`` 执行，
文件走 tar/cp，用完即毁。参考：
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/docker.py
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/kubernetes.py

本工程提供三个后端：
- DockerBackend（默认，需宿主机有 docker CLI/daemon）
- LocalBackend（临时目录 + asyncio.subprocess，【无隔离】，仅用于测试/低风险任务）
- RemoteBackend（HTTP 调用独立部署的 sandbox/server.py，适配“外接沙箱”）
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from swift_agent.spec import TaskSpec


@dataclass
class ExecResult:
    """单条命令执行结果（对齐 mimoagent {output, returncode, reason}）。"""
    output: str
    returncode: int
    reason: str = 'ok'          # ok | timeout | transport_error | backend_error
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'output': self.output,
            'returncode': self.returncode,
            'reason': self.reason,
            'truncated': self.truncated,
        }


def truncate_output(text: str, max_chars: int) -> tuple[str, bool]:
    """超长输出保留头尾，中间以截断标记替换（对齐 mimoagent 保留尾部的思路）。"""
    if max_chars <= 0 or text is None:
        return text or '', False
    if len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.4)
    tail = max_chars - head
    marker = f'\n\n... [TRUNCATED {len(text) - max_chars} chars] ...\n\n'
    return text[:head] + marker + text[-tail:], True


class SandboxBackend(ABC):
    """沙箱后端接口。一个实例对应一个任务环境。"""

    name: str = 'base'

    def __init__(self, spec: TaskSpec, **opts):
        self.spec = spec
        self.opts = opts
        self._started = False

    # ---------------- 生命周期 ----------------
    @abstractmethod
    async def start(self) -> None:
        """拉起/准备环境（拉镜像、起容器、建临时目录等）。"""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """销毁环境，用完即毁。必须幂等、吞异常。"""
        raise NotImplementedError

    # ---------------- 命令与文件 ----------------
    @abstractmethod
    async def exec(self, command: str, timeout: Optional[float] = None,
                   cwd: Optional[str] = None) -> ExecResult:
        """在环境内执行 shell（等价 ``timeout N bash -lc <command>``）。"""
        raise NotImplementedError

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """向环境内写入文本文件。"""
        raise NotImplementedError

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """读取环境内文本文件；不存在时抛出 FileNotFoundError。"""
        raise NotImplementedError

    async def reset_files(self, relpaths: List[str]) -> bool:
        """打分前把指定文件还原到基线（code 域：git reset 受 test_patch 影响的文件）。

        默认实现返回 False（后端不支持）；Docker/Local 后端在是 git 仓库时实现。
        """
        return False

    # ---------------- async context manager ----------------
    async def __aenter__(self) -> 'SandboxBackend':
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()


# ----------------------------------------------------------------------
# 注册表 + 工厂
# ----------------------------------------------------------------------
BACKENDS: Dict[str, type] = {}


def register_backend(name: str):
    def _wrap(cls):
        BACKENDS[name] = cls
        cls.name = name
        return cls
    return _wrap


def get_backend(spec: TaskSpec, **opts) -> SandboxBackend:
    """按 TaskSpec.backend 选择后端；惰性导入具体实现以避免缺依赖时整体不可用。"""
    name = spec.backend or 'docker'
    if name not in BACKENDS:
        # 惰性注册内置后端
        if name == 'docker':
            from sandbox.docker_backend import DockerBackend  # noqa: F401
        elif name == 'local':
            from sandbox.local_backend import LocalBackend  # noqa: F401
        elif name == 'remote':
            from sandbox.remote_backend import RemoteBackend  # noqa: F401
        else:
            raise ValueError(f'Unknown sandbox backend: {name}. '
                             f'Available: {sorted(set(BACKENDS) | {"docker", "local", "remote"})}')
    cls = BACKENDS.get(name)
    if cls is None:
        raise RuntimeError(f'backend {name} registered but class missing')
    return cls(spec, **opts)
