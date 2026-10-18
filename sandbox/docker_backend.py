"""Docker 沙箱后端：一任务一隔离容器（对齐 mimoagent ``environments/docker.py``）。

事实要点（对齐小米，见 research/02_xiaomi_verl_flow.md §3.2）：
- 每个任务一个容器，命令经 ``docker exec``（等价 ``timeout N bash -lc``），用完即毁。
- 镜像来自 ``TaskSpec.docker_image``（每行数据自带，非全局）；cwd 来自 ``TaskSpec.cwd``。
- 文件经 tar 流（``docker exec tar``）传递；git 仓库打分前 ``git checkout --`` 还原。

安全：
- 所有 docker 调用一律用 **argv 列表 + shell=False**（``create_subprocess_exec``），
  杜绝宿主机侧命令注入；绝不直接在宿主机 exec 模型生成的代码。
- 默认 ``--network=none`` 断网，仅当 ``spec.network=True`` 才放开网络。
- 资源隔离：``--memory / --cpus / --pids-limit``（可由 opts 或环境变量覆盖）。

参考：
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/docker.py
"""
from __future__ import annotations

# ---- 项目根 sys.path 兜底（可从任意目录 python -m 运行）----
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import asyncio
import io
import re
import time
import tarfile
import uuid
from typing import List, Optional, Tuple

from sandbox.base import (
    SandboxBackend, ExecResult, register_backend, truncate_output,
)
from swift_agent.spec import TaskSpec


def _sanitize_container_name(s: str) -> str:
    """docker 容器名只允许 [a-zA-Z0-9][a-zA-Z0-9_.-]。"""
    s = re.sub(r'[^a-zA-Z0-9_.-]', '-', s or '')
    s = s.strip('.-')
    return (s or 'task')[:40]


@register_backend('docker')
class DockerBackend(SandboxBackend):
    name = 'docker'

    def __init__(self, spec: TaskSpec, **opts):
        super().__init__(spec, **opts)
        self.container_name: Optional[str] = None
        # 资源限制：任务级 spec.cpus/memory_mb 优先，其次 opts，再次环境变量，最后安全默认值
        self.memory = (
            opts.get('memory')
            or (f'{int(spec.memory_mb)}m' if spec.memory_mb else None)
            or _os.environ.get('SANDBOX_DOCKER_MEMORY', '2g')
        )
        self.cpus = str(
            opts.get('cpus')
            or spec.cpus
            or _os.environ.get('SANDBOX_DOCKER_CPUS', '2.0')
        )
        self.pids_limit = str(opts.get('pids_limit') or _os.environ.get('SANDBOX_DOCKER_PIDS_LIMIT', '256'))
        self._default_timeout = float(spec.tool_timeout_sec or 30.0)
        # 容器自毁 TTL：env SANDBOX_TTL_SEC（缺省 7200）+ verifier_timeout_sec 余量，opts 可覆盖。
        # 即便 stop() 永不被调用（驱动器 max_turns 强杀、未走 close），容器也会在 TTL 后
        # 自动退出；配合 --rm 即自动删除，杜绝容器泄漏。
        self.ttl_sec = self._compute_ttl(opts)

    def _resolve_image(self) -> str:
        """镜像解析优先序：dockerhub_image（image-mapping.jsonl 解析的全名）
        > docker_image（parquet 原始名，用户可能已 retag）。
        两者都非 xiaomimimo/ 前缀且本地不存在时，_ensure_image 会尝试按
        xiaomimimo/mimo-v2.6-rl-oss:<tag> 猜测并 pull（覆盖用户手工 retag 场景）。
        """
        img = (getattr(self.spec, 'dockerhub_image', '') or '').strip()
        if img:
            return img
        return (self.spec.docker_image or '').strip()

    async def _ensure_image(self, image: str) -> None:
        """镜像本地不存在时自动 docker pull（仅对 xiaomimimo/ 官方仓可靠；
        其它名字 pull 失败不吞，由 start() 报错）。"""
        out, err, rc = await self._run(
            ['docker', 'image', 'inspect', image], wall_timeout=30.0)
        if rc == 0:
            return
        pull_argv = ['docker', 'pull', image]
        out, err, rc = await self._run(pull_argv, wall_timeout=self._pull_timeout)
        if rc != 0:
            raise RuntimeError(
                f'docker pull failed for {image} (rc={rc}): {err.strip()[:500]}. '
                f'若为任务镜像，请确认 data/raw/image-mapping.jsonl 已生成 env_config.dockerhub_image，'
                f'或手工 docker pull xiaomimimo/mimo-v2.6-rl-oss:<tag> 后 docker tag。'
            )

    @property
    def _pull_timeout(self) -> float:
        return float(_os.environ.get('SANDBOX_DOCKER_PULL_TIMEOUT', '1800'))

    def _compute_ttl(self, opts) -> int:
        # opts.ttl_sec 为显式覆盖：视为最终 TTL（不再叠加余量，仅保底 60s）
        if opts.get('ttl_sec') is not None:
            return max(60, int(float(opts['ttl_sec'])))
        # 默认：env SANDBOX_TTL_SEC（缺省 7200）+ verifier_timeout_sec 余量
        base = float(_os.environ.get('SANDBOX_TTL_SEC', '7200'))
        margin = float(getattr(self.spec, 'verifier_timeout_sec', 0.0) or 0.0)
        return max(60, int(base + margin))

    # ------------------------------------------------------------------
    # 底层：统一执行一条宿主机侧 argv（shell=False），返回原始字节
    # ------------------------------------------------------------------
    async def _run(self, argv: List[str], input_bytes: Optional[bytes] = None,
                   wall_timeout: Optional[float] = None
                   ) -> Tuple[Optional[bytes], str, int]:
        """运行宿主机命令（docker CLI）。

        返回 (stdout_bytes, stderr_text, returncode)。
        - docker CLI 缺失/daemon 不可达/外层墙钟超时 -> returncode 约定 -1（transport）。
        """
        stdin = asyncio.subprocess.PIPE if input_bytes is not None else None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=stdin,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return None, f'docker CLI not found on host: {e}', -1
        try:
            outb, errb = await asyncio.wait_for(
                proc.communicate(input=input_bytes), timeout=wall_timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            await proc.wait()
            return None, f'docker cli wall-timeout after {wall_timeout}s', -2
        return outb, errb.decode('utf-8', errors='replace'), proc.returncode

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        # 1) 检测 docker daemon 可用
        out, err, rc = await self._run(['docker', 'version', '--format', '{{.Server.Version}}'],
                                        wall_timeout=15.0)
        if rc != 0:
            raise RuntimeError(
                f'docker daemon unavailable (rc={rc}): {err.strip() or "is docker installed/running?"}'
            )
        if not self.spec.docker_image and not getattr(self.spec, 'dockerhub_image', ''):
            raise RuntimeError('TaskSpec.docker_image is empty; cannot start container.')

        image = self._resolve_image()
        await self._ensure_image(image)

        name = f"mimo-{_sanitize_container_name(self.spec.instance_id)}-{uuid.uuid4().hex[:8]}"
        argv: List[str] = [
            'docker', 'run', '-d',
            '--rm',                       # 容器一旦退出即自动删除（兜底防泄漏）
            '--name', name,
            # 网络隔离：默认断网，仅 spec.network=True 时放开（用默认 bridge）
        ]
        if not self.spec.network:
            argv += ['--network', 'none']
        argv += [
            '--memory', self.memory,
            '--cpus', self.cpus,
            '--pids-limit', self.pids_limit,
        ]
        if self.spec.cwd:
            argv += ['-w', self.spec.cwd]
        # 带 TTL 的自毁入口：TTL 后向自身发 TERM 退出；配合 --rm 自动删除。
        # 以 argv 列表传容器内 bash -lc，TTL 为受控整数，无宿主机注入面。
        ttl_cmd = (
            f"trap 'exit 0' TERM INT; ( sleep {self.ttl_sec}; kill -TERM $$ ) & wait"
        )
        argv += [image, 'bash', '-lc', ttl_cmd]

        out, err, rc = await self._run(argv, wall_timeout=300.0)
        if rc != 0:
            raise RuntimeError(f'docker run failed (rc={rc}): {err.strip()}')
        self.container_name = name
        self._started = True

    async def stop(self) -> None:
        # 幂等、吞异常：docker rm -f，容器不存在也不报错
        if self.container_name:
            try:
                await self._run(['docker', 'rm', '-f', self.container_name], wall_timeout=30.0)
            except Exception:
                pass
            self.container_name = None
        self._started = False

    # ------------------------------------------------------------------
    # 命令执行
    # ------------------------------------------------------------------
    async def exec(self, command: str, timeout: Optional[float] = None,
                   cwd: Optional[str] = None) -> ExecResult:
        if not self._started or not self.container_name:
            return ExecResult(output='sandbox container not started', returncode=-1,
                              reason='backend_error')
        eff_cwd = cwd or self.spec.cwd or '.'
        t = float(timeout) if timeout is not None else self._default_timeout
        t_int = max(1, int(t))
        # 容器内用 coreutils timeout 包裹：超时 -> returncode 124
        argv = [
            'docker', 'exec', '-w', eff_cwd, self.container_name,
            'timeout', str(t_int), 'bash', '-lc', command,
        ]
        outb, err, rc = await self._run(argv, wall_timeout=t + 60.0)

        if rc in (-1, -2):  # docker CLI 缺失 / 墙钟超时 -> 传输层故障
            return ExecResult(output=(outb or b'').decode('utf-8', 'replace') + err,
                              returncode=125, reason='transport_error')
        # docker exec 自身 rc=125 表示 docker daemon/容器问题（非命令本身失败）
        if rc == 125:
            return ExecResult(output=err, returncode=rc, reason='transport_error')
        output = (outb or b'').decode('utf-8', errors='replace')
        if err:
            output = output + err if output.endswith('\n') else output + '\n' + err
        reason = 'timeout' if rc == 124 else 'ok'
        return ExecResult(output=output, returncode=rc, reason=reason)

    # ------------------------------------------------------------------
    # 文件：tar 流经 docker exec
    # ------------------------------------------------------------------
    def _tar_root_and_member(self, path: str) -> Tuple[str, str]:
        """把一个容器内路径拆成 (tar -C 根目录, tar 成员名)。"""
        if _os.path.isabs(path):
            return '/', path.lstrip('/')
        return (self.spec.cwd or '.'), path

    async def write_file(self, path: str, content: str) -> None:
        if not self.container_name:
            raise RuntimeError('container not started')
        root, member = self._tar_root_and_member(path)
        data = content.encode('utf-8')
        buf = io.BytesIO()
        info = tarfile.TarInfo(name=member)
        info.size = len(data)
        info.mtime = time.time()
        info.mode = 0o644
        with tarfile.open(fileobj=buf, mode='w') as tf:
            tf.addfile(info, io.BytesIO(data))
        argv = ['docker', 'exec', '-i', self.container_name, 'tar', 'xmf', '-', '-C', root]
        out, err, rc = await self._run(argv, input_bytes=buf.getvalue(), wall_timeout=60.0)
        if rc != 0:
            raise IOError(f'docker tar extract failed (rc={rc}): {err.strip()}')

    async def read_file(self, path: str) -> str:
        if not self.container_name:
            raise RuntimeError('container not started')
        root, member = self._tar_root_and_member(path)
        argv = ['docker', 'exec', self.container_name, 'tar', 'cf', '-', '-C', root, member]
        outb, err, rc = await self._run(argv, wall_timeout=60.0)
        if rc != 0:
            raise FileNotFoundError(f'{path}: {err.strip() or "not found in container"}')
        # 从 tar 流中取出该成员文本
        with tarfile.open(fileobj=io.BytesIO(outb or b''), mode='r:') as tf:
            for m in tf.getmembers():
                if m.isfile() and (m.name == member or m.name.endswith('/' + member)):
                    f = tf.extractfile(m)
                    if f is None:
                        break
                    return f.read().decode('utf-8', errors='replace')
        raise FileNotFoundError(f'{path}: not found in tar stream')

    async def upload_dir(self, local_dir: str, remote_dir: str) -> int:
        """把本地目录整树 tar 进容器 remote_dir（general 域 env_task_dir 上传）。

        返回写入的文件数；空目录/异常抛错。逐文件上传的上限问题：
        上游实测 exec 流 >33MB 会断（kubernetes.py），docker exec tar 整目录
        同样受限，故大目录建议分批（这里先整体 tar，失败由调用方降级）。
        """
        if not self.container_name:
            raise RuntimeError('container not started')
        local_dir = _os.path.abspath(local_dir)
        if not _os.path.isdir(local_dir):
            raise FileNotFoundError(f'local dir not found: {local_dir}')
        n = 0
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w') as tf:
            for root, _dirs, files in _os.walk(local_dir):
                for fname in files:
                    fpath = _os.path.join(root, fname)
                    arcname = _os.path.relpath(fpath, local_dir)
                    tf.add(fpath, arcname=arcname)
                    n += 1
        if n == 0:
            return 0
        return await self.upload_tarball(buf.getvalue(), remote_dir)

    async def upload_tarball(self, tar_bytes: bytes, remote_dir: str) -> int:
        """tar 字节流直接经 docker exec tar xmf - 解包进容器（不在宿主机落盘）。"""
        if not self.container_name:
            raise RuntimeError('container not started')
        out, err, rc = await self._run(
            ['docker', 'exec', self.container_name, 'mkdir', '-p', remote_dir],
            wall_timeout=30.0,
        )
        if rc != 0:
            raise IOError(f'mkdir {remote_dir} failed: {err.strip()[:300]}')
        argv = ['docker', 'exec', '-i', self.container_name, 'tar', 'xmf', '-', '-C', remote_dir]
        out, err, rc = await self._run(argv, input_bytes=tar_bytes, wall_timeout=300.0)
        if rc != 0:
            raise IOError(f'docker tar extract failed (rc={rc}): {err.strip()[:300]}')
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:') as tf:
            return sum(1 for m in tf.getmembers() if m.isfile())

    # ------------------------------------------------------------------
    # 打分前还原受测文件
    # ------------------------------------------------------------------
    async def reset_files(self, relpaths: List[str]) -> bool:
        if not relpaths or not self.container_name:
            return False
        cwd = self.spec.cwd or '.'
        _, _, rc = await self._run(
            ['docker', 'exec', '-w', cwd, self.container_name,
             'git', 'rev-parse', '--is-inside-work-tree'], wall_timeout=15.0,
        )
        if rc != 0:
            return False  # 不是 git 仓库
        argv = ['docker', 'exec', '-w', cwd, self.container_name,
                'git', 'checkout', '--', *relpaths]
        _, err, rc = await self._run(argv, wall_timeout=30.0)
        return rc == 0
