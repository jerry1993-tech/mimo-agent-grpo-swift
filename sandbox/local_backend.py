"""Local 沙箱后端：在本机临时目录跑命令——【无任何隔离】，仅供闭环测试/低风险任务。

安全声明（必须显著可见）：
- 本后端直接在宿主机 ``cwd=self.root`` 下 exec 模型生成的命令，**没有容器/虚拟化隔离**。
- 严禁在不可信模型或真实训练/生产中使用；仅用于本机 smoke test 与离线调试。
- 仍做了路径穿越防护：read/write 解析后的 realpath 必须落在临时 root 之内。
- 命令一律 ``create_subprocess_exec('bash','-lc',cmd, cwd=root)``（argv 列表、shell=False），
  外层用 ``asyncio.wait_for`` 实现超时；超时按约定合成 returncode=124 / reason='timeout'。

对齐 base.py：一实例一临时目录，stop() 即 rmtree，用完即毁。
"""
from __future__ import annotations

# ---- 项目根 sys.path 兜底 ----
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import asyncio
import atexit
import os
import shutil
import tempfile
from typing import List, Optional

from sandbox.base import SandboxBackend, ExecResult, register_backend
from swift_agent.spec import TaskSpec

_WARNING = (
    '\n'
    '====================================================================\n'
    '  [LocalBackend] 无隔离沙箱！命令直接在宿主机临时目录执行。\n'
    '  仅供本机闭环测试 / 低风险任务；禁止用于不可信模型或生产训练。\n'
    '====================================================================\n'
)


@register_backend('local')
class LocalBackend(SandboxBackend):
    name = 'local'

    def __init__(self, spec: TaskSpec, **opts):
        super().__init__(spec, **opts)
        self.root: Optional[str] = None
        self._default_timeout = float(spec.tool_timeout_sec or 30.0)

    # ------------------------------------------------------------------
    def _safe_abspath(self, path: str) -> str:
        """把用户给的路径限制在 self.root 内，防路径穿越；越界抛 ValueError。

        绝对路径（容器语义，如 /work/workspace）在 local 后端映射为
        root 下的同名相对路径（/work/workspace -> root/work/workspace），
        与 write_file 的 tar-root 语义一致；root 自身除外。
        """
        if self.root is None:
            raise RuntimeError('local backend not started')
        if os.path.isabs(path):
            stripped = path.replace('\\', '/').lstrip('/')
            if not stripped:  # '/' -> root 自身
                cand = self.root
            else:
                cand = os.path.normpath(os.path.join(self.root, stripped))
        else:
            cand = os.path.normpath(os.path.join(self.root, path))
        real_root = os.path.realpath(self.root)
        real_cand = os.path.realpath(cand)
        # real_cand 必须在 real_root 之下（含等于 root）
        if real_cand != real_root and not real_cand.startswith(real_root + os.sep):
            raise ValueError(f'path escapes sandbox root: {path!r} -> {real_cand}')
        return real_cand

    # ------------------------------------------------------------------
    async def start(self) -> None:
        # 显著警告：打印到 stderr，避免被日志吞掉
        print(_WARNING, flush=True)
        self.root = tempfile.mkdtemp(prefix='mimo-local-')
        # 兜底回收：即便驱动器 max_turns 强杀、未调用 stop()/close()，
        # 进程退出时 atexit 也会删掉临时目录，防泄漏。
        atexit.register(shutil.rmtree, self.root, ignore_errors=True)
        # 若 spec.cwd 指定了子目录，建好（相对 root）
        if self.spec.cwd:
            sub = self._safe_abspath(self.spec.cwd)
            os.makedirs(sub, exist_ok=True)
        self._started = True

    def __del__(self):
        # best-effort 兜底：对象被 GC 时也尝试清理（atexit 已覆盖主路径）
        try:
            if getattr(self, 'root', None) and os.path.isdir(self.root):
                shutil.rmtree(self.root, ignore_errors=True)
        except Exception:
            pass

    async def stop(self) -> None:
        if self.root and os.path.isdir(self.root):
            shutil.rmtree(self.root, ignore_errors=True)
        self.root = None
        self._started = False

    # ------------------------------------------------------------------
    # local 后端模拟容器文件系统时，exec 命令里的容器绝对路径前缀改写到临时 root。
    # 只重写已知的官方挂载点（manifest.json 的 target 均在这些前缀下），避免误伤。
    _CONTAINER_PREFIXES = ('/work/', '/installed-agent/', '/logs/', '/tmp/agent_output/')

    def _rewrite_container_paths(self, command: str) -> str:
        """把命令中出现的容器绝对路径前缀替换为 root 下的实际位置。"""
        if not self.root:
            return command
        out = command
        for p in self._CONTAINER_PREFIXES:
            # ' /work/x' -> ' <root>/work/x'（词首匹配，避免匹配子串）
            out = out.replace(f' {p}', f' {self.root}/{p}')
            out = out.replace(f'={p}', f'={self.root}/{p}')
            out = out.replace(f'({p}', f'({self.root}/{p}')
        return out

    async def exec(self, command: str, timeout: Optional[float] = None,
                   cwd: Optional[str] = None) -> ExecResult:
        if not self._started or not self.root:
            return ExecResult(output='local backend not started', returncode=-1,
                              reason='backend_error')
        eff_cwd = self.root
        if cwd:
            eff_cwd = self._safe_abspath(cwd)
            os.makedirs(eff_cwd, exist_ok=True)
        command = self._rewrite_container_paths(command)
        t = float(timeout) if timeout is not None else self._default_timeout
        try:
            proc = await asyncio.create_subprocess_exec(
                'bash', '-lc', command,
                cwd=eff_cwd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # 便于超时整组杀进程
            )
        except FileNotFoundError as e:
            return ExecResult(output=f'bash not found: {e}', returncode=127,
                              reason='transport_error')
        try:
            outb, errb = await asyncio.wait_for(proc.communicate(), timeout=t)
        except asyncio.TimeoutError:
            # 整组杀掉，回收
            try:
                os.killpg(proc.getpid(), 9)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            await proc.wait()
            tail = errb.decode('utf-8', 'replace') if (locals().get('errb') is not None) else ''
            return ExecResult(output=(tail + f'\n[timed out after {t}s]').strip(),
                              returncode=124, reason='timeout')
        out = outb.decode('utf-8', errors='replace')
        err = errb.decode('utf-8', errors='replace')
        output = out + err if out.endswith('\n') or not err else out + '\n' + err
        return ExecResult(output=output, returncode=proc.returncode, reason='ok')

    # ------------------------------------------------------------------
    async def write_file(self, path: str, content: str) -> None:
        abspath = self._safe_abspath(path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, 'w', encoding='utf-8') as f:
            f.write(content)

    async def read_file(self, path: str) -> str:
        abspath = self._safe_abspath(path)
        if not os.path.isfile(abspath):
            raise FileNotFoundError(path)
        with open(abspath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    async def upload_dir(self, local_dir: str, remote_dir: str) -> int:
        """把本地目录整树拷进临时 root 下的 remote_dir（仅测试用）。"""
        target = self._safe_abspath(remote_dir)
        os.makedirs(target, exist_ok=True)
        n = 0
        for root, _dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            dst_dir = target if rel == '.' else os.path.join(target, rel)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in files:
                shutil.copy2(os.path.join(root, fname), os.path.join(dst_dir, fname))
                n += 1
        return n

    async def upload_tarball(self, tar_bytes: bytes, remote_dir: str) -> int:
        """tar 字节流解包到临时 root 下的 remote_dir（远程上传的服务端落地）。"""
        import io
        import tarfile
        target = self._safe_abspath(remote_dir)
        os.makedirs(target, exist_ok=True)
        n = 0
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:') as tf:
            for m in tf.getmembers():
                if not (m.isfile() or m.isdir()):
                    continue
                # 归一化成员路径，防 tar 内 ../ 逃逸（tarfile extract 手工逐成员）
                dest = os.path.normpath(os.path.join(target, m.name))
                real_target = os.path.realpath(target)
                if not (dest == real_target or dest.startswith(real_target + os.sep)):
                    raise ValueError(f'tar member escapes sandbox root: {m.name!r}')
                if m.isdir():
                    os.makedirs(dest, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, 'wb') as f:
                        f.write(tf.extractfile(m).read())
                    n += 1
        return n

    # ------------------------------------------------------------------
    async def reset_files(self, relpaths: List[str]) -> bool:
        if not relpaths or not self.root:
            return False
        # 是否 git 仓库
        git_dir = os.path.join(self.root, '.git')
        if not os.path.isdir(git_dir):
            return False
        argv = ['git', 'checkout', '--', *relpaths]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=self.root,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            return proc.returncode == 0
        except Exception:
            return False
