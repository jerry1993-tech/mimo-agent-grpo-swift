"""general 域环境装配：env_task_dir（本地 HF general/envs/<task>/）-> 容器内可用。

对齐上游（verl mimo-oss recipes/general/env_actor.py + mimoagent）：
- ``manifest.json`` 声明 uploads（source -> target，container: main/sidecar）、
  setup（sidecar 启动 MCP 服务）、wait_ports、mcp_servers（http://127.0.0.1:<port>/mcp）、
  verifier（verify 脚本上传 + 运行命令 + reward 文件路径）；
- 上游是双容器（main agent + sidecar MCP/verify，iptables 隔离）；本工程 DockerBackend
  是单容器，采用**单容器近似**：sidecar 的文件也落到同一容器、MCP 服务在容器内
  127.0.0.1 起、verify 在容器内跑。安全边界 = 容器边界（上游多一层 uid 防火墙）。

流程（env.reset 时调用 setup_general_env）：
1. 定位本地 env_task_dir（data/raw/general/envs/<task>/，download_general_envs.py 产物）；
2. 读 manifest.json，把 uploads 全部落进容器（tar 目录 / base64 文件）；
3. 跑 setup.command（等价 sidecar 启动），轮询 wait_ports 就绪；
4. 读 verifier_meta.json -> spec.rubrics（供 GeneralVerifier LLM judge 用）；
5. 把 mcp_servers 记到 spec.extra['mcp_servers']（mcp_client 据此装配工具）。

缺文件/失败 -> 抛 GeneralEnvError；mimo_env.reset 会把它转成 infra（组内剔除）。
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import posixpath
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any, Dict, List, Optional  # noqa: E402

from swift_agent.spec import TaskSpec  # noqa: E402


class GeneralEnvError(RuntimeError):
    """general 域环境装配失败（应按 is_infra 处理，不当模型 0 分）。"""


# env_task_dir（parquet 值，如 "envs/s3k_xxx"）-> 本地目录的搜索根
ENV_ROOT = os.path.join(_ROOT, 'data', 'raw', 'general')


def find_env_task_dir(spec: TaskSpec) -> Optional[str]:
    """定位 env_task_dir 的本地路径；找不到返回 None。

    parquet 里 env_task_dir 形如 ``envs/s3k_xxx``（相对 general/），
    本地布局为 ``data/raw/general/envs/s3k_xxx/``。
    """
    d = (spec.env_task_dir or '').strip()
    if not d:
        return None
    d = os.path.normpath(d).replace('\\', '/')
    candidates = [os.path.join(ENV_ROOT, d)]
    if d.startswith('envs/'):
        candidates.append(os.path.join(ENV_ROOT, d))  # 同上；normpath 已处理
    else:
        candidates.append(os.path.join(ENV_ROOT, 'envs', d))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None


async def _upload_binary(backend, src: str, dst: str, timeout: float = 300.0) -> None:
    """二进制安全上传：base64 写临时文件再解码（分块 append）。"""
    with open(src, 'rb') as f:
        content = f.read()
    d = posixpath.dirname(dst)
    if d:
        res = await backend.exec(f'mkdir -p {d}', timeout=30.0, cwd='/')
        if res.returncode != 0:
            raise GeneralEnvError(f'mkdir {d} failed: {res.output[:200]}')
    b64 = base64.b64encode(content).decode('ascii')
    tmp = dst + '.b64'
    res = await backend.exec(f': > {tmp}', timeout=30.0, cwd='/')
    if res.returncode != 0:
        raise GeneralEnvError(f'truncate {tmp} failed: {res.output[:200]}')
    chunk = 1_400_000  # ~1MB 源数据/块
    for i in range(0, len(b64), chunk):
        part = b64[i:i + chunk]
        res = await backend.exec(f'printf %s {part} >> {tmp}', timeout=timeout, cwd='/')
        if res.returncode != 0:
            raise GeneralEnvError(f'upload {src} -> {dst} chunk {i} failed: {res.output[:200]}')
    res = await backend.exec(f'base64 -d {tmp} > {dst} && rm -f {tmp}', timeout=60.0, cwd='/')
    if res.returncode != 0:
        raise GeneralEnvError(f'base64 decode {dst} failed: {res.output[:200]}')
    # 大小校验
    expect = len(content)
    res = await backend.exec(f'stat -c %s {dst}', timeout=30.0, cwd='/')
    got = (res.output or '').strip().splitlines()[-1] if res.output else ''
    if got.isdigit() and int(got) != expect:
        raise GeneralEnvError(f'size mismatch {dst}: {got} != {expect}')


async def _put_file(backend, src: str, dst: str, timeout: float = 300.0) -> None:
    """按内容类型分派：文本 write_file / 二进制 _upload_binary。"""
    try:
        with open(src, 'rb') as f:
            head = f.read(64_000)
        head.decode('utf-8')
        is_text = True
    except (UnicodeDecodeError, OSError):
        is_text = False
    if is_text:
        d = posixpath.dirname(dst)
        if d:
            res = await backend.exec(f'mkdir -p {d}', timeout=30.0, cwd='/')
            if res.returncode != 0:
                raise GeneralEnvError(f'mkdir {d} failed: {res.output[:200]}')
        with open(src, 'r', encoding='utf-8') as f:
            await backend.write_file(dst, f.read())
    else:
        await _upload_binary(backend, src, dst, timeout=timeout)


async def setup_general_env(spec: TaskSpec, backend) -> Dict[str, Any]:
    """把 env_task_dir 装配进容器。返回 {'mcp_servers': [...], 'rubrics': {...}}。"""
    local_dir = find_env_task_dir(spec)
    if local_dir is None:
        raise GeneralEnvError(
            f'env_task_dir {spec.env_task_dir!r} not found locally; '
            f'run: python -m data.download_general_envs --tasks <task_id>')

    manifest_path = os.path.join(local_dir, 'manifest.json')
    if not os.path.isfile(manifest_path):
        raise GeneralEnvError(f'manifest.json missing under {local_dir}')
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    # 1) uploads：单容器近似，main/sidecar 都落同一容器
    uploads: List[Dict[str, Any]] = manifest.get('uploads') or []
    for up in uploads:
        src = os.path.join(local_dir, str(up.get('source', '')))
        dst = str(up.get('target', ''))
        if not src or not dst or not os.path.exists(src):
            continue  # 该任务没有此文件（如无 mcp_bridge）
        if os.path.isdir(src):
            await backend.upload_dir(src, dst)
        else:
            await _put_file(backend, src, dst)

    # 2) setup：启动 MCP 服务（等价 sidecar entrypoint），轮询端口。
    #    仅在隔离后端（docker/remote）下执行：官方 sidecar 依赖镜像内的
    #    /opt/openai-agents-venv 与 /work 目录布局，local 后端无容器文件系统，
    #    setup 必然失败——降级为跳过（rubrics 仍装配，MCP 工具不可用，
    #    verify 走 LLM judge 兜底），不视为装配失败。
    backend_name = getattr(backend, 'name', '')
    setup = manifest.get('setup') or {}
    setup_cmd = setup.get('command')
    if setup_cmd and backend_name not in ('local',):
        timeout = float(setup.get('timeout_sec') or 300)
        res = await backend.exec(setup_cmd, timeout=timeout, cwd='/')
        if res.returncode != 0:
            raise GeneralEnvError(
                f'setup command failed (rc={res.returncode}): {res.output[:300]}')
        for port in manifest.get('wait_ports') or []:
            if not await _wait_port(backend, int(port), timeout=60.0):
                raise GeneralEnvError(f'MCP port {port} not ready after setup')
    elif setup_cmd and backend_name == 'local':
        # local：跳过 MCP 启动（容器内依赖缺失），mcp_servers 仍写入 spec.extra
        # 供上游感知，但工具调用会失败 -> 提示走 bash 兜底。
        pass

    # 3) verifier_meta.json -> rubrics
    out: Dict[str, Any] = {}
    vmeta_path = os.path.join(local_dir, 'verifier_meta.json')
    if os.path.isfile(vmeta_path):
        try:
            with open(vmeta_path, encoding='utf-8') as f:
                out['rubrics'] = json.load(f)
        except json.JSONDecodeError as e:
            raise GeneralEnvError(f'verifier_meta.json corrupt: {e}')

    # 4) mcp_servers -> spec.extra（mcp_client 据此装配工具）
    mcp_servers = manifest.get('mcp_servers') or []
    if mcp_servers:
        spec.extra['mcp_servers'] = {
            str(s.get('name', f'mcp_{i}')): str(s.get('url', ''))
            for i, s in enumerate(mcp_servers) if s.get('url')
        }
        out['mcp_servers'] = mcp_servers
    return out


async def _wait_port(backend, port: int, timeout: float = 60.0) -> bool:
    """轮询容器内端口就绪（bash /dev/tcp 探测）。"""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    probe = f'(echo > /dev/tcp/127.0.0.1/{port}) >/dev/null 2>&1 && echo READY'
    while loop.time() < deadline:
        res = await backend.exec(probe, timeout=5.0, cwd='/')
        if 'READY' in (res.output or ''):
            return True
        await asyncio.sleep(2.0)
    return False
