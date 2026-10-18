"""general 域 env_task_dir 装配与打分链路测试（local 后端 + mock manifest）。

覆盖：
- find_env_task_dir 定位（envs/ 前缀两种写法、不存在返回 None）；
- setup_general_env：uploads（目录+文件）落容器、setup 命令、wait_ports、
  rubrics/mcp_servers 装配（用真实 manifest.json 结构 mock 一个最小任务）；
- GeneralVerifier._official_verify：写 agent_output -> 跑官方脚本 -> 读 reward.json
  （mock 一个 manifest 风格的 verify 命令，产出 reward.json）。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox import general_env_setup as ges
from sandbox.general_env_setup import GeneralEnvError, setup_general_env
from sandbox.local_backend import LocalBackend  # noqa: F401  (registers 'local')
from swift_agent.spec import TaskSpec


def _write(p: str, content: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)


@pytest.fixture
def env_task(tmp_path, monkeypatch):
    """mock 一个 general 任务目录（结构对齐真实 s3k 任务）。"""
    task_dir = tmp_path / 'general' / 'envs' / 's3k_test_task'
    task_dir.mkdir(parents=True)
    # workspace（目录上传 -> /work/workspace）
    _write(str(task_dir / 'workspace' / 'note.txt'), 'workspace file')
    # system（目录上传 -> /work/system）
    _write(str(task_dir / 'system' / 'crm_db' / 'schema.sql'), 'CREATE TABLE t(x);')
    # 单文件上传
    _write(str(task_dir / 'mcp_bridge.py'), 'print("bridge")')
    # verifier 脚本（写 reward.json 的最小实现）。
    # 注意：local 后端无容器文件系统，脚本内用相对路径（cwd 语义），
    # manifest 的 reward_file 也配成相对路径；docker 后端下真实任务用绝对路径。
    _write(str(task_dir / 'run_verify.py'), (
        'import json, os\n'
        'os.makedirs("logs/verifier", exist_ok=True)\n'
        'with open("logs/verifier/reward.json", "w") as f:\n'
        '    json.dump({"reward": 0.75}, f)\n'
        'print("[verify] score=0.750")\n'
    ))
    _write(str(task_dir / 'verifier_meta.json'), json.dumps({
        'items': [
            {'id': 'r1', 'tier': 'critical', 'method': 'llm', 'weight': 0.5,
             'files': ['answer.md'],
             'question': 'Is the answer correct?', 'pass_anchor': 'Yes.'},
            {'id': 'r2', 'tier': 'normal', 'method': 'llm', 'weight': 0.5,
             'files': ['answer.md'],
             'question': 'Is it complete?', 'pass_anchor': 'Yes.'},
        ],
    }))
    _write(str(task_dir / 'manifest.json'), json.dumps({
        'cwd': '/work/workspace',
        'uploads': [
            {'source': 'workspace', 'target': '/work/workspace', 'container': 'main'},
            {'source': 'system', 'target': '/work/system', 'container': 'sidecar'},
            {'source': 'mcp_bridge.py', 'target': '/work/_setup/mcp_bridge.py', 'container': 'main'},
        ],
        'setup': None,          # 测试不起 MCP 服务
        'wait_ports': [],
        'mcp_servers': [],
        'verifier': {
            'uploads': [
                {'source': 'run_verify.py', 'target': '/work/run_verify.py'},
                {'source': 'verifier_meta.json', 'target': '/work/verifier_meta.json'},
            ],
            'command': 'python3 /work/run_verify.py',
            'reward_file': '/logs/verifier/reward.json',
            'reward_detail_file': '/logs/verifier/reward_detail.json',
            'timeout_sec': 60,
        },
    }))
    # 把搜索根指到 tmp_path（ENV_ROOT 是模块级常量，monkeypatch）
    monkeypatch.setattr(ges, 'ENV_ROOT', str(tmp_path / 'general'))
    return task_dir


def general_spec(env_task_dir='envs/s3k_test_task') -> TaskSpec:
    return TaskSpec(
        instance_id='s3k_test_task', dataset_type='general_agent', domain='general',
        backend='local', cwd='/work/workspace',
        env_task_dir=env_task_dir,
        problem_statement='Do the knowledge-work task.',
        verifier_timeout_sec=60,
    )


# ----------------------------------------------------------------------
def test_find_env_task_dir(tmp_path, monkeypatch, env_task):
    monkeypatch.setattr(ges, 'ENV_ROOT', str(tmp_path / 'general'))
    spec = general_spec('envs/s3k_test_task')
    assert ges.find_env_task_dir(spec) == str(env_task)
    # 不带 envs/ 前缀也能找到
    assert ges.find_env_task_dir(general_spec('s3k_test_task')) == str(env_task)
    # 不存在 -> None
    assert ges.find_env_task_dir(general_spec('envs/s3k_missing')) is None
    # 空值 -> None
    assert ges.find_env_task_dir(TaskSpec(domain='general')) is None


@pytest.mark.asyncio
async def test_setup_general_env_uploads(env_task):
    spec = general_spec()
    backend = LocalBackend(spec)
    await backend.start()
    try:
        info = await setup_general_env(spec, backend)
        # uploads 落盘校验：local 后端把容器绝对路径映射进临时 root，
        # 用 read_file（走同一映射）验证，避免 exec 命令内绝对路径语义分歧
        assert 'workspace file' in await backend.read_file('/work/workspace/note.txt')
        assert 'CREATE TABLE' in await backend.read_file('/work/system/crm_db/schema.sql')
        assert 'bridge' in await backend.read_file('/work/_setup/mcp_bridge.py')
        # rubrics 装配
        assert isinstance(info.get('rubrics'), dict)
        assert len(info['rubrics']['items']) == 2
        assert spec.rubrics is None  # setup 不直接改 spec.rubrics（mimo_env 负责）
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_setup_general_env_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ges, 'ENV_ROOT', str(tmp_path / 'general'))
    spec = general_spec('envs/s3k_missing')
    backend = LocalBackend(spec)
    await backend.start()
    try:
        with pytest.raises(GeneralEnvError):
            await setup_general_env(spec, backend)
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_setup_general_env_mcp_servers(env_task):
    """manifest 带 mcp_servers 时写入 spec.extra（URL 保持容器内 loopback）。"""
    manifest_path = env_task / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['mcp_servers'] = [
        {'name': 'crm', 'url': 'http://127.0.0.1:39101/mcp'},
        {'name': 'erp', 'url': 'http://127.0.0.1:39102/mcp'},
    ]
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    spec = general_spec()
    backend = LocalBackend(spec)
    await backend.start()
    try:
        info = await setup_general_env(spec, backend)
        assert spec.extra['mcp_servers'] == {
            'crm': 'http://127.0.0.1:39101/mcp',
            'erp': 'http://127.0.0.1:39102/mcp',
        }
        assert len(info['mcp_servers']) == 2
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_official_verify_skipped_on_local(env_task):
    """local 后端：官方 verify 主路径跳过（脚本内绝对路径契约不可满足），
    无 judge 配置时按 infra 处理，不当模型 0 分。"""
    from verifier.general_verifier import GeneralVerifier

    spec = general_spec()
    backend = LocalBackend(spec)
    await backend.start()
    try:
        await setup_general_env(spec, backend)
        spec.rubrics = {'items': [{'id': 'r1', 'method': 'llm', 'weight': 1.0,
                                   'question': 'ok?', 'pass_anchor': 'yes',
                                   'files': ['answer.md']}]}
        spec.extra['answer'] = 'The final answer is 42.'
        v = GeneralVerifier(spec)
        # _official_verify 显式跳过 local
        assert await v._official_verify(backend, 'answer') is None
        # 兜底 judge：未配置 judge -> infra（不是 0 分）
        res = await v.verify(backend)
        assert res.is_infra
        assert 'judge_not_configured' in res.message
    finally:
        await backend.stop()


class _FakeDockerBackend:
    """模拟 docker 语义的最小后端：内存文件系统 + 命令记录。

    _official_verify 只用到 exec / read_file / upload（经 _put_file），
    name 必须非 'local'。verify 命令由测试注入行为。
    """
    name = 'docker'

    def __init__(self):
        self.files: dict = {}
        self.commands: list = []

    async def exec(self, command, timeout=None, cwd=None):
        from sandbox.base import ExecResult
        self.commands.append(command)
        if 'printf %s' in command and 'base64 -d' in command:
            # agent_output 写入（GeneralVerifier 第2步）
            b64 = command.split('printf %s ')[1].split(' | base64')[0]
            import base64 as _b64
            path = command.rsplit('> ', 1)[1].strip()
            self.files[path] = _b64.b64decode(b64).decode('utf-8')
            return ExecResult(output='', returncode=0)
        if command == 'python3 /work/run_verify.py':
            # 官方脚本行为模拟：读 agent_output -> 写 reward.json
            self.files['/logs/verifier/reward.json'] = '{"reward": 0.75}'
            return ExecResult(output='[verify] score=0.750', returncode=0)
        return ExecResult(output='', returncode=0)

    async def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path, content):
        self.files[path] = content

    async def upload_dir(self, local_dir, remote_dir):
        for root, _d, files in os.walk(local_dir):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), local_dir)
                self.files[os.path.join(remote_dir, rel)] = open(
                    os.path.join(root, fn), encoding='utf-8', errors='replace').read()
        return len(self.files)


@pytest.mark.asyncio
async def test_official_verify_docker_path(env_task):
    """docker 语义后端：官方 verify 主路径全链路（上传->agent_output->执行->reward.json）。"""
    from verifier.general_verifier import GeneralVerifier

    spec = general_spec()
    backend = _FakeDockerBackend()
    # 任务文件已按 manifest 装配（uploads 部分）——这里只需 verifier 侧逻辑
    spec.extra['answer'] = 'The final answer is 42.'
    v = GeneralVerifier(spec)
    res = await v.verify(backend)
    assert not res.is_infra, res.message
    assert res.score == 0.75
    assert res.components.get('official') == 0.75
    # agent_output 契约：写到 /tmp/agent_output/sessions/main.jsonl 且是 assistant 事件
    ao = backend.files['/tmp/agent_output/sessions/main.jsonl']
    assert '"type": "assistant"' in ao
    assert 'The final answer is 42.' in ao
    # 官方命令被执行
    assert 'python3 /work/run_verify.py' in backend.commands
