"""测试夹具：构造 code/music 任务与 local 后端 / MiMoAgentEnv。

- 框架无关闭环（sandbox+verifier / agent_runner）不依赖 ms-swift；
- gym 适配（MiMoAgentEnv）仅在 ms-swift 已安装时可用（HAS_SWIFT）。
"""
from __future__ import annotations

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swift_agent.spec import TaskSpec  # noqa: E402
from sandbox.local_backend import LocalBackend  # noqa: E402  (registers 'local')

HAS_SWIFT = importlib.util.find_spec('swift') is not None

# --- code：隐藏测试补丁（打分时才注入；新增 test_solution.py）---
TEST_PATCH = (
    'diff --git a/test_solution.py b/test_solution.py\n'
    'new file mode 100644\n'
    '--- /dev/null\n'
    '+++ b/test_solution.py\n'
    '@@ -0,0 +1,3 @@\n'
    '+from solution import fib\n'
    '+assert fib(10) == 55\n'
    "+print('ok')\n"
)

SOLUTION_GOOD = (
    'def fib(n):\n'
    '    a, b = 0, 1\n'
    '    for _ in range(n):\n'
    '        a, b = b, a + b\n'
    '    return a\n'
)

SOLUTION_BAD = (
    'def fib(n):\n'
    '    return n\n'
)

# --- music：合法 / 非法 ABC ---
GOOD_ABC = (
    '```abc\n'
    'X:1\n'
    'M:4/4\n'
    'L:1/8\n'
    'Q:1/4=120\n'
    'K:C\n'
    'V:1\n'
    'C D E F | G A B c | c B A G |\n'
    'V:2\n'
    'E, F, G, A, | B, C D E | E D C B, |\n'
    '```'
)

BAD_ABC = 'I cannot compose music, sorry.'


def code_spec() -> TaskSpec:
    return TaskSpec(
        instance_id='demo-code-fib',
        dataset_type='opensource-code',
        domain='code',
        backend='local',
        cwd='',
        problem_statement='Implement fib(n) in solution.py so that fib(10) == 55.',
        test_command='python test_solution.py',
        test_patch=TEST_PATCH,
        verifier_timeout_sec=60,
        tool_timeout_sec=10,
    )


def music_spec() -> TaskSpec:
    return TaskSpec(
        instance_id='demo-music-1',
        dataset_type='music',
        domain='music',
        backend='local',
        problem_statement='Compose a short 4/4 tune at 120 BPM for 2 voices in C major.',
        music_meta={'meter': '4/4', 'bpm': 120, 'nvoice_want': 2,
                    'length': 'short', 'lang': 'en'},
    )


async def make_code_backend(spec: TaskSpec = None) -> LocalBackend:
    """起一个 local 后端并播种基线 git 仓库（不经过 mimo_env，无需 swift）。"""
    spec = spec or code_spec()
    backend = LocalBackend(spec)
    await backend.start()
    r = await backend.exec(
        'git init -q && git config user.email t@t && git config user.name t '
        '&& echo readme > README.md && git add -A && git commit -qm base'
    )
    assert r.returncode == 0, r.output
    return backend


if HAS_SWIFT:
    from swift_agent.mimo_env import MiMoAgentEnv  # noqa: E402

    async def make_code_env() -> 'MiMoAgentEnv':
        spec = code_spec()
        env = MiMoAgentEnv(spec.to_env_config())
        await env.reset(None)
        r = await env.backend.exec(
            'git init -q && git config user.email t@t && git config user.name t '
            '&& echo readme > README.md && git add -A && git commit -qm base'
        )
        assert r.returncode == 0, r.output
        return env
