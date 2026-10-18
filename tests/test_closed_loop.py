"""端到端闭环测试（框架无关，经 agent_runner；无需 docker/GPU/ms-swift）。

mock policy（ScriptedPolicyClient）产出工具调用 -> local 沙箱真实执行 -> verifier 打分：
- code：write_file(解) -> submit -> 隐藏测试打分；正确=1，错误=0。
- music：单轮 ABC -> 规则打分；合法高分，非法 0。
"""
import json

from swift_agent.agent_runner import run_episode, ScriptedPolicyClient
from tests import _fixtures as fx


def _tool(name, arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


async def _seed_git(backend):
    r = await backend.exec(
        'git init -q && git config user.email t@t && git config user.name t '
        '&& echo readme > README.md && git add -A && git commit -qm base'
    )
    assert r.returncode == 0, r.output


async def test_closed_loop_code_correct():
    turns = [
        _tool('write_file', {'path': 'solution.py', 'content': fx.SOLUTION_GOOD}),
        _tool('submit', {}),
    ]
    res = await run_episode(fx.code_spec(), ScriptedPolicyClient(turns),
                            max_turns=5, env_setup_hook=_seed_git)
    assert not res.is_infra, res.components
    assert res.reward == 1.0, res.components


async def test_closed_loop_code_wrong():
    turns = [
        _tool('write_file', {'path': 'solution.py', 'content': fx.SOLUTION_BAD}),
        _tool('submit', {}),
    ]
    res = await run_episode(fx.code_spec(), ScriptedPolicyClient(turns),
                            max_turns=5, env_setup_hook=_seed_git)
    assert not res.is_infra, res.components
    assert res.reward == 0.0, res.components


async def test_closed_loop_music_good():
    res = await run_episode(fx.music_spec(),
                            ScriptedPolicyClient([fx.GOOD_ABC]), max_turns=1)
    assert not res.is_infra
    assert res.reward >= 0.9, res.components


async def test_closed_loop_music_bad():
    res = await run_episode(fx.music_spec(),
                            ScriptedPolicyClient([fx.BAD_ABC]), max_turns=1)
    assert not res.is_infra
    assert res.reward == 0.0, res.components
