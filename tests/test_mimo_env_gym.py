"""gym 适配测试：MiMoAgentEnv（ms-swift 训练实际使用的集成）。

仅在 ms-swift 已安装时运行；未安装则整模块跳过。
mock 模型输出 -> 沙箱真实执行 -> verifier 打分，验证 reset/step/close 契约。
"""
import json

import pytest

from tests import _fixtures as fx

pytestmark = pytest.mark.skipif(not fx.HAS_SWIFT,
                                reason='ms-swift 未安装，跳过 gym 适配测试')


def _tool(name, arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


async def test_gym_code_correct():
    env = await fx.make_code_env()
    try:
        problem = env.spec.problem_statement
        a1 = _tool('write_file', {'path': 'solution.py', 'content': fx.SOLUTION_GOOD})
        obs1, r1, done1, _ = await env.step([
            {'role': 'user', 'content': problem},
            {'role': 'assistant', 'content': a1},
        ])
        assert r1 == 0.0 and not done1
        a2 = _tool('submit', {})
        obs2, r2, done2, info = await env.step([
            {'role': 'user', 'content': problem},
            {'role': 'assistant', 'content': a1},
            {'role': 'user', 'content': obs1},
            {'role': 'assistant', 'content': a2},
        ])
        assert done2 and info.get('is_infra') is False
        assert r2 == 1.0, info
    finally:
        await env.close()


async def test_gym_code_wrong():
    env = await fx.make_code_env()
    try:
        problem = env.spec.problem_statement
        a1 = _tool('write_file', {'path': 'solution.py', 'content': fx.SOLUTION_BAD})
        obs1, _, _, _ = await env.step([
            {'role': 'user', 'content': problem},
            {'role': 'assistant', 'content': a1},
        ])
        a2 = _tool('submit', {})
        _, r2, done2, info = await env.step([
            {'role': 'user', 'content': problem},
            {'role': 'assistant', 'content': a1},
            {'role': 'user', 'content': obs1},
            {'role': 'assistant', 'content': a2},
        ])
        assert done2 and info.get('is_infra') is False
        assert r2 == 0.0, info
    finally:
        await env.close()


async def test_gym_music():
    spec = fx.music_spec()
    env = fx.MiMoAgentEnv(spec.to_env_config())
    await env.reset(None)
    try:
        _, r, done, info = await env.step([
            {'role': 'user', 'content': spec.problem_statement},
            {'role': 'assistant', 'content': fx.GOOD_ABC},
        ])
        assert done and info.get('is_infra') is False and r >= 0.9
    finally:
        await env.close()


async def test_gym_scheduler_closes_env_at_max_turns():
    """MiMoGymScheduler：max_turns 到顶的轨迹必须关闭 env（防会话泄漏）。"""
    from types import SimpleNamespace
    from swift_agent.mimo_env import MiMoGymScheduler

    class _FakeEnv:
        def __init__(self):
            self.closed = False

        async def step(self, action):
            return 'obs', 0.0, False, {}

        async def close(self):
            self.closed = True

    sched = MiMoGymScheduler(infer_engine=None, max_turns=2)
    fake = _FakeEnv()
    req = SimpleNamespace(uuid='u1', messages=[{'role': 'assistant', 'content': 'x'}])
    sched._envs['u1'] = fake
    sched._total_rewards['u1'] = 0.0
    sched._step_rewards['u1'] = []
    sched._pending_obs['u1'] = None

    # 第 1 轮：未到 max_turns，不关闭
    r1 = await sched.on_turn_end(req, None, 1)
    assert not r1['done'] and not fake.closed
    # 第 2 轮：到达 max_turns，关闭
    r2 = await sched.on_turn_end(req, None, 2)
    assert not r2['done'] and fake.closed
    assert r2['rollout_infos'].get('stop_reason') == 'max_turns'
    assert 'u1' not in sched._envs
