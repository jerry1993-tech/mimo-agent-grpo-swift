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
