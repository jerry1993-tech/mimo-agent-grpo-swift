"""CodeVerifier 直接评判测试（local 后端 + 最小 git/python 工程，无需 ms-swift）。"""
from verifier.code_verifier import CodeVerifier
from tests import _fixtures as fx


async def test_correct_solution_passes():
    spec = fx.code_spec()
    backend = await fx.make_code_backend(spec)
    try:
        await backend.write_file('solution.py', fx.SOLUTION_GOOD)
        res = await CodeVerifier(spec).verify(backend)
        assert not res.is_infra, res.message
        assert res.score == 1.0 and res.passed
        assert res.components['tests_passed'] == 1.0
    finally:
        await backend.stop()


async def test_wrong_solution_fails():
    spec = fx.code_spec()
    backend = await fx.make_code_backend(spec)
    try:
        await backend.write_file('solution.py', fx.SOLUTION_BAD)
        res = await CodeVerifier(spec).verify(backend)
        assert not res.is_infra, res.message
        assert res.score == 0.0 and not res.passed
    finally:
        await backend.stop()
