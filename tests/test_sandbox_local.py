"""LocalBackend 基础能力测试（真实在临时目录执行）。"""
import pytest

from sandbox.base import get_backend
from swift_agent.spec import TaskSpec


def _spec():
    return TaskSpec(instance_id='t', domain='code', dataset_type='opensource-code',
                    backend='local', tool_timeout_sec=5)


async def test_exec_ok():
    async with get_backend(_spec()) as b:
        r = await b.exec('echo hi && python3 -c "print(1+1)"')
        assert r.returncode == 0
        assert 'hi' in r.output and '2' in r.output


async def test_write_read():
    async with get_backend(_spec()) as b:
        await b.write_file('dir/a.txt', 'hello')
        assert await b.read_file('dir/a.txt') == 'hello'
        with pytest.raises(FileNotFoundError):
            await b.read_file('missing.txt')


async def test_timeout():
    async with get_backend(_spec()) as b:
        r = await b.exec('sleep 5', timeout=1)
        assert r.returncode == 124
        assert r.reason == 'timeout'


async def test_path_traversal_blocked():
    async with get_backend(_spec()) as b:
        # ../ 逃逸出 root：必须拦截
        with pytest.raises(ValueError):
            await b.read_file('../../etc/passwd')
        # 容器语义的绝对路径映射进 root（local 模拟容器文件系统），不越界
        await b.write_file('/work/a.txt', 'x')
        assert await b.read_file('/work/a.txt') == 'x'
        # 相对 .. 逃逸同样拦截
        with pytest.raises(ValueError):
            await b.write_file('dir/../../evil', 'x')
