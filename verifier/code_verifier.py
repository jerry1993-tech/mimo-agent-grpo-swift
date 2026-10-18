"""Code 域评判器：git apply 隐藏测试补丁 -> 跑 test_command，exit 0 即 1.0（二值）。

严格对齐小米 OpenSourceCodeEnvironment._do_calculate_reward：
1. 用正则从 test_patch 解析受影响文件，先 ``backend.reset_files(files)`` 还原基线；
2. 把 test_patch 写到容器临时路径并 ``git apply``（失败再 --whitespace=nowarn）；
3. 跑 ``test_command``（timeout=verifier_timeout_sec），returncode==0 -> score=1.0 否则 0.0。

铁律：隐藏测试只能在 ``verify()`` 内、agent 结束后注入；agent 工作期间容器内无测试文件。
reset/apply 失败 = 基建故障（is_infra），不是模型 0 分。

参考：
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/datasets/opensource_code.py
"""
from __future__ import annotations

import os
import re
import sys
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sandbox.base import SandboxBackend  # noqa: E402
from verifier.base import VerificationResult, Verifier, register_verifier  # noqa: E402

# diff --git a/path/to/file b/path/to/file
_DIFF_FILE_RE = re.compile(r'^diff --git a/(.+?) b/(.+?)\s*$', re.MULTILINE)

# 相对路径：DockerBackend 以 spec.cwd 为 tar 根、LocalBackend 以临时 root 为根，
# write_file 与 exec(git apply, cwd=spec.cwd) 落到同一目录；绝对路径会触发 LocalBackend
# 的 realpath 路径穿越防护而抛 ValueError。
_PATCH_TMP = '.mimo_verify_test.patch'


def _affected_files(test_patch: str) -> List[str]:
    """从 unified diff 提取受影响文件路径（取 b 侧；a==b 等价）。"""
    files: List[str] = []
    for a, b in _DIFF_FILE_RE.findall(test_patch or ''):
        p = b or a
        if p not in files:
            files.append(p)
    return files


@register_verifier('code')
class CodeVerifier(Verifier):
    domain = 'code'

    async def verify(self, backend: SandboxBackend) -> VerificationResult:
        spec = self.spec
        patch = spec.test_patch or ''
        if not patch.strip():
            return self.infra_failure('testbed_corrupted: test_patch 为空，无法注入隐藏测试')

        files = _affected_files(patch)

        # 1) 还原受影响文件到基线。reset_files 默认返回 False 表示“后端不支持 reset”
        #    （见 sandbox/base.py 约定），不视为 infra；只有抛异常才算基建故障。
        try:
            reset_ok = await backend.reset_files(files)
        except Exception as e:  # 传输/后端异常
            return self.infra_failure(f'testbed_reset_failed: {e!r}')
        reset_note = '' if reset_ok else '（后端未支持 git reset，直接 apply）'

        cwd = spec.cwd or None

        # 2) 写入补丁并 git apply
        try:
            await backend.write_file(_PATCH_TMP, patch)
        except Exception as e:
            return self.infra_failure(f'testbed_corrupted: 写补丁失败 {e!r}')

        apply_res = await backend.exec(f'git apply {_PATCH_TMP}', timeout=120.0, cwd=cwd)
        if apply_res.returncode != 0:
            # 退避一次：容忍空白/上下文差异
            apply_res = await backend.exec(
                f'git apply --whitespace=nowarn {_PATCH_TMP}', timeout=120.0, cwd=cwd,
            )
        if apply_res.reason in ('transport_error', 'backend_error'):
            return self.infra_failure(f'testbed_corrupted: git apply 传输错误 {apply_res.output[:200]}')
        if apply_res.returncode != 0:
            tail = apply_res.output[-800:]
            return self.infra_failure(
                f'testbed_corrupted: git apply 失败（受影响文件={files}）{reset_note}\n{tail}'
            )

        # 3) 跑测试
        timeout = float(spec.verifier_timeout_sec or 300.0)
        test_res = await backend.exec(spec.test_command, timeout=timeout, cwd=cwd)

        if test_res.reason in ('transport_error', 'backend_error'):
            return self.infra_failure(
                f'test_command 执行基建错误 reason={test_res.reason}: {test_res.output[:300]}'
            )

        passed = test_res.returncode == 0
        score = 1.0 if passed else 0.0
        tail = test_res.output[-1500:]
        msg = (f'{"PASS" if passed else "FAIL"} 受影响文件={files}{reset_note}\n'
               f'$ {spec.test_command}\n[exit={test_res.returncode} reason={test_res.reason}]\n{tail}')
        return VerificationResult(
            score=score,
            passed=passed,
            components={'tests_passed': 1.0 if passed else 0.0},
            is_infra=False,
            message=msg,
        )
