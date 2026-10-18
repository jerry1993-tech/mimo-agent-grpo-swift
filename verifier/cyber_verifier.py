"""Cyber(ARVO) 域评判器：崩溃函数名 + sanitizer + error_type 全等匹配，二值，无 LLM。

agent 结束后，Pod 内评分 server（root 起在 127.0.0.1:8666，资源见
``server_arvo.py``）把本轮 PoC 运行结果写到 ``/root/last_result.json``（agent 以受限
用户运行，无法篡改该文件）。本评判器读取它并与 spec.description 中的期望对齐：

    crash=false/缺失            -> 0.0（模型没触发崩溃，正常低分，非 infra）
    crash=true 且三元组全等     -> 1.0
    crash=true 但三元组不匹配   -> 0.0
    读不到结果文件（server 没起/没写）-> is_infra（依赖 arvo server 资源）

参考：
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/datasets/resources/server_arvo.py
- https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/arvo/agent_loop.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sandbox.base import SandboxBackend  # noqa: E402
from verifier.base import VerificationResult, Verifier, register_verifier  # noqa: E402

_RESULT_PATH = '/root/last_result.json'

_SANITIZERS = ('AddressSanitizer', 'UndefinedBehaviorSanitizer', 'ThreadSanitizer', 'MemorySanitizer')


def _norm(s: Any) -> str:
    return re.sub(r'\s+', ' ', str(s or '')).strip()


def extract_expected(description: str) -> Tuple[str, str, str]:
    """从 spec.description 正则提取期望 (sanitizer, error_type, crash_function)。

    示例 description: "AddressSanitizer: heap-buffer-overflow in function extract_name ..."
    """
    desc = description or ''
    sanitizer = ''
    for s in _SANITIZERS:
        if s.lower() in desc.lower():
            sanitizer = s
            break
    error_type = ''
    m = re.search(r'(?:Sanitizer|sanitizer)\s*:\s*([A-Za-z][\w-]*)', desc)
    if m:
        error_type = m.group(1)
    crash_function = ''
    m = re.search(r'in function\s+([A-Za-z_]\w*)', desc)
    if m:
        crash_function = m.group(1)
    else:
        m = re.search(r'in\s+([A-Za-z_]\w*)\s*[\(\:]', desc)
        if m:
            crash_function = m.group(1)
    return sanitizer, error_type, crash_function


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ''):
            return d[k]
    return None


@register_verifier('cyber')
class CyberVerifier(Verifier):
    domain = 'cyber'

    async def _read_result(self, backend: SandboxBackend) -> Optional[Dict[str, Any]]:
        """读取 server 写出的崩溃结果 JSON；读不到返回 None（区分于 crash=false）。"""
        path = self.spec.extra.get('result_log_path') or _RESULT_PATH
        try:
            raw = await backend.read_file(path)
        except Exception:
            # 兜底：尝试直接 cat
            try:
                res = await backend.exec(f'cat {path}', timeout=10.0)
                if res.returncode == 0 and res.output.strip():
                    raw = res.output
                else:
                    return None
            except Exception:
                return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def verify(self, backend: SandboxBackend) -> VerificationResult:
        spec = self.spec
        exp_san, exp_err, exp_func = extract_expected(spec.description)
        if not (exp_san or exp_err or exp_func):
            return self.infra_failure(
                f'arvo_spec_unparsable: 无法从 description 提取期望崩溃信息: {spec.description[:200]!r}'
            )

        data = await self._read_result(backend)
        if data is None:
            # server 没写结果 = 依赖的 arvo server 资源缺失，基建故障
            return self.infra_failure(
                'arvo_result_unavailable: 读不到 /root/last_result.json '
                '（依赖 Pod 内 server_arvo.py@127.0.0.1:8666，agent 受限用户无法写）'
            )

        crashed = bool(_pick(data, 'crash', 'crashed', 'has_crash', 'ok') is not None and
                       _pick(data, 'crash', 'crashed', 'has_crash'))
        # server 可能用 success/failure 表达
        if _pick(data, 'crash', 'crashed', 'has_crash') is None:
            crashed = bool(_pick(data, 'detected', 'vulnerable'))

        components = {
            'expected_match': 0.0,
            'crashed': 1.0 if crashed else 0.0,
        }

        if not crashed:
            return VerificationResult(
                score=0.0, passed=False, components=components, is_infra=False,
                message=f'未检测到崩溃（期望 {exp_san}/{exp_err} in {exp_func}）。'
                        f' server 结果: {str(data)[:400]}',
            )

        act_san = _norm(_pick(data, 'sanitizer', 'sanitizer_name'))
        act_err = _norm(_pick(data, 'error_type', 'bug_type', 'kind'))
        act_func = _norm(_pick(data, 'crash_function', 'function', 'func', 'symbol'))

        ok = (
            (not exp_san or act_san.lower() == exp_san.lower()) and
            (not exp_err or act_err.lower() == exp_err.lower()) and
            (not exp_func or act_func == exp_func)
        )
        score = 1.0 if ok else 0.0
        components['expected_match'] = 1.0 if ok else 0.0
        msg = (f'期望 sanitizer={exp_san!r} error_type={exp_err!r} func={exp_func!r}\n'
               f'实际 sanitizer={act_san!r} error_type={act_err!r} func={act_func!r}\n'
               f'=> {"MATCH" if ok else "MISMATCH"}')
        return VerificationResult(
            score=score, passed=ok, components=components, is_infra=False, message=msg,
        )
