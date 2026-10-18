"""Verifier 评判器抽象层（框架无关，禁止 import ms-swift）。

对齐小米“真实执行结果 -> outcome reward”的设计，五个领域各自实现：
- code:    git apply 隐藏测试补丁 -> 跑 test_command，exit 0 即 1.0（二值）
- cyber:   解析 sanitizer 崩溃函数 + error_type，全等匹配（二值，无 LLM）
- general: LLM-as-judge 对 rubric 逐条 1/0，取均值（可与程序判罚合成）
- webdev:  渲染截图 -> 视觉评分；组级相对排序需外部 grader 服务
- music:  ABC 记谱校验 + 特征规则分（无沙箱）

参考：
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/datasets/opensource_code.py
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/rubric_judge.py
- https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/design/music/scorer/score.py
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from swift_agent.spec import TaskSpec
from sandbox.base import SandboxBackend


@dataclass
class VerificationResult:
    """评判结果。score 为最终 outcome；components 仅供观测，不回传梯度。"""
    score: float
    passed: bool = False
    components: Dict[str, float] = field(default_factory=dict)
    is_infra: bool = False
    message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'passed': self.passed,
            'components': self.components,
            'is_infra': self.is_infra,
            'message': self.message,
        }


class Verifier(ABC):
    """评判器接口。一个实例针对一个任务。"""

    domain: str = ''

    def __init__(self, spec: TaskSpec, **opts):
        self.spec = spec
        self.opts = opts

    @abstractmethod
    async def verify(self, backend: SandboxBackend) -> VerificationResult:
        """基于沙箱真实执行结果（或 music 的纯文本）计算最终 outcome。"""
        raise NotImplementedError

    # 便捷构造
    def infra_failure(self, message: str) -> VerificationResult:
        """基建故障统一出口：is_infra=True，score=0（调用方应在 GRPO 组内剔除，勿当模型 0 分）。"""
        return VerificationResult(score=0.0, passed=False, is_infra=True, message=message)


# ----------------------------------------------------------------------
# 注册表 + 工厂
# ----------------------------------------------------------------------
VERIFIERS: Dict[str, type] = {}


def register_verifier(domain: str):
    def _wrap(cls):
        VERIFIERS[domain] = cls
        cls.domain = domain
        return cls
    return _wrap


def get_verifier(spec: TaskSpec, **opts) -> Verifier:
    """按 TaskSpec.domain 选择评判器；惰性导入。"""
    domain = spec.domain
    if domain not in VERIFIERS:
        if domain == 'code':
            from verifier.code_verifier import CodeVerifier  # noqa: F401
        elif domain == 'cyber':
            from verifier.cyber_verifier import CyberVerifier  # noqa: F401
        elif domain == 'general':
            from verifier.general_verifier import GeneralVerifier  # noqa: F401
        elif domain == 'webdev':
            from verifier.webdev_verifier import WebdevVerifier  # noqa: F401
        elif domain == 'music':
            from verifier.music_verifier import MusicVerifier  # noqa: F401
        else:
            raise ValueError(f'No verifier for domain: {domain!r}. '
                             f'Available: {sorted(set(VERIFIERS) | {"code","cyber","general","webdev","music"})}')
    cls = VERIFIERS.get(domain)
    if cls is None:
        raise RuntimeError(f'verifier {domain} registered but class missing')
    return cls(spec, **opts)
