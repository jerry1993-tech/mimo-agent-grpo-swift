"""Webdev 域评判器：渲染 dist/index.html 截图做视觉评分；组级相对排序需外部 grader。

对齐小米 webdev 两段式：
1. per-rollout：渲染截图 -> POST {DESIGN_GRADER_URL}/grade 拿 query_score；
2. 组级（GRPO 同 prompt 的 N=16 兄弟 rollout 到齐后）：POST /grade_group，做 8 轮成对
   相对美学排序（groupwise relative pick），本类只留清晰接口 TODO。

未配置 DESIGN_GRADER_URL 时降级为【确定性检查】并在 message 注明“未做组内相对排序”：
- dist/index.html（或 index.html）存在 -> score 1.0；否则 0.0（模型没产出 = 正常低分）；
渲染/截图/HTTP 传输基建失败 -> is_infra。

参考：
- https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/design/webdev/group_reward.py
- grader 服务端点 POST /grade、POST /grade_group、GET /healthz
"""
from __future__ import annotations

import base64
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sandbox.base import SandboxBackend  # noqa: E402
from verifier.base import VerificationResult, Verifier, register_verifier  # noqa: E402

_CANDIDATE_ARTIFACTS = ('dist/index.html', 'index.html', 'build/index.html')


@register_verifier('webdev')
class WebdevVerifier(Verifier):
    domain = 'webdev'

    # ------------------------------------------------------------------
    # 确定性检查
    # ------------------------------------------------------------------
    async def _find_artifact(self, backend: SandboxBackend) -> Optional[str]:
        cwd = self.spec.cwd or None
        for rel in _CANDIDATE_ARTIFACTS:
            try:
                res = await backend.exec(f'test -f {rel} && echo OK', timeout=10.0, cwd=cwd)
                if res.returncode == 0 and 'OK' in res.output:
                    return rel
            except Exception:
                continue
        return None

    def _grader_url(self) -> str:
        return (self.spec.extra.get('design_grader_url')
                or os.environ.get('DESIGN_GRADER_URL') or '').rstrip('/')

    # ------------------------------------------------------------------
    # 外部 grader（可选；未配置则跳过）
    # ------------------------------------------------------------------
    async def post_grade(self, grader_url: str, *, query: str, shot_b64: str,
                         runtime_factors: Optional[Dict[str, Any]] = None) -> float:
        """POST {grader_url}/grade，返回 query_score∈[0,1]。

        TODO(基建)：shot_b64 当前需由容器内 playwright/chromium 渲染 dist/index.html 截图后
        提供（本类不内置浏览器）。未截图时调用方应跳过本步、走确定性降级分。
        """
        import httpx  # 延迟导入
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f'{grader_url}/grade',
                json={'query': query, 'shot_jpg_b64': shot_b64,
                      'runtime_factors': runtime_factors or {}},
                timeout=120.0,
            )
            r.raise_for_status()
            data = r.json()
        return float(data.get('query_score', 0.0))

    async def post_grade_group(self, grader_url: str, *, shots_b64: List[str],
                               query: str, qscores: List[float],
                               runtime_factors: Optional[Dict[str, Any]] = None) -> List[float]:
        """组级相对排序接口（TODO）：同 GRPO 组 N=16 张截图切片 GROUP_SLICE=8 成对比较。

        由 driver 侧在 ``_compute_advantage`` 整组到齐后调用，单 rollout 的 verify 不调用。
        返回每行 reward∈[0,1]。失败行由组内有效行均值替换。
        """
        raise NotImplementedError('组级相对排序 /grade_group 由 driver 侧整组回调，单 rollout 不调用')

    # ------------------------------------------------------------------
    async def verify(self, backend: SandboxBackend) -> VerificationResult:
        spec = self.spec
        try:
            artifact = await self._find_artifact(backend)
        except Exception as e:
            return self.infra_failure(f'webdev_artifact_check_failed: {e!r}')

        components: Dict[str, float] = {'artifact_present': 1.0 if artifact else 0.0}
        if not artifact:
            return VerificationResult(
                score=0.0, passed=False, components=components, is_infra=False,
                message='未找到 dist/index.html/index.html 产物（模型未交付，正常 0 分）',
            )

        grader = self._grader_url()
        if not grader:
            return VerificationResult(
                score=1.0, passed=True, components=components, is_infra=False,
                message=(f'产物存在: {artifact}。未配置 DESIGN_GRADER_URL，已降级为确定性检查；'
                         f'未做组内相对排序（/grade_group 需 driver 侧整组回调）。'),
            )

        # 配置了 grader：尝试视觉 query 分。截图基建（playwright）缺失时按 infra 上报，
        # 这里通过 spec.extra 透传已截好的图（base64）；否则降级。
        shot_b64 = spec.extra.get('shot_b64') or ''
        if not shot_b64:
            return VerificationResult(
                score=1.0, passed=True, components=components, is_infra=False,
                message=(f'产物存在: {artifact}。已配置 grader={grader} 但未注入截图(shot_b64)，'
                         f'跳过视觉 query 分，按确定性 1.0 计；未做组内相对排序。'),
            )
        try:
            qscore = await self.post_grade(grader, query=spec.problem_statement, shot_b64=shot_b64)
            components['query_score'] = qscore
            return VerificationResult(
                score=qscore, passed=qscore >= 0.5, components=components, is_infra=False,
                message=f'产物存在: {artifact}; /grade query_score={qscore:.3f}; 未做组内相对排序',
            )
        except Exception as e:
            return self.infra_failure(f'webdev_grader_call_failed: {e!r}')
