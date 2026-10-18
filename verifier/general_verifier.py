"""General 域评判器：容器内官方 verify 优先，LLM-as-judge rubric 兜底。

对齐小米 rubric_judge / run_verify.py：
- 主路径（与上游一致）：manifest.json 的 verifier 块声明 verify 脚本上传与运行命令；
  在容器内跑 ``run_verify.py``（读 agent 最终回答 + system/*/state.db post_state +
  LLM judge），从 ``reward.json`` 读分数。score 即官方加权分。
  （setup_general_env 已把 env_task_dir 的文件按 manifest 装进容器；本 verifier
  只需补 verifier.uploads 的文件、写 agent_output 日志、跑 command、读 reward。）
- 兜底路径（官方脚本缺失/失败时）：对 rubric（verifier_meta.items）逐条 LLM 判 1/0，
  B=加权均值，可选分级 S 按 w=0.5 合成。与上游同构但非官方实现。
- 官方脚本失败 = infra（不是模型 0 分）；兜底路径 judge 未配置也 = infra。

参考：
- https://github.com/XiaomiMiMo/mimoagent/blob/mimo-oss/src/mimoagent/environments/rubric_judge.py
- 数据集 general/envs/<task>/run_verify.py（官方打分入口）
"""
from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sandbox.base import SandboxBackend  # noqa: E402
from sandbox.general_env_setup import find_env_task_dir, GeneralEnvError  # noqa: E402
from verifier.base import VerificationResult, Verifier, register_verifier  # noqa: E402
from verifier.llm_judge import (  # noqa: E402
    JudgeError, judge_batch, judge_configured, resolve_judge_config,
)

_SYNTH_W = 0.5  # w·S + (1−w)·B，与小米默认一致

# agent 最终回答的日志路径（run_verify.py 的 extract_agent_output 从这里读）
_AGENT_OUTPUT_SESSION = '/tmp/agent_output/sessions/main.jsonl'


def _items(rubrics: Any) -> List[Dict[str, Any]]:
    if isinstance(rubrics, dict):
        if isinstance(rubrics.get('items'), list):
            return [i for i in rubrics['items'] if isinstance(i, dict)]
        return []
    if isinstance(rubrics, list):
        return [i for i in rubrics if isinstance(i, dict)]
    return []


@register_verifier('general')
class GeneralVerifier(Verifier):
    domain = 'general'

    # ------------------------------------------------------------------
    # 主路径：容器内官方 verify（manifest.verifier）
    # ------------------------------------------------------------------
    async def _official_verify(self, backend: SandboxBackend,
                               answer: str) -> Optional[VerificationResult]:
        """按 manifest.verifier 在容器内跑官方打分。返回 None 表示无官方脚本可跑。

        官方脚本内部使用容器绝对路径（/work、/logs、/tmp/agent_output 等），
        仅在隔离后端（docker/remote）下可用；local 后端（宿主机临时目录模拟）
        无法满足脚本内路径契约，直接跳过主路径。
        """
        if getattr(backend, 'name', '') == 'local':
            return None
        local_dir = find_env_task_dir(self.spec)
        if local_dir is None:
            return None
        manifest_path = os.path.join(local_dir, 'manifest.json')
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        verifier = manifest.get('verifier') or {}
        if not verifier.get('command'):
            return None

        # 1) verifier.uploads：本地 -> 容器
        from sandbox.general_env_setup import _put_file
        for up in verifier.get('uploads') or []:
            src = os.path.join(local_dir, str(up.get('source', '')))
            dst = str(up.get('target', ''))
            if src and dst and os.path.isfile(src):
                try:
                    await _put_file(backend, src, dst)
                except Exception as e:
                    return self.infra_failure(f'verifier upload failed: {e!r}')

        # 2) 写 agent 最终回答（run_verify.extract_agent_output 的输入契约：
        #    /tmp/agent_output/sessions/**/*.jsonl，每行一个 assistant 事件。
        #    base64 落盘避免任何 shell 引号/转义问题）
        event = {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': answer}]}}
        line = json.dumps(event, ensure_ascii=False)
        b64 = base64.b64encode(line.encode('utf-8')).decode('ascii')
        res = await backend.exec(
            f'mkdir -p /tmp/agent_output/sessions '
            f'&& printf %s {b64} | base64 -d > {_AGENT_OUTPUT_SESSION}',
            timeout=30.0, cwd='/')
        if res.returncode != 0:
            return self.infra_failure(f'write agent_output failed: {res.output[:200]}')

        # 3) 跑官方 verify 命令
        timeout = float(verifier.get('timeout_sec') or self.spec.verifier_timeout_sec or 600)
        res = await backend.exec(str(verifier['command']), timeout=timeout, cwd='/')
        if res.reason in ('transport_error', 'backend_error'):
            return self.infra_failure(f'verifier command transport error: {res.output[:200]}')

        # 4) 读 reward.json（masking contract：失败时无 reward 键 -> reward_error）
        reward_file = verifier.get('reward_file') or '/logs/verifier/reward.json'
        try:
            payload_text = await backend.read_file(reward_file)
        except Exception as e:
            return self.infra_failure(f'reward.json unreadable: {e!r}')
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return self.infra_failure('reward.json invalid JSON')
        if 'reward' not in payload:
            return self.infra_failure(
                f"official verifier masked (reward_error={payload.get('reward_error')!r}); "
                f'tail: {res.output[-300:]}')
        score = float(payload.get('reward', 0.0) or 0.0)
        # reward_detail.json 供观测
        detail_file = verifier.get('reward_detail_file') or '/logs/verifier/reward_detail.json'
        try:
            detail_text = await backend.read_file(detail_file)
            detail = json.loads(detail_text)
        except Exception:
            detail = {}
        passed_cnt = detail.get('passed')
        total_cnt = detail.get('total')
        msg = (f'official verify score={score:.3f}'
               + (f' ({passed_cnt}/{total_cnt} rubrics)' if total_cnt else '')
               + f'\n{res.output[-800:]}')
        return VerificationResult(
            score=score,
            passed=score >= 0.5,
            components={'official': score},
            is_infra=False,
            message=msg,
        )

    # ------------------------------------------------------------------
    # 兜底路径：LLM rubric judge（无官方脚本时）
    # ------------------------------------------------------------------
    async def _collect_answer(self, backend: SandboxBackend, items: List[Dict[str, Any]]) -> str:
        """取 doer 最终产物：优先 spec.extra 里的 submit answer；其次读 rubric 指定/默认文件。"""
        spec = self.spec
        for k in ('answer', 'final_answer', 'submission', 'response'):
            v = spec.extra.get(k)
            if isinstance(v, str) and v.strip():
                return v

        candidates: List[str] = []
        seen = set()
        for it in items:
            for f in (it.get('files') or []):
                if f not in seen:
                    seen.add(f)
                    candidates.append(f)
        # 兜底常见提交文件
        for f in ('answer.md', 'response.md', 'final_answer.md'):
            if f not in seen:
                seen.add(f)
                candidates.append(f)

        cwd = spec.cwd or None
        for rel in candidates:
            try:
                txt = await backend.read_file(rel)
                if txt and txt.strip():
                    return txt
            except Exception:
                continue
            # read_file 对绝对路径/相对路径后端实现不一，再试 exec cat
            try:
                res = await backend.exec(f'cat {rel}', timeout=15.0, cwd=cwd)
                if res.returncode == 0 and res.output.strip():
                    return res.output
            except Exception:
                continue
        return ''

    async def verify(self, backend: SandboxBackend) -> VerificationResult:
        spec = self.spec
        items = _items(spec.rubrics)

        answer = await self._collect_answer(backend, items)
        if not answer.strip():
            # doer 没产出可读答案 -> 环境侧问题（文件读不到），按 infra
            return self.infra_failure('answer_not_found: 读不到 doer 最终答案（answer/submit/answer.md）')

        # ---- 主路径：官方容器内 verify ----
        official = await self._official_verify(backend, answer)
        if official is not None and not official.is_infra:
            return official
        # 官方脚本存在但 infra 失败：直接返回（不当模型 0 分）
        if official is not None and official.is_infra:
            return official

        # ---- 兜底：LLM rubric judge ----
        if not items:
            return self.infra_failure('general: rubrics 为空且无官方 verify 脚本（verifier_meta.items 缺失）')

        cfg = resolve_judge_config(spec)
        if not judge_configured(cfg):
            return self.infra_failure(
                'judge_not_configured: 未配置 judge_base_url/judge_api_key/judge_model '
                '（或 JUDGE_*/OPENAI_* 环境变量），无法做 LLM rubric 判定'
            )
        base_url, api_key, model = cfg

        context_base = (f'[Problem]\n{spec.problem_statement}\n\n[Doer Answer]\n{answer}')

        llm_items = [it for it in items if it.get('method', 'llm') == 'llm']
        if not llm_items:
            return self.infra_failure('general: 没有 method==llm 的 rubric 可判')

        # 逐条判定
        try:
            verdicts = await judge_batch(
                [(it['question'], f'{context_base}\n\n[Reference Anchor]\n{it.get("pass_anchor","")}')
                 for it in llm_items],
                base_url=base_url, api_key=api_key, model=model,
            )
        except JudgeError as e:
            return self.infra_failure(f'judge_call_failed: {e!r}')

        components: Dict[str, float] = {}
        num = den = 0.0
        for it, (ok, raw) in zip(llm_items, verdicts):
            vid = str(it.get('id') or f'rubric_{len(components)}')
            components[vid] = 1.0 if ok else 0.0
            w = float(it.get('weight') or 0.0)
            num += w * (1.0 if ok else 0.0)
            den += w
        B = (num / den) if den > 0 else (sum(1.0 if ok else 0.0 for ok, _ in verdicts) / len(verdicts))

        # 可选分级 rubric -> S（这里取所有 graded 项是否达成最高档的均值，钩子）
        graded = [it for it in items if it.get('method') in ('graded', 'grade', 'solution')]
        S: Optional[float] = None
        if graded:
            # 简化：graded 项同样走 yes/no 判定其是否达到“满足等级”
            try:
                gv = await judge_batch(
                    [(it['question'], f'{context_base}\n\n[Reference Anchor]\n{it.get("pass_anchor","")}')
                     for it in graded],
                    base_url=base_url, api_key=api_key, model=model,
                )
                S = sum(1.0 if ok else 0.0 for ok, _ in gv) / len(gv)
            except JudgeError:
                S = None

        if S is not None:
            score = _SYNTH_W * S + (1 - _SYNTH_W) * B
            components['S'] = S
        else:
            score = B
        components['B'] = B

        return VerificationResult(
            score=score,
            passed=score >= 0.5,
            components=components,
            is_infra=False,
            message=(f'general rubric (fallback judge): B={B:.3f}'
                     + (f' S={S:.3f} (合成 w={_SYNTH_W})' if S is not None else '')
                     + f' over {len(llm_items)} items'),
        )
