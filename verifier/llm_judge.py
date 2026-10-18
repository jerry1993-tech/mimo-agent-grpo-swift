"""OpenAI 兼容异步判官（LLM-as-judge）。框架无关，禁止 import ms-swift。

供 general / webdev 域对 rubric 逐条打 1/0。走标准 ``POST {base_url}/chat/completions``，
优先 httpx.AsyncClient（无第三方 SDK 依赖）。配置解析顺序：
显式传入 > spec.judge_* > 环境变量（JUDGE_* / OPENAI_*）。

判官不可用（未配置 / 网络失败 / 重试后仍无有效裁决）视为【基建故障】，
由调用方置 ``is_infra=True``，绝不能当作模型答错的 0 分。

参考小米 rubric_judge（逐条独立 1/0 判定）：
- https://github.com/XiaomiMiMo/mimoagent/blob/main/src/mimoagent/environments/rubric_judge.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# 项目根 sys.path 兜底（允许 ``python -m verifier.llm_judge`` 或脚本直跑）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import httpx
except Exception:  # pragma: no cover - 依赖缺失时延迟到真正调用才报错
    httpx = None  # type: ignore


class JudgeError(Exception):
    """判官基建故障：未配置 / 传输失败 / 重试后仍无可用裁决。调用方应映射为 is_infra。"""


# ----------------------------------------------------------------------
# 配置解析
# ----------------------------------------------------------------------
def resolve_judge_config(spec: Any) -> Tuple[str, str, str]:
    """从 spec + 环境变量解析 (base_url, api_key, model)。任一为空串即视为未配置。"""
    base_url = (
        getattr(spec, 'judge_base_url', '')
        or os.environ.get('JUDGE_BASE_URL')
        or os.environ.get('OPENAI_BASE_URL')
        or ''
    )
    api_key = (
        getattr(spec, 'judge_api_key', '')
        or os.environ.get('JUDGE_API_KEY')
        or os.environ.get('OPENAI_API_KEY')
        or ''
    )
    model = (
        getattr(spec, 'judge_model', '')
        or os.environ.get('JUDGE_MODEL')
        or ''
    )
    return base_url.rstrip('/'), api_key, model


def judge_configured(cfg: Tuple[str, str, str]) -> bool:
    base_url, api_key, model = cfg
    return bool(base_url and api_key and model)


# ----------------------------------------------------------------------
# 核心：单条 yes/no 判定
# ----------------------------------------------------------------------
_YES_TOKENS = ('yes', 'true', '1', 'pass', 'correct', '满足', '通过')
_NO_TOKENS = ('no', 'false', '0', 'fail', 'incorrect', '不满足', '未通过')


def _parse_yes_no(text: str) -> Optional[bool]:
    """把判官输出解析为 bool；无法判定返回 None。"""
    if not text:
        return None
    t = text.strip().lower()
    # 取首行/首词，避免判官啰嗦
    head = t.splitlines()[0].strip() if t else t
    for tok in _YES_TOKENS:
        if head.startswith(tok):
            return True
    for tok in _NO_TOKENS:
        if head.startswith(tok):
            return False
    # 兜底：整句包含
    if any(tok in t for tok in _YES_TOKENS) and not any(tok in t for tok in _NO_TOKENS):
        return True
    if any(tok in t for tok in _NO_TOKENS) and not any(tok in t for tok in _YES_TOKENS):
        return False
    return None


def _build_messages(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': ('You are a strict, precise rubric judge. Read the requirement and the '
                        'answer carefully. Reply with ONLY a single word: "yes" or "no". '
                        'Do not explain.'),
        },
        {
            'role': 'user',
            'content': (
                f'Requirement:\n{question}\n\n'
                f'Material (problem + answer + reference anchor):\n{context}\n\n'
                f'Does the answer fully and correctly satisfy the requirement? '
                f'Answer yes or no.'
            ),
        },
    ]


async def _chat_once(
    client: 'httpx.AsyncClient',
    *,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    context: str,
    timeout: float,
) -> str:
    """单次 chat/completions，返回模型文本内容。失败抛 httpx 异常 / JudgeError。"""
    if httpx is None:
        raise JudgeError('httpx 未安装，无法调用判官')
    url = f'{base_url}/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload: Dict[str, Any] = {
        'model': model,
        'messages': _build_messages(question, context),
        'temperature': 0.0,
        'max_tokens': 8,
    }
    resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data['choices'][0]['message']['content'] or ''
    except (KeyError, IndexError, TypeError) as e:
        raise JudgeError(f'判官响应结构异常: {e!r}; body={str(data)[:300]}')


async def judge_yes_no(
    question: str,
    context: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
    retries: int = 3,
) -> Tuple[bool, str]:
    """对单条 rubric 做 yes/no 判定。

    Returns:
        (passed, raw_text)：passed=True 表示判为 yes/满足。
    Raises:
        JudgeError：未配置或重试后仍无有效裁决 -> 调用方置 is_infra。
    """
    if not (base_url and api_key and model):
        raise JudgeError('判官未配置（base_url/api_key/model 缺一不可）')
    if httpx is None:
        raise JudgeError('httpx 未安装')

    last_err: Optional[Exception] = None
    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                text = await _chat_once(
                    client, base_url=base_url, api_key=api_key, model=model,
                    question=question, context=context, timeout=timeout,
                )
                verdict = _parse_yes_no(text)
                if verdict is not None:
                    return verdict, text.strip()
                last_err = JudgeError(f'无法解析判官输出: {text[:200]!r}')
            except JudgeError as e:
                last_err = e
            except Exception as e:  # httpx 超时 / 连接 / 5xx 等
                last_err = e
            # 指数退避后重试
            if attempt < retries - 1:
                await asyncio.sleep(min(2 ** attempt, 5))
    raise JudgeError(f'判官重试 {retries} 次仍失败: {last_err!r}')


# ----------------------------------------------------------------------
# 批量：并发判多条 rubric
# ----------------------------------------------------------------------
async def judge_batch(
    items: List[Tuple[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
    max_concurrency: int = 5,
) -> List[Tuple[bool, str]]:
    """并发判定多条 (question, context)。任一条基建失败则整体抛 JudgeError（视为本任务 infra）。"""
    if not items:
        return []
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(q: str, c: str) -> Tuple[bool, str]:
        async with sem:
            return await judge_yes_no(
                q, c, base_url=base_url, api_key=api_key, model=model, timeout=timeout,
            )

    return await asyncio.gather(*[_one(q, c) for q, c in items])


if __name__ == '__main__':  # 简单冒烟（无真实 endpoint 时应抛 JudgeError）
    async def _main():
        cfg = resolve_judge_config(object())  # 无 spec 属性、无 env -> 空
        print('judge_configured(empty):', judge_configured(cfg))
        try:
            await judge_yes_no('q', 'ctx', base_url='', api_key='', model='')
        except JudgeError as e:
            print('expected JudgeError:', e)
    asyncio.run(_main())
