"""Music 域评判器：ABC 记谱格式/结构规则分（无沙箱，纯文本）。

小米完整版用 abc2midi 转 MIDI 后提取 18 个特征（rhythm/texture/tonal/register/acoustic/
structure 六组）与人类参考分布 p10–p90 对比，组权重 rhythm .22/texture .20/tonal .16/
register .16/acoustic .14/structure .12，总分 100×(0.85 加权 + 0.15 距离)。

本实现做【忠实的结构性子集】：不依赖 abc2midi 与参考分布，直接在 ABC 文本上校验
X/M/Q/L/K/V 字段与音符体，权重按小米组分做映射且和=1，并保留 ``_feature_score`` 钩子。
无 abc 内容 -> score=0（模型没产出，非 infra）。

参考：
- https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/design/music/scorer/score.py
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sandbox.base import SandboxBackend  # noqa: E402
from verifier.base import VerificationResult, Verifier, register_verifier  # noqa: E402

_ABC_FENCE_RE = re.compile(r'```abc\s*(.*?)```', re.IGNORECASE | re.DOTALL)
_FIELD_RE = re.compile(r'^\s*([A-Za-z])\s*:\s*(.*?)\s*$', re.MULTILINE)


def _extract_abc(text: str) -> str:
    """优先取 ```abc fenced 块；没有 fence 则把整段当 ABC。"""
    if not text:
        return ''
    m = _ABC_FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _fields(abc: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, val in _FIELD_RE.findall(abc or ''):
        out.setdefault(key.upper(), val)
    return out


def _parse_bpm(q: str) -> Optional[float]:
    """从 Q: 字段解析 BPM，容忍 '1/4=120' / '120' / 'C 120'。"""
    if not q:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*$', q.strip())
    return float(m.group(1)) if m else None


def _bar_count(abc: str) -> int:
    """粗略小节数：按行内 '|' 计数（去掉开头/结尾空竖线）。"""
    body = '\n'.join(l for l in (abc or '').splitlines() if not re.match(r'^[A-Za-z]:', l.strip()))
    return max(0, body.count('|') - 1)


# length -> 期望小节区间（近似）
_BAR_RANGE = {
    'short': (4, 64),
    'medium': (16, 128),
    'long': (32, 256),
}


@register_verifier('music')
class MusicVerifier(Verifier):
    domain = 'music'

    # 结构性子项权重（映射小米六组，和=1.0）
    W = {
        'format_valid': 0.22,   # 对应 rhythm/整体记谱正确性
        'meter_match': 0.20,     # 对应 texture/结构框架
        'tempo_match': 0.16,    # 对应 acoustic
        'voices_match': 0.16,    # 对应 register/声部规模
        'structure': 0.26,       # 对应 tonal+structure（L/K/音符体）
    }

    def score_text(self, abc_text: str) -> VerificationResult:
        """核心规则打分（同步）。abc_text 可为含 ```abc fence 的模型原始输出。"""
        meta = self.spec.music_meta or {}
        abc = _extract_abc(abc_text)
        if not abc.strip():
            return VerificationResult(
                score=0.0, passed=False, components={k: 0.0 for k in self.W},
                is_infra=False, message='未提取到 ABC 记谱内容（无 ```abc 块）',
            )

        f = _fields(abc)
        components: Dict[str, float] = {}

        # format_valid：X 与 K 齐全、含音符体（至少一行非头部含字母音符/休止）
        has_x = bool(f.get('X'))
        has_k = bool(f.get('K'))
        body_notes = bool(re.search(r'[a-gA-GzZ]', '\n'.join(
            l for l in abc.splitlines() if not re.match(r'^[A-Za-z]:', l.strip()))))
        components['format_valid'] = 1.0 if (has_x and has_k and body_notes) else 0.0

        # meter_match：M: 与 music_meta.meter 一致
        want_meter = str(meta.get('meter', '') or '').strip()
        got_meter = re.sub(r'\s+', '', f.get('M', '') or '')
        if not want_meter:
            components['meter_match'] = 1.0  # 任务未指定则不罚
        else:
            components['meter_match'] = 1.0 if got_meter == re.sub(r'\s+', '', want_meter) else 0.0

        # tempo_match：Q: BPM 在期望 ±10% 内
        want_bpm = meta.get('bpm')
        got_bpm = _parse_bpm(f.get('Q', ''))
        if not want_bpm:
            components['tempo_match'] = 1.0
        elif got_bpm is None:
            components['tempo_match'] = 0.0
        else:
            try:
                lo, hi = float(want_bpm) * 0.9, float(want_bpm) * 1.1
                components['tempo_match'] = 1.0 if lo <= got_bpm <= hi else 0.0
            except (TypeError, ValueError):
                components['tempo_match'] = 0.0

        # voices_match：V: 声部数 ~ nvoice_want（容忍 ±1）
        nvoice_want = meta.get('nvoice_want') or meta.get('nvoice')
        # 数 V: 出现次数（去重：V:1, V:2 ...）
        v_count = len(set(re.findall(r'(?m)^\s*V\s*:\s*([^\s]+)', abc)))
        if v_count == 0 and nvoice_want and int(nvoice_want) <= 1:
            v_count = 1  # 单声部常省略 V:
        if not nvoice_want:
            components['voices_match'] = 1.0
        else:
            components['voices_match'] = 1.0 if abs(v_count - int(nvoice_want)) <= 1 else 0.0

        # structure：L: 存在、K 调号合法、小节数落在 length 期望区间
        length = str(meta.get('length', '') or '').lower()
        bars = _bar_count(abc)
        lo, hi = _BAR_RANGE.get(length, (4, 256))
        l_ok = bool(f.get('L'))
        k_ok = bool(f.get('K'))
        bar_ok = lo <= bars <= hi
        components['structure'] = 1.0 if (l_ok and k_ok and bar_ok and bars >= 1) else 0.0

        score = sum(self.W[k] * components[k] for k in self.W)
        return VerificationResult(
            score=score, passed=score >= 0.6, components=components, is_infra=False,
            message=(f'ABC 结构分={score:.3f} (meter={got_meter} want={want_meter}; '
                     f'bpm={got_bpm} want={want_bpm}; voices={v_count} want={nvoice_want}; '
                     f'bars={bars} length={length})'),
        )

    def _feature_score(self, abc: str) -> float:
        """TODO(钩子)：完整版需 abc2midi -> MIDI -> 18 特征 vs 人类 p10-p90 分布，
        总分 100×(0.85 加权组分 + 0.15 直方图 JS 距离)。本结构性子集未实现，预留。"""
        raise NotImplementedError

    async def verify(self, backend: Optional[SandboxBackend]) -> VerificationResult:
        # music 通常无沙箱；仍兼容：env/submission 文件 -> 否则 spec.extra -> 否则空
        text = ''
        sub_file = (self.spec.extra.get('submission_file')
                    or self.spec.extra.get('abc_file') or '')
        if backend is not None and sub_file:
            try:
                text = await backend.read_file(sub_file)
            except Exception:
                text = ''
        if not text:
            text = self.spec.extra.get('abc_text') or self.spec.extra.get('answer') or ''
        return self.score_text(text)
