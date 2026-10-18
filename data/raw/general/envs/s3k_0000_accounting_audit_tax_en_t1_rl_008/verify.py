#!/usr/bin/env python3
"""verify — 客观分。两种运行环境自适应:
  本地(PYTHONPATH=src,有 office_gen)→ grade_rubric.grade(rule+llm 完整)。
  harbor sidecar(无 office_gen,仅本目录 _helpers.py + verifier_meta.json)→ 自包含 rule 判分(det)。
入口:CLI  python verify.py --workspace <ws>   /   函数 verify(env_path=...)(harbor run_verify.py 调)。"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent


def _meta():
    return json.loads((HERE / "verifier_meta.json").read_text(encoding="utf-8"))


def _run_rule_sc(check_code, fn, ws, timeout=300):
    """sidecar 无 office_gen:_helpers.py(=task_helpers,含 _num 等全部辅助,不做名字过滤)
    + check_code 拼进临时脚本跑 fn(ws)。ws.parent/system/<mcp>/state.db 即 sidecar 的 DB。"""
    helpers = (HERE / "_helpers.py").read_text(encoding="utf-8") if (HERE / "_helpers.py").exists() else ""
    if "import random" in check_code or "random." in check_code:
        return {"score": 0.0, "detail": "含 random 禁用"}
    raw = ("from pathlib import Path\n" + helpers + "\n" + check_code + "\n"
           + "import json as _j, sys as _s\n"
           + "print('___R___' + _j.dumps(" + fn + "(Path(_s.argv[1])), ensure_ascii=False, default=str))\n")
    # `from __future__ import ...` 必须是文件首个语句;helpers/check_code 里若带上它(常见
    # annotations),拼在 pathlib 之后会 SyntaxError → 整项判 0。统一提到最顶(去重保序)。
    _ls = raw.split("\n")
    _fut = list(dict.fromkeys(l for l in _ls if l.lstrip().startswith("from __future__ import")))
    _rest = [l for l in _ls if not l.lstrip().startswith("from __future__ import")]
    src = "\n".join(_fut + _rest)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src); p = f.name
    try:
        r = subprocess.run([sys.executable, p, str(ws)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "detail": "rule " + fn + " 超时"}
    finally:
        Path(p).unlink(missing_ok=True)
    out = ""
    for line in (r.stdout or "").splitlines():
        if line.startswith("___R___"):
            out = line[len("___R___"):]
    if not out:
        return {"score": 0.0, "detail": "rule " + fn + " 无输出: " + (r.stderr or "")[-200:]}
    try:
        d = json.loads(out)
        return {"score": float(d.get("score", 0)), "detail": str(d.get("detail", ""))[:200]}
    except Exception as e:
        return {"score": 0.0, "detail": "rule " + fn + " 输出非法: " + str(e)}


# ── LLM 判分项(sidecar 自包含,judge 多级 fallback 链)──
# 链全部来自 env(GA_JUDGE_KEY/_KEY2/_KEY3 + 同后缀 MODEL/API/URL),key 不落盘;
# controller 从自身的 judge 通道配置注入并只在
# sidecar exec 前缀上透传。api 取值:
#   "responses" OpenAI Responses 流式(任意 OpenAI-compatible endpoint)
#   "chat"      OpenAI /chat/completions 非流式(sglang/vllm 本地部署等 OpenAI 兼容端点)
#   "gemini"    原生 v1beta generateContent —— Gemini-native endpoint 只认原生
#               wire,OpenAI 兼容层对该组 500 "contents is required"
# 必须通过环境变量 GA_JUDGE_URL 指定 judge endpoint (未设则 judge 调用会失败)
_JUDGE_URL = os.environ.get("GA_JUDGE_URL", "")
_JUDGE_EFFORT = os.environ.get("GA_JUDGE_EFFORT", "high")


def _judge_chain_sc():
    chain = []
    for suf in ("", "2", "3"):
        k = os.environ.get("GA_JUDGE_KEY" + suf, "")
        if not k or any(e["key"] == k for e in chain):
            continue
        chain.append({"key": k,
                      "model": os.environ.get("GA_JUDGE_MODEL" + suf, "gpt-4o-mini"),
                      "api": os.environ.get("GA_JUDGE_API" + suf, "responses"),
                      "url": os.environ.get("GA_JUDGE_URL" + suf, "") or None})
    # 负载均分(GA_JUDGE_ROTATE=0 关):每个 verify 随机选链上起点,全池流量按 key
    # 均分(router 队列各减半);失败仍按粘性降级切到下一把。同模型多 key 时
    # 起点不影响判分结果,只影响走哪把 key。
    if len(chain) > 1 and os.environ.get("GA_JUDGE_ROTATE", "1") != "0":
        import random as _rnd
        k = _rnd.randrange(len(chain))
        chain = chain[k:] + chain[:k]
    return chain


_JUDGE_CHAIN = _judge_chain_sc()

_LLM_PROMPT = """你在核验一份工作交付物是否满足一个**客观检查点**(有唯一对错,不是审美)。
下面给出交付物相关文件的抽取文本,以及检查点。严格判断:满足=1,不满足=0。
**只判 0 或 1,不给部分分**(部分满足=不满足)。**只依据给出的文本证据**,看不到证据就判 0;不要臆测。
证据文本可能含【已截断…】标记或「⚠ 未抽取内容」清单:若检查点的目标内容恰好落在被截断/
未抽取的部分,判 0,并在 why 里写明「证据截断」或「证据未抽取」(供审计区分,仍不要臆测给分)。

【交付物文本】
{evidence}

【检查点】
{items}

只输出一个 json 对象(可包在 json 代码块):
{"results": {"检查点id": {"score": 0, "why": "≤25字依据"}}}"""


def _extract_json_sc(text):
    if not text or "{" not in text:
        return None
    frag = text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(frag)
    except Exception:
        return None



class _JudgeUnavailable(Exception):
    """judge 传输在预算内拿不到任何可用返回 —— 上层应 mask,不是记 0。"""


_JUDGE_DEADLINE = [0.0]   # 首次调用时 = now + GA_JUDGE_PHASE_BUDGET;全部 llm 项共享
_JUDGE_ACTIVE = [0]       # fallback 链上当前入口(粘性:跨 item 不回退)
_JUDGE_FAILS = {}         # 入口 idx -> 连续传输失败计数


def _judge_left_sc():
    import time as _rt
    if _JUDGE_DEADLINE[0] <= 0:
        _JUDGE_DEADLINE[0] = _rt.time() + float(os.environ.get("GA_JUDGE_PHASE_BUDGET", "600"))
    return _JUDGE_DEADLINE[0] - _rt.time()


def _chat_judge_sc(prompt, max_tokens=4000, timeout=None):
    """自包含调 judge,带 key/model 多级 fallback(20260904 排队友好版)。
      * 全 judge 阶段共享墙钟预算 GA_JUDGE_PHASE_BUDGET(默认 600s):router 抖动时
        verifier 在外层 exec 限时(900s)内自己收尾走 mask,而不是被掐死连
        reward.json 都不剩。
      * per-attempt 读超时 = min(500s, 剩余预算):judge 变慢多半是排队,掐死
        队列里的请求再重试 = 丢队位 + 双倍流量,是拥塞放大器 —— 在预算内等它。
      * 重试最多 4 次,退避 5→20s;入口连续失败 2 次(401/403/404/429/503 或
        读超时则 1 次)→ 粘性切下一把 key;链尾不弃;全部耗尽 → _JudgeUnavailable。
    有效返回里的 0 分是判分结论,不触发重试。"""
    if not _JUDGE_CHAIN:
        raise _JudgeUnavailable("no judge key")
    import time as _rt
    # 启动时打一次 judge 链信息(每个 verify 进程只打一次,不随重试重复)
    if not globals().get("_JUDGE_LOGGED"):
        globals()["_JUDGE_LOGGED"] = True
        for _i, _e in enumerate(_JUDGE_CHAIN):
            _k = _e.get("key", "")
            print("[judge] chain[%d]: model=%s api=%s key=%s..." % (
                _i, _e.get("model"), _e.get("api", "responses"), _k[:12]))
    last = ""
    for _att in range(4):
        left = _judge_left_sc()
        if left <= 10:
            raise _JudgeUnavailable("judge phase budget exhausted (attempt %d); last_err=%s"
                                    % (_att, last or "-"))
        if _att:
            _rt.sleep(min(20.0, 5.0 * 2 ** (_att - 1), max(1.0, left / 8)))
            left = _judge_left_sc()
        idx = _JUDGE_ACTIVE[0]
        ent = _JUDGE_CHAIN[idx]
        err = None
        try:
            out = _judge_once_sc(ent, prompt, max_tokens, min(500.0, max(30.0, left - 20.0)))
        except Exception as e:
            err = e
            out = ""
        if out and out.strip():
            _JUDGE_FAILS[idx] = 0
            return out
        if err is not None:
            code = getattr(getattr(err, "response", None), "status_code", None)
            hard = code in (401, 403, 404, 429, 503) or "Timeout" in type(err).__name__
            last = ("e%d/%s %s: %s" % (idx, ent.get("model"), type(err).__name__, err))[:200]
        else:
            hard = False
            last = "e%d/%s empty body" % (idx, ent.get("model"))
        _JUDGE_FAILS[idx] = _JUDGE_FAILS.get(idx, 0) + (2 if hard else 1)
        if (_JUDGE_FAILS[idx] >= 2 and _JUDGE_ACTIVE[0] == idx
                and idx + 1 < len(_JUDGE_CHAIN)):
            _nx = _JUDGE_CHAIN[idx + 1]
            _nk = _nx.get("key", "")
            print("[judge] switching e%d->e%d: model=%s api=%s key=%s... (last_err=%s)"
                  % (idx, idx + 1, _nx.get("model"), _nx.get("api", "responses"),
                     _nk[:12], (last or "")[:80]))
            _JUDGE_ACTIVE[0] = idx + 1
    raise _JudgeUnavailable("no judge response after 4 attempts; last_err=" + (last or "empty body"))


def _judge_once_sc(ent, prompt, max_tokens=4000, read_to=240.0):
    api = ent.get("api") or "responses"
    if api == "chat":
        return _judge_once_chat_sc(ent, prompt, max_tokens, read_to)
    if api == "gemini":
        return _judge_once_gemini_sc(ent, prompt, max_tokens, read_to)
    return _judge_once_responses_sc(ent, prompt, max_tokens, read_to)


def _judge_once_responses_sc(ent, prompt, max_tokens=4000, read_to=240.0):
    import requests
    url = (ent.get("url") or _JUDGE_URL).rstrip("/") + "/responses"
    payload = {"model": ent["model"], "input": prompt, "max_output_tokens": max_tokens,
               "reasoning": {"effort": _JUDGE_EFFORT}, "stream": True}
    deltas, full = [], None
    with requests.post(url, json=payload, timeout=(10, read_to),
                       headers={"Authorization": "Bearer " + ent["key"], "Content-Type": "application/json"},
                       proxies={"http": None, "https": None}, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", "ignore")
            if not s.startswith("data:"):
                continue
            data = s[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            t = obj.get("type", "")
            if t == "response.output_text.delta":
                deltas.append(obj.get("delta", ""))
            elif t in ("response.completed", "response.incomplete"):
                resp = obj.get("response") or {}
                if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
                    full = resp["output_text"]
                elif not full:
                    parts = [ct.get("text", "") for item in (resp.get("output") or [])
                             for ct in (item.get("content") or [])
                             if ct.get("type") in ("output_text", "text")]
                    if parts:
                        full = "".join(parts)
    return full or "".join(deltas)


def _judge_once_chat_sc(ent, prompt, max_tokens=4000, read_to=240.0):
    """OpenAI-compatible /chat/completions —— sglang/vllm 本地部署走这里(/responses 404)。
    非流式,temperature=0 判分可复现。"""
    import requests
    url = (ent.get("url") or _JUDGE_URL).rstrip("/") + "/chat/completions"
    payload = {"model": ent["model"], "max_tokens": max_tokens, "temperature": 0,
               "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(url, json=payload, timeout=(10, read_to),
                      headers={"Authorization": "Bearer " + ent["key"],
                               "Content-Type": "application/json"},
                      proxies={"http": None, "https": None})
    r.raise_for_status()
    msg = ((r.json().get("choices") or [{}])[0].get("message") or {})
    content = msg.get("content") or ""
    if not content.strip():
        content = msg.get("reasoning_content") or ""
    return content


def _judge_once_gemini_sc(ent, prompt, max_tokens=4000, read_to=240.0):
    """Gemini 原生 generateContent。Gemini-native endpoint 只透传原生 wire:base 去掉
    /v1 前缀,x-goog-api-key 认证;非流式,temperature=0。"""
    import requests
    base = (ent.get("url") or _JUDGE_URL).rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    url = base + "/v1beta/models/" + ent["model"] + ":generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0}}
    r = requests.post(url, json=payload, timeout=(10, read_to),
                      headers={"x-goog-api-key": ent["key"], "Content-Type": "application/json"},
                      proxies={"http": None, "https": None})
    r.raise_for_status()
    parts = [p.get("text", "") for c in (r.json().get("candidates") or [])
             for p in ((c.get("content") or {}).get("parts") or [])]
    return "".join(parts)


def _evidence_sc(ws, files, budget=20000):
    """复用已上传 sidecar 的 _helpers.py(=task_helpers):优先 extract_evidence(保文档序/
    表格坐标/显式截断/丢失清单),旧 helpers 无该函数则回退 extract_text。无静默截断。"""
    import importlib.util, re as _re
    try:
        spec = importlib.util.spec_from_file_location("_th_ev", str(HERE / "_helpers.py"))
        th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
    except Exception:
        return ""
    parts = []
    for f in files or []:
        p = ws / f
        if not p.exists():
            kws = [w for w in _re.split(r"[·\s_./]+", Path(f).stem) if len(w) >= 2]
            try:
                hit = th.locate_deliverable(ws, *kws)
            except Exception:
                hit = None
            p = hit or p
        if p.exists():
            try:
                if hasattr(th, "extract_evidence"):
                    ev = th.extract_evidence(p, budget=budget)
                    seg = "===== " + p.name + " =====\n" + ev["text"]
                    if ev.get("loss"):
                        seg += "\n⚠ 本文件未抽取内容: " + "; ".join(ev["loss"])
                    parts.append(seg)
                else:
                    parts.append("===== " + p.name + " =====\n" + th.extract_text(p)[:budget])
            except Exception as e:
                parts.append("===== " + p.name + " (抽取失败: " + str(e) + ") =====")
        else:
            parts.append("===== " + str(f) + " (文件缺失) =====")
    return "\n\n".join(parts)



def _run_llm_sc(items, ws, votes=1):
    """text-judge llm 项:**逐 item 独立调用**(判定完全隔离,与筛分口径一致;证据按 files
    组合缓存,同交付物不重复抽)。全局限流在 controller 侧判分闸(GA_JUDGE_GATE),不在
    这里。judge 拿不到任何可用 verdict → 抛 _JudgeUnavailable(由 _grade_sc 转 mask),
    绝不记 0。"""
    import statistics
    out = {}
    ev_cache = {}
    for it in items:
        key = tuple(sorted(it.get("files") or []))
        if key not in ev_cache:
            ev_cache[key] = _evidence_sc(ws, list(key))
        item_txt = ("- [" + it["id"] + "] " + str(it.get("question", it.get("desc", "")))
                    + "(满足=1:" + str(it.get("pass_anchor", "")) + ")")
        prompt = _LLM_PROMPT.replace("{evidence}", ev_cache[key]).replace("{items}", item_txt)
        vals = []
        for _ in range(max(1, votes)):
            j = None
            for _p in range(2):   # 解析感知重试:传输成功但 JSON 不可解析,再要一次
                j = _extract_json_sc(_chat_judge_sc(prompt))
                if j is not None:
                    break
            sc = (((j or {}).get("results") or {}).get(it["id"]) or {}).get("score")
            if isinstance(sc, (int, float)):
                vals.append(max(0.0, min(1.0, float(sc))))
        if not vals:
            raise _JudgeUnavailable("no usable verdict for item " + str(it.get("id")))
        out[it["id"]] = {"score": round(statistics.median(vals), 4),
                         "detail": "llm " + str(len(vals)) + "票"}
    return out


def _grade_sc(ws):
    """sidecar 自包含判分(赋分体系 v2):红线 gate(method=rule)先跑,任一挂 → score=0(llm 跳过);
    原子 rule 子进程 + text-llm 直调 judge 模型,Σw·s/Σw 归一(gate 不进分母,与 grade_meta 口径一致);
    agent 项与 judge=vision 项 sidecar 不跑、不进分母(pod 无渲染/vision 通道)。"""
    meta = _meta(); items = meta.get("items") or []; code = meta.get("check_code", "")
    results = {}
    gates = [it for it in items if it.get("gate")]
    atoms = [it for it in items if not it.get("gate")]
    gate_failed = []
    sp = meta.get("src_protect")
    if sp:   # 源守恒 gate(自动):solve 写足迹之外的文件/DB 表被动 → 整题 0(防篡改源底稿)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_th_fp", str(HERE / "_helpers.py"))
            th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
            bad = th.protect_diff(ws, sp) if hasattr(th, "protect_diff") else []
            note_sp = None
        except Exception as e:
            bad, note_sp = [], "源守恒 gate 执行失败,放行: " + str(e)
        if bad:
            gate_failed.append("src_protect")
            results["src_protect"] = {"score": 0.0, "detail": "源守恒 gate: " + "; ".join(bad[:5])}
        else:
            results["src_protect"] = {"score": 1.0, "detail": note_sp or "源守恒 gate 通过"}
    for it in gates:
        fn = it.get("fn")
        results[it["id"]] = _run_rule_sc(code, fn, ws) if fn else {"score": 0.0, "detail": "gate 缺 fn"}
        if float(results[it["id"]].get("score", 0.0) or 0.0) < 0.999:
            gate_failed.append(it["id"])
    llm_items = [it for it in atoms if it.get("method") == "llm"
                 and (it.get("judge") or "text") != "vision"]
    if llm_items and not gate_failed:
        try:
            results.update(_run_llm_sc(llm_items, ws, votes=int(os.environ.get("GRADE_VOTES", "1"))))
        except _JudgeUnavailable as e:
            for it in llm_items:
                results.setdefault(it["id"], {"score": 0.0, "detail": "judge_unavailable"})
            return {"score": 0.0, "passed": False, "reward_error": "judge_crashed",
                    "judge_error": str(e)[:300], "items": results,
                    "note": "sidecar self-contained: judge unavailable -> mask"}
    elif llm_items:
        for it in llm_items:
            results[it["id"]] = {"score": 0.0, "detail": "gate 未过(" + ",".join(gate_failed) + "),llm 跳过"}
    for it in atoms:
        if it.get("method") == "rule":   # rule 便宜,gate 挂也照跑供审计
            fn = it.get("fn")
            results[it["id"]] = _run_rule_sc(code, fn, ws) if fn else {"score": 0.0, "detail": "缺 fn"}
    wsum = tsum = 0.0; bm = {}; skipped = []; all_pass = True
    for it in atoms:
        mth = it.get("method", "?")
        if mth == "agent" or (mth == "llm" and (it.get("judge") or "text") == "vision"):
            skipped.append(it["id"]); continue
        w = float(it.get("weight", 1)); sc = float(results.get(it["id"], {}).get("score", 0.0) or 0.0)
        if sc < 0.999:
            all_pass = False
        wsum += w * sc; tsum += w
        b = bm.setdefault(mth, [0.0, 0.0]); b[0] += w; b[1] += w * sc
    sc = 0.0 if gate_failed else (round(wsum / tsum, 4) if tsum else 0.0)
    return {"score": sc, "passed": (not gate_failed) and all_pass and tsum > 0,
            "gate_failed": gate_failed,
            "by_method": {m: (round(v[1] / v[0], 4) if v[0] else None) for m, v in bm.items()},
            "items": {k: v for k, v in results.items() if k not in skipped},
            "note": "sidecar self-contained: rule+text-llm 计分" + (", agent/vision 跳过" if skipped else "")}


def _grade(ws):
    try:
        from synthesis.office_gen.grade_rubric import grade
    except Exception:
        return _grade_sc(ws)
    r = grade(str(HERE), str(ws), votes=int(os.environ.get("GRADE_VOTES", "1")),
              with_agent=os.environ.get("GRADE_AGENT") == "1")
    return {"score": r.get("score", 0.0), "passed": r.get("passed"),
            "gate_failed": r.get("gate_failed"), "by_method": r.get("by_method"),
            "items": r.get("items")}


def _materialize_answer(ws, agent_output):
    """text-answer 任务的最终回复在 chat 而非文件。harbor run_verify.py 已从 session 抽出
    agent_output 传进来;若 ws/answer.md 缺失或空而 agent_output 有内容,写入供 answer_report
    等项判分——不覆盖 agent 已产出的非空文件。避免依赖 after_agent 钩子的日志路径。"""
    if not agent_output or not str(agent_output).strip():
        return
    ans = ws / "answer.md"
    try:
        if ans.exists() and ans.stat().st_size > 0:
            return
        ws.mkdir(parents=True, exist_ok=True)
        ans.write_text(str(agent_output), encoding="utf-8")
    except Exception:
        pass


def _as_results(items):
    """把 _grade 的 items 拍平成 harbor run_verify 期望的 results 列表 [{id,passed,score,message}]。
    _grade_sc 给 dict{id:{score,detail}};grade_rubric 给 dict 或 list——都归一。reward 仍以加权
    score 为准,这里只喂 reward_detail 的逐项明细(否则 total=0/results=[])。"""
    out = []
    if isinstance(items, dict):
        for k, v in items.items():
            v = v if isinstance(v, dict) else {"score": v}
            sc = float(v.get("score", 0) or 0)
            out.append({"id": k, "score": sc, "passed": sc >= 0.999,
                        "message": str(v.get("detail") or v.get("message") or "")[:300]})
    elif isinstance(items, list):
        for v in items:
            if not isinstance(v, dict):
                continue
            sc = float(v.get("score", 0) or 0)
            out.append({"id": v.get("id") or v.get("fn") or "?", "score": sc,
                        "passed": bool(v.get("passed")) if "passed" in v else sc >= 0.999,
                        "message": str(v.get("detail") or v.get("message") or "")[:300]})
    return out


def verify(env_path="/work", post_state=None, agent_output="", **kw):
    """harbor run_verify.py 入口:ws=<env_path>/workspace(共享卷,含 answer.md + 交付物);
    DB 后态由 check_code 经 _helpers.db_path 读 ws.parent/system/<mcp>/state.db(sidecar 的 /work/system)。"""
    ws = Path(env_path) / "workspace"
    _materialize_answer(ws, agent_output)
    r = _grade(ws)
    sc = float(r.get("score", 0.0) or 0.0)
    out = {"reward": sc, "score": sc, "detail": r, "results": _as_results(r.get("items"))}
    if isinstance(r, dict) and r.get("reward_error"):
        out["reward_error"] = str(r["reward_error"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ws = Path(ap.parse_args().workspace)
    r = _grade(ws)
    print(json.dumps({"score": r.get("score", 0.0), "detail": r}, ensure_ascii=False))


if __name__ == "__main__":
    main()
