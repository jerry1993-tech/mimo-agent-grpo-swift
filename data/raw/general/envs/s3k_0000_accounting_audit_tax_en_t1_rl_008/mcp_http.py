#!/usr/bin/env python3
"""HTTP MCP server exposing ONE ``tools/<mcp>.py`` file via streamable-http.

Same module-loading logic as mcp_stdio.py, but serves over HTTP instead of
stdio.  Used by the sidecar container to expose tools to the main container
over localhost.

Usage (inside the sidecar container):
    python mcp_http.py \
        --tools-path /work/tools/riskgauge.py \
        --server-name riskgauge \
        --port 9101

The server listens on 0.0.0.0:{port} and exposes the MCP endpoint at /mcp
(FastMCP default).  The agent connects via http://localhost:{port}/mcp.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import re
import sys
import textwrap
import types
import typing
from pathlib import Path
from typing import Any, get_args, get_origin


# 从函数体的 isinstance(param, T) 反推每个形参的期望类型(仅当无显式注解/默认值推不出时兜底)。
# db_gen 生成的工具常无参数注解但用 isinstance(int/bool/list) 严格校验 → 若不据此定 schema,
# FastMCP 会把无注解参数当 string,与函数体冲突(MCP 边界死结)。函数体 isinstance 是"body 到底收什么"
# 的权威来源,故据它广告 schema 最保险。优先级:int>float>list>dict>str>bool(bool 置后:
# isinstance(x,bool) or not isinstance(x,int) 这种"拒 bool 收 int"应判 int)。
_ISINST_PY = {"int": int, "bool": bool, "float": float, "str": str, "list": list, "dict": dict}
_ISINST_PRIO = ["int", "float", "list", "dict", "str", "bool"]


def _body_isinstance_types(fn) -> dict:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:
        return {}
    fname = getattr(fn, "__name__", None)
    fnode = next((n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fname), None)
    if fnode is None:
        return {}
    params = {a.arg for a in list(fnode.args.args) + list(fnode.args.kwonlyargs)}
    raw: dict = {}
    for n in ast.walk(fnode):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "isinstance" and len(n.args) == 2):
            tgt = n.args[0]
            if isinstance(tgt, ast.Name) and tgt.id in params:
                elts = n.args[1].elts if isinstance(n.args[1], ast.Tuple) else [n.args[1]]
                for t in elts:
                    if isinstance(t, ast.Name) and t.id in _ISINST_PY:
                        raw.setdefault(tgt.id, set()).add(t.id)
    return {p: _ISINST_PY[next(t for t in _ISINST_PRIO if t in s)]
            for p, s in raw.items() if any(t in _ISINST_PRIO for t in s)}


# ─────────────────────────── tool-result size cap ───────────────────────────
# A single tool call must not be able to flood the agent's context (and, with a
# chatty model that calls tools thousands of times, blow up the agent
# container's memory → OOM/exit 137). Cap the serialized result size; the
# verifier recomputes expected answers independently via SQL, so truncating
# what the *agent* sees never affects scoring.
MAX_RESULT_CHARS = int(os.environ.get("GA_MCP_MAX_RESULT_CHARS", "200000"))


def _cap_result(result: Any) -> Any:
    try:
        s = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return result
    if len(s) <= MAX_RESULT_CHARS:
        return result
    note = (f"result capped at {MAX_RESULT_CHARS} chars; narrow with "
            "filters / date range / limit args")
    if isinstance(result, list):
        kept: list[Any] = []
        total = 0
        for item in result:
            js = json.dumps(item, ensure_ascii=False, default=str)
            if total + len(js) > MAX_RESULT_CHARS:
                break
            kept.append(item)
            total += len(js)
        kept.append({"_truncated": True, "_returned": len(kept),
                     "_total": len(result), "_note": note})
        return kept
    return {"_truncated": True, "_total_chars": len(s),
            "_note": note, "_preview": s[:MAX_RESULT_CHARS]}


def _wrap_capped(fn):
    """Wrap a tool fn so its return value is size-capped, preserving the
    original signature so FastMCP still derives the correct input schema.

    We deliberately do NOT use functools.wraps: it sets ``__wrapped__`` to the
    original fn, and some FastMCP versions ``inspect.unwrap()`` before calling —
    which would run the *uncapped* original and silently bypass the cap.

    Annotations are RESOLVED to concrete typing objects here (not copied as the
    PEP 563 strings the tool module carries under ``from __future__ import
    annotations``). Reason: FastMCP builds a pydantic arg-model from this
    wrapper's signature and, on the sidecar's mcp/pydantic (mcp 1.29 +
    pydantic 2.13), resolves any leftover forward-ref against the *model's*
    namespace at ``model_json_schema`` time — which lacks the tool's imports
    (bare ``Optional`` etc.) → ``PydanticUserError: ... is not fully defined``
    → the whole MCP server FATALs at startup. Resolving here, against the
    tool module's own globals PLUS all of ``typing`` as a fallback, means
    FastMCP only ever sees concrete types and never needs to resolve a name.
    If a tool has a genuinely unresolvable annotation we fall back to the raw
    value — no worse than before.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        sig = None
    if inspect.iscoroutinefunction(fn):
        async def wrapper(*args, **kwargs):
            return _cap_result(await fn(*args, **kwargs))
    else:
        def wrapper(*args, **kwargs):
            return _cap_result(fn(*args, **kwargs))
    wrapper.__name__ = getattr(fn, "__name__", "tool")
    wrapper.__qualname__ = getattr(fn, "__qualname__", wrapper.__name__)
    wrapper.__doc__ = fn.__doc__
    raw_ann = dict(getattr(fn, "__annotations__", {}))
    globalns = {**vars(typing), **getattr(fn, "__globals__", {})}
    try:
        resolved = typing.get_type_hints(fn, globalns=globalns)
    except Exception:
        resolved = {}
    # 关键修复:无类型标注的参数,按 注解 → 函数体 isinstance → 默认值 推一个具体类型注入。
    # 否则 FastMCP 把无标注参数当 string 广告出去,而函数体常 isinstance(int/bool/list) 严格校验 →
    # agent 传 int/bool 被 schema 拒、传 string 被函数体拒(死结,只在 MCP 边界现形,单测直调发现不了)。
    body_types = _body_isinstance_types(fn)

    def _infer(param):
        a = resolved.get(param.name, raw_ann.get(param.name))
        if a not in (None, inspect.Parameter.empty):
            return a
        if param.name in body_types:                 # 函数体 isinstance 是"body 收什么"的权威来源
            return body_types[param.name]
        d = param.default
        if d not in (inspect.Parameter.empty, None):
            for t in (bool, int, float, str):        # bool 先于 int(isinstance(True,int) 为真)
                if type(d) is t:
                    return t
        return None
    ann, params2 = {}, []
    for p in (sig.parameters.values() if sig is not None else []):
        if p.name in ("self", "cls"):
            params2.append(p); continue
        t = _infer(p)
        if t is not None:
            ann[p.name] = t
            params2.append(p.replace(annotation=t))
        else:
            params2.append(p)
    wrapper.__annotations__ = ann
    if sig is not None:
        wrapper.__signature__ = sig.replace(parameters=params2)
    # NB: no __wrapped__ set — keeps FastMCP from unwrapping to the original.
    return wrapper


# ─────────────────────────── tools/<mcp>.py loading ───────────────────────────

def _load_tools_module(tools_path: Path):
    mod_name = f"env_tools_{tools_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, tools_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tools module from {tools_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(tools_path.resolve().parent))
    spec.loader.exec_module(module)
    return module


def _public_functions(module) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        out.append((name, obj))
    out.sort(key=lambda kv: kv[0])
    return out


# ─────────────────────────── type → JSON schema ───────────────────────────

_PRIMITIVE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    # typing.Union[X, None] 与 PEP-604 的 X | None(types.UnionType)都要解包(否则 Optional 参数落成 string)
    if origin is typing.Union or origin is getattr(types, "UnionType", None):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _annotation_to_schema(args[0])
        return {}
    if annotation in _PRIMITIVE:
        return {"type": _PRIMITIVE[annotation]}
    if origin in (list, set, tuple) or annotation in (list, set, tuple):
        item_args = get_args(annotation)
        items = _annotation_to_schema(item_args[0]) if item_args else {}
        return {"type": "array", "items": items or {}}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    return {}


def _schema_from_default(default: Any) -> dict[str, Any]:
    """无类型标注时从默认值推 JSON schema 类型(bool 先于 int:isinstance(True,int) 为真)。"""
    if default is inspect.Parameter.empty or default is None:
        return {}
    if isinstance(default, bool):
        return {"type": "boolean"}
    if isinstance(default, int):
        return {"type": "integer"}
    if isinstance(default, float):
        return {"type": "number"}
    if isinstance(default, str):
        return {"type": "string"}
    if isinstance(default, (list, tuple, set)):
        return {"type": "array"}
    if isinstance(default, dict):
        return {"type": "object"}
    return {}


_DOC_TYPE = {
    "str": "string", "string": "string", "text": "string", "date": "string",
    "datetime": "string", "iso": "string", "uuid": "string", "id": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "decimal": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "sequence": "array",
    "dict": "object", "object": "object", "json": "object", "mapping": "object",
}


def _doc_param_types(doc: str | None) -> dict[str, str]:
    """从 docstring 的 Args/Parameters 段推每个参数类型。两级:
    ① 显式 `name (type):`(Google 风带类型);② `name: <描述>` → 扫描描述里的类型线索关键词
    (viewer 展示用的启发式,非严格契约)。优先级:boolean>array>object>integer>number>string。"""
    if not doc:
        return {}
    # 描述关键词 → 类型(按优先级顺序判定)
    kw_order = [
        ("boolean", r"\b(whether|true/false|boolean|\bflag\b|enable[sd]?|toggle)\b"),
        ("array", r"\b(list|array|sequence|multiple|\bids\b|comma-separated)\b|\[\]"),
        ("object", r"\b(dict|object|json|mapping|key-value)\b"),
        ("integer", r"\b(count|number of|integer|\blimit\b|offset|\bindex\b|how many|page size|top ?-?n)\b"),
        ("number", r"\b(amount|ratio|percentage|percent|basis points|decimal|float|\brate\b|price|weight|balance|_wan\b|quantity)\b"),
        ("string", r"\b(date|datetime|timestamp|yyyy|identifier|\bid\b|\bcode\b|\bname\b|string|text|format|status|scope|category|currency|period)\b"),
    ]
    out: dict[str, str] = {}
    for line in doc.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*\(([^)]+)\)\s*:", line)   # ① name (type):
        if m:
            raw = m.group(2).lower()
            for kw, jt in _DOC_TYPE.items():
                if kw in raw:
                    out[m.group(1)] = jt
                    break
            continue
        m = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*(\S.*)$", line)        # ② name: 描述
        if not m:
            continue
        name, desc = m.group(1), m.group(2).lower()
        for jt, pat in kw_order:
            if re.search(pat, desc):
                out[name] = jt
                break
    return out


def _build_input_schema(fn) -> dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    doc_types = _doc_param_types(fn.__doc__)   # 无标注时的兜底类型源
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                           inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, param.annotation)
        schema = _annotation_to_schema(annotation)
        if not schema.get("type"):                       # 标注缺失 → ① 从默认值推
            schema = _schema_from_default(param.default) or schema
        if not schema.get("type") and pname in doc_types:  # ② 从 docstring 兜底
            schema = {"type": doc_types[pname]}
        properties[pname] = schema or {}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    out["additionalProperties"] = False
    return out


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


# ─────────────────────────── FastMCP HTTP server ───────────────────────────

def serve(server_name: str, tools_path: Path, port: int) -> None:
    from mcp.server.fastmcp import FastMCP

    module = _load_tools_module(tools_path)
    fns = _public_functions(module)
    if not fns:
        print(f"FATAL: no public tool functions found in {tools_path}",
              file=sys.stderr)
        sys.exit(1)

    mcp = FastMCP(server_name, host="0.0.0.0", port=port)
    for name, fn in fns:
        mcp.add_tool(_wrap_capped(fn), name=name,
                     description=_first_line(fn.__doc__) or name)

    print(f"[mcp_http] server={server_name} tools={len(fns)} port={port} "
          f"src={tools_path}", file=sys.stderr)
    mcp.run(transport="streamable-http")


# ─────────────────────────── CLI ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve one env tools/<mcp>.py file as an HTTP MCP server"
    )
    parser.add_argument(
        "--tools-path", required=True,
        help="path to a single tools/<mcp>.py file",
    )
    parser.add_argument(
        "--server-name", default=None,
        help="MCP server name (defaults to file stem)",
    )
    parser.add_argument(
        "--port", type=int, required=True,
        help="HTTP port to listen on",
    )
    args = parser.parse_args()

    tools_path = Path(args.tools_path).resolve()
    if not tools_path.exists():
        print(f"FATAL: tools file not found at {tools_path}", file=sys.stderr)
        sys.exit(1)

    server_name = args.server_name or tools_path.stem
    serve(server_name, tools_path, args.port)


if __name__ == "__main__":
    main()
