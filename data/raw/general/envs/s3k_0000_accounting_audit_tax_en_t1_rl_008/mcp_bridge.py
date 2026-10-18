#!/usr/bin/env python3
"""HTTP MCP bridge — two modes.

  1. stdio bridge (default): a stdio-only MCP client ←stdio→ this
     bridge ←HTTP→ sidecar MCP server. One long-lived process per namespace.

         python mcp_bridge.py --url http://127.0.0.1:39101/mcp --name riskgauge

  2. one-shot (``--list`` / ``--call``): connect, do a single MCP op, print the
     result as one marker-prefixed JSON line to stdout, exit. This is how the
     out-of-pod agent runner reaches the sidecar: it tunnels each op
     through KubernetesEnvironment.execute() into the main container, which
     dials 127.0.0.1. Stateless — the agent's MCP client is the caller.

         python mcp_bridge.py --url http://127.0.0.1:39101/mcp --name riskgauge --list
         python mcp_bridge.py --url ... --name riskgauge --call get_x --args-json '{"a":1}'

The one-shot result is emitted as a SINGLE line ``__MCP_ONESHOT__<compact json>``
so the caller can recover it even when import warnings or other noise share the
exec stream (k8s exec merges stdout+stderr).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

ONESHOT_MARKER = "__MCP_ONESHOT__"


async def run_bridge(url: str, server_name: str) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent

    async with streamablehttp_client(url, timeout=120) as (read_remote, write_remote, _):
        async with ClientSession(read_remote, write_remote) as client:
            await client.initialize()

            tools_resp = await client.list_tools()
            remote_tools = tools_resp.tools

            srv = Server(server_name)

            @srv.list_tools()
            async def list_tools():
                return remote_tools

            @srv.call_tool()
            async def call_tool(name: str, arguments: dict | None):
                try:
                    result = await client.call_tool(name, arguments or {})
                    if result.content:
                        return result.content
                    return [TextContent(type="text", text="(empty result)")]
                except Exception as exc:
                    err = json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    )
                    return [TextContent(type="text", text=err)]

            async with stdio_server() as (read_local, write_local):
                await srv.run(
                    read_local,
                    write_local,
                    srv.create_initialization_options(),
                )


def _emit(payload: dict) -> None:
    """Emit the one-shot result as a single marker-prefixed JSON line."""
    print(ONESHOT_MARKER + json.dumps(payload, ensure_ascii=False), flush=True)


def _content_to_text(result) -> str:
    """Flatten an MCP CallToolResult's content blocks to a single text string."""
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            data = getattr(block, "data", None)
            parts.append(data if isinstance(data, str) else str(block))
    return "\n".join(parts)


async def run_list(url: str) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, timeout=120) as (r, w, _):
        async with ClientSession(r, w) as client:
            await client.initialize()
            resp = await client.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema or {"type": "object", "properties": {}},
                }
                for t in resp.tools
            ]
    _emit({"ok": True, "tools": tools})


async def run_call(url: str, fn: str, arguments: dict) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, timeout=120) as (r, w, _):
        async with ClientSession(r, w) as client:
            await client.initialize()
            result = await client.call_tool(fn, arguments or {})
    payload = {"ok": True, "is_error": bool(getattr(result, "isError", False)),
               "content": _content_to_text(result)}
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        payload["structured"] = structured
    _emit(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP MCP bridge (stdio + one-shot)")
    parser.add_argument("--url", default=None, help="Remote HTTP MCP endpoint URL")
    parser.add_argument("--name", default="mcp", help="Server name to advertise (stdio mode)")
    parser.add_argument("--list", action="store_true", help="one-shot: list tools, print JSON, exit")
    parser.add_argument("--call", default=None, help="one-shot: call this tool, print JSON, exit")
    parser.add_argument("--args-json", default="{}", help="JSON arguments for --call")
    args = parser.parse_args()

    url = os.environ.get("MCP_ENDPOINT_URL") or args.url
    if not url:
        parser.error("no endpoint: set MCP_ENDPOINT_URL or pass --url")

    try:
        if args.list:
            asyncio.run(run_list(url))
        elif args.call is not None:
            try:
                arguments = json.loads(args.args_json or "{}")
            except json.JSONDecodeError as e:
                _emit({"ok": False, "error": f"bad --args-json: {e}"})
                return
            asyncio.run(run_call(url, args.call, arguments))
        else:
            asyncio.run(run_bridge(url, args.name))
    except Exception as exc:
        if args.list or args.call is not None:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            raise


if __name__ == "__main__":
    main()
