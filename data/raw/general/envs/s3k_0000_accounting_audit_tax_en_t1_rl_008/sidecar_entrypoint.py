#!/usr/bin/env python3
"""Sidecar container entrypoint for MCP-isolated rollouts.

Starts one HTTP MCP server per tools file, exposes a health+dump+verify
endpoint, and supervises child processes.

Expects tools/ and system/ to already be extracted at /work (via harbor
[[uploads]] + [preprocess]).

Listens on:
    :39100  — Health/management (GET /health, GET /dump, POST /verify)
    :39101+ — One MCP HTTP server per tools file (sorted alphabetically)
"""

from __future__ import annotations

import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HEALTH_PORT = 39100
MCP_PORT_BASE = 39101
WORK_DIR = Path("/work")
TOOLS_DIR = WORK_DIR / "tools"
SYSTEM_DIR = WORK_DIR / "system"
VENV_PYTHON = "/opt/openai-agents-venv/bin/python"
MCP_HTTP_SCRIPT = "/installed-agent/mcp_http.py"


# ─────────────────────────── MCP server management ───────────────────────────

def enumerate_tools() -> list[str]:
    """Return sorted tool file stems (matches prepare_task._enumerate_mcps)."""
    if not TOOLS_DIR.is_dir():
        return []
    return sorted(
        p.stem for p in TOOLS_DIR.iterdir()
        if p.is_file() and p.suffix == ".py"
        and not p.name.startswith("_") and p.name != "tools_test.py"
    )


def start_mcp_server(tools_stem: str, port: int) -> subprocess.Popen:
    """Launch one mcp_http.py subprocess."""
    tools_path = str(TOOLS_DIR / f"{tools_stem}.py")
    cmd = [VENV_PYTHON, MCP_HTTP_SCRIPT,
           "--tools-path", tools_path,
           "--server-name", tools_stem,
           "--port", str(port)]
    print(f"[sidecar] starting MCP: {tools_stem} on :{port}", file=sys.stderr)
    return subprocess.Popen(
        cmd, stdout=sys.stderr, stderr=sys.stderr,
    )


def check_port_ready(port: int, timeout: float = 1.0) -> bool:
    """Readiness check: is ``port`` in LISTEN state?

    Reads /proc/net/tcp{,6} rather than opening a TCP connection. A connect
    would emit a packet on the loopback OUTPUT chain, which — when the pod runs
    the MCP-isolation iptables uid-owner firewall — is REJECTed for any process
    that isn't the bridge uid (this entrypoint runs as root). Parsing the LISTEN
    state emits no packet, so it works regardless of the firewall while still
    confirming the server is up. ``timeout`` kept for signature compatibility.
    """
    hexport = f"{port:04X}"
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                next(f, None)  # header
                for line in f:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    local, state = parts[1], parts[3]
                    # state 0A = TCP_LISTEN; local is ADDR:PORT in hex
                    if state == "0A" and local.rsplit(":", 1)[-1].upper() == hexport:
                        return True
        except (OSError, StopIteration):
            continue
    return False


def wait_all_ready(ports: list[int], max_wait: float = 120.0) -> bool:
    """Block until all ports are accepting connections."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if all(check_port_ready(p) for p in ports):
            return True
        time.sleep(0.5)
    return False


# ─────────────────────────── Verify execution ───────────────────────────

# ─────────────────────────── Health server ───────────────────────────

class _HealthHandler(http.server.BaseHTTPRequestHandler):
    mcp_ports: list[int] = []

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            ok = all(check_port_ready(p, timeout=0.5) for p in self.mcp_ports)
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"ok": ok, "ports": self.mcp_ports})
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):  # noqa: A002
        pass


def start_health_server(ports: list[int]) -> http.server.HTTPServer:
    """Start the health+dump HTTP server on HEALTH_PORT in a daemon thread."""
    _HealthHandler.mcp_ports = ports
    server = http.server.HTTPServer(("0.0.0.0", HEALTH_PORT), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[sidecar] health server on :{HEALTH_PORT}", file=sys.stderr)
    return server


# ─────────────────────────── Supervisor ───────────────────────────

def supervise(processes: dict[str, tuple[subprocess.Popen, int]]) -> None:
    """Restart crashed MCP server processes. Runs forever."""
    while True:
        time.sleep(5)
        for name, (proc, port) in list(processes.items()):
            if proc.poll() is not None:
                print(f"[sidecar] {name} (port {port}) exited with "
                      f"rc={proc.returncode}, restarting ...", file=sys.stderr)
                new_proc = start_mcp_server(name, port)
                processes[name] = (new_proc, port)


# ─────────────────────────── Main ───────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-and-detach", action="store_true",
                        help="Start servers, wait until healthy, fork supervisor, exit 0 (for preprocess)")
    args = parser.parse_args()

    # Enumerate tools and assign ports
    tools = enumerate_tools()
    if not tools:
        print("[sidecar] FATAL: no tools found in /work/tools/", file=sys.stderr)
        sys.exit(1)

    port_map = {name: MCP_PORT_BASE + i for i, name in enumerate(tools)}
    ports = list(port_map.values())
    print(f"[sidecar] tools={len(tools)}, ports={ports}", file=sys.stderr)

    # Start MCP HTTP servers
    processes: dict[str, tuple[subprocess.Popen, int]] = {}
    for name, port in port_map.items():
        proc = start_mcp_server(name, port)
        processes[name] = (proc, port)

    # Wait for all servers to be ready
    print("[sidecar] waiting for MCP servers to accept connections ...",
          file=sys.stderr)
    if not wait_all_ready(ports, max_wait=120.0):
        failed = [p for p in ports if not check_port_ready(p)]
        print(f"[sidecar] FATAL: servers not ready after 120s, "
              f"failed ports: {failed}", file=sys.stderr)
        sys.exit(1)
    print("[sidecar] all MCP servers ready", file=sys.stderr)

    # Start health server
    try:
        start_health_server(ports)
    except OSError as e:
        print(f"[sidecar] WARNING: health server failed to start: {e}",
              file=sys.stderr)

    # --start-and-detach: fork supervisor into background and exit 0
    if args.start_and_detach:
        pid = os.fork()
        if pid > 0:
            # Parent: exit 0 (preprocess succeeds)
            print(f"[sidecar] servers ready, supervisor forked (pid={pid}), exiting",
                  file=sys.stderr)
            sys.exit(0)
        else:
            # Child: become the supervisor daemon
            os.setsid()
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            supervise(processes)
            return

    # --start-servers or legacy: block in foreground
    signal.signal(signal.SIGTERM, lambda *_: (
        [proc.terminate() for _, (proc, _) in processes.items()],
        sys.exit(0)
    ))
    supervise(processes)


if __name__ == "__main__":
    main()
