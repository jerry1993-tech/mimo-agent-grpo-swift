#!/usr/bin/env bash
# 启动“外接沙箱” HTTP 服务（sandbox/server.py）。
# 默认用 Docker 后端管理 per-task 容器；本机无 docker 时可 SANDBOX_BACKEND=local 演示。
set -euo pipefail
cd "$(dirname "$0")/.."

export SANDBOX_BACKEND="${SANDBOX_BACKEND:-docker}"
HOST="${SANDBOX_HOST:-0.0.0.0}"
PORT="${SANDBOX_PORT:-8100}"

echo "Starting sandbox HTTP server: backend=$SANDBOX_BACKEND on $HOST:$PORT"
exec python3 -m uvicorn sandbox.server:app --host "$HOST" --port "$PORT"
