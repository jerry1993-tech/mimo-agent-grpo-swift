#!/usr/bin/env bash
# 启动“外接沙箱”HTTP 服务（sandbox/server.py）—— 前台运行，适合手动/调试。
# 生产/长期部署请用 scripts/deploy_sandbox_server.sh（systemd 自启 + token）。
#
# 用法：
#   bash scripts/start_sandbox.sh                                # docker 后端
#   SANDBOX_BACKEND=local SANDBOX_PORT=8100 bash scripts/start_sandbox.sh   # 本机演示
#   SANDBOX_TOKEN=xxx SANDBOX_MAX_CONCURRENT_SESSIONS=8 bash scripts/start_sandbox.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export SANDBOX_BACKEND="${SANDBOX_BACKEND:-docker}"
HOST="${SANDBOX_HOST:-0.0.0.0}"
PORT="${SANDBOX_PORT:-8100}"
MAX_SESSIONS="${SANDBOX_MAX_CONCURRENT_SESSIONS:-0}"

echo "Starting sandbox HTTP server: backend=$SANDBOX_BACKEND on $HOST:$PORT (max_sessions=$MAX_SESSIONS)"
if [ -n "${SANDBOX_TOKEN:-}" ]; then
  echo "Auth: X-Sandbox-Token required on all business endpoints."
else
  echo "WARNING: SANDBOX_TOKEN 未设置 —— 公网监听时任何人都可创建容器执行命令！"
fi
exec python3 -m uvicorn sandbox.server:app --host "$HOST" --port "$PORT" \
  --timeout-keep-alive 900
