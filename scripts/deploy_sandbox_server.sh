#!/usr/bin/env bash
# =====================================================================
# 沙箱服务器一键部署（在 CPU 服务器上运行，如阿里云 2核4G 实例）。
#
# 用法：
#   bash scripts/deploy_sandbox_server.sh              # 交互式（生成随机 token）
#   SANDBOX_TOKEN=my-secret bash scripts/deploy_sandbox_server.sh
#   MAX_SESSIONS=8 bash scripts/deploy_sandbox_server.sh
#
# 做了什么：
#   1. 创建 Python venv（/opt/mimo-sandbox/venv），安装 fastapi/uvicorn/httpx 等轻依赖
#      （不需要 torch/ms-swift/vllm —— 服务端只跑 docker CLI + FastAPI）；
#   2. 写入 /etc/mimo-sandbox.env（token/端口/并发上限）；
#   3. 安装 systemd 服务 mimo-sandbox（开机自启、崩溃自动拉起）；
#   4. 预拉取 webdev/general 共享镜像（可选，code/cyber 按任务拉取）。
#
# 部署后 GPU 实例侧只需：
#   export SANDBOX_URL=http://<本机IP>:8100 SANDBOX_TOKEN=<token>
#   python -m data.prepare_grpo --backend remote --sandbox-url $SANDBOX_URL ...
# =====================================================================
set -euo pipefail

# ---- 配置 ----
INSTALL_DIR="${INSTALL_DIR:-/opt/mimo-sandbox}"
SERVICE_NAME="mimo-sandbox"
PORT="${SANDBOX_PORT:-8100}"
MAX_SESSIONS="${MAX_SESSIONS:-8}"          # 2核4G 建议 4~8；每容器默认 2g/2cpu
TOKEN="${SANDBOX_TOKEN:-$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 32)}"
ENV_FILE="/etc/mimo-sandbox.env"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "== [1/5] 系统依赖检查 =="
command -v docker >/dev/null || { echo "ERROR: docker 未安装"; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon 不可用"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 未安装 (apt install python3-venv)"; exit 1; }

echo "== [2/5] 创建 venv 并安装轻依赖（无 torch/vllm） =="
mkdir -p "$INSTALL_DIR"
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install -q --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -q fastapi 'uvicorn[standard]' httpx pydantic python-multipart
# 同步工程代码（sandbox/ + swift_agent/spec.py 依赖）
rsync -a --delete \
  --include 'sandbox/***' --include 'swift_agent/***' --include 'verifier/***' \
  --exclude '*' "$REPO_DIR/" "$INSTALL_DIR/app/"
# 若无 rsync，退化为整目录拷贝
if [ ! -d "$INSTALL_DIR/app/sandbox" ]; then
  mkdir -p "$INSTALL_DIR/app"
  cp -r "$REPO_DIR/sandbox" "$REPO_DIR/swift_agent" "$REPO_DIR/verifier" "$INSTALL_DIR/app/"
fi

echo "== [3/5] 写入环境配置 $ENV_FILE =="
cat > "$ENV_FILE" <<EOF
SANDBOX_HOST=0.0.0.0
SANDBOX_PORT=$PORT
SANDBOX_BACKEND=docker
SANDBOX_TOKEN=$TOKEN
SANDBOX_MAX_CONCURRENT_SESSIONS=$MAX_SESSIONS
# 任务容器资源默认值（可被每任务的 cpus/memory_mb 覆盖）
SANDBOX_DOCKER_MEMORY=1g
SANDBOX_DOCKER_CPUS=1.0
SANDBOX_DOCKER_PIDS_LIMIT=128
SANDBOX_TTL_SEC=7200
SANDBOX_DOCKER_PULL_TIMEOUT=1800
EOF
chmod 600 "$ENV_FILE"

echo "== [4/5] 安装 systemd 服务 =="
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=MiMo Agent GRPO sandbox server (per-task docker)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR/app
EnvironmentFile=$ENV_FILE
ExecStart=$INSTALL_DIR/venv/bin/python -m sandbox.server
Restart=always
RestartSec=5
# 收集任务容器退出后自动回收（systemd 停止时清理 session）
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
sleep 2
systemctl --no-pager status "$SERVICE_NAME" | head -8

echo "== [5/5] 健康检查 =="
curl -s "http://127.0.0.1:${PORT}/healthz" && echo

echo
echo "===================================================================="
echo "  部署完成！GPU 实例侧使用以下配置："
echo "    export SANDBOX_URL=http://<服务器IP>:${PORT}"
echo "    export SANDBOX_TOKEN=${TOKEN}"
echo "  安全组请放行 TCP ${PORT}（建议仅白名单 GPU 实例出口 IP）。"
echo "  并发上限：${MAX_SESSIONS} 个容器（改 $ENV_FILE 后 systemctl restart $SERVICE_NAME）"
echo "===================================================================="
