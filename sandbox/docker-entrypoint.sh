#!/usr/bin/env bash
# 通用沙箱容器入口：保证 /work 归属 agent，再把 CMD（通常是 sleep infinity）exec 出去。
set -e

# 容器以非 root 起，/work 已在镜像内 chown；这里只做兜底。
if [ "$(id -u)" = "0" ]; then
    chown -R agent:agent /work 2>/dev/null || true
    exec gosu agent "$@" 2>/dev/null || exec "$@"
fi

exec "$@"
