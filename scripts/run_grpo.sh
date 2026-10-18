#!/usr/bin/env bash
# GRPO 训练。用法：bash scripts/run_grpo.sh [code|music|general]
set -euo pipefail
cd "$(dirname "$0")/.."

DOMAIN="${1:-code}"
CFG="configs/grpo_${DOMAIN}.yaml"
[ -f "$CFG" ] || { echo "未知域: $DOMAIN（配置 $CFG 不存在）"; exit 1; }

if [ -n "${NPROC_PER_NODE:-}" ]; then
  echo "多卡训练 nproc=$NPROC_PER_NODE, config=$CFG"
  exec python3 -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    swift_agent/train_entry.py "$CFG"
else
  echo "单进程训练, config=$CFG"
  exec python3 swift_agent/train_entry.py "$CFG"
fi
