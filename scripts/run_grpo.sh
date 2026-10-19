#!/usr/bin/env bash
# GRPO 训练。用法：bash scripts/run_grpo.sh [code|music|general]
set -euo pipefail
cd "$(dirname "$0")/.."

DOMAIN="${1:-code}"
CFG="configs/grpo_${DOMAIN}.yaml"
[ -f "$CFG" ] || { echo "未知域: $DOMAIN（配置 $CFG 不存在）"; exit 1; }

# Blackwell(sm120) + flashinfer 0.6.12 兼容绕开：
# 该 wheel 未编 sm120 cubin，vLLM sampler 初始化时 import flashinfer 会因
# arch 探测失败而崩（"SM 12.x requires CUDA >= 12.9"）。关掉 flashinfer
# sampler（回退原生 top-k/top-p，功能等价）。其它架构无副作用；
# 升级到支持 sm120 的 flashinfer 后可移除。
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# vLLM colocate EngineCore 用 fork 而非 spawn：spawn 子进程会 re-import
# 训练入口与 swift 全量包，在本环境实测出现握手死锁（EngineCore 0% CPU
# 挂 17min+，GPU 无显存分配）。fork 复用主进程状态，无 re-import 问题。
# CUDA 已初始化的进程 fork 子进程用 CUDA 有风险，但 vLLM 的 EngineCore
# 在 fork 后自行初始化 CUDA 上下文（V1 引擎设计如此），本机实测可用。
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"

if [ -n "${NPROC_PER_NODE:-}" ]; then
  echo "多卡训练 nproc=$NPROC_PER_NODE, config=$CFG"
  exec python3 -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    swift_agent/train_entry.py "$CFG"
else
  echo "单进程训练, config=$CFG"
  exec python3 swift_agent/train_entry.py "$CFG"
fi
