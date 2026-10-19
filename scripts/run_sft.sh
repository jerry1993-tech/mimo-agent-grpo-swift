#!/usr/bin/env bash
# IL/SFT 预热（可选）。需要先有 data/outputs/sft.jsonl：
#   1) python -m data.collect_il_demo ...  采集成功轨迹 -> rollout_dump.jsonl
#   2) python -m data.prepare_sft --input data/outputs/rollout_dump.jsonl
# 小米开源仓只含 RL（起点为蒸馏权重），此阶段为等价 IL 预热。
set -euo pipefail
cd "$(dirname "$0")/.."
# Blackwell(sm120) + flashinfer 0.6.12 兼容（同 run_grpo.sh 说明）
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
exec swift sft configs/sft.yaml
