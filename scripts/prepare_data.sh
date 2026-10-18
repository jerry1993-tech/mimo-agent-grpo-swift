#!/usr/bin/env bash
# 下载 MiMo 数据并生成 GRPO / SFT 所需文件。
# 用法：bash scripts/prepare_data.sh [domains...]
#   domains 缺省为全部：code webdev general cyber music
set -euo pipefail
cd "$(dirname "$0")/.."

DOMAINS="${*:-code webdev general cyber music}"
BACKEND="${SANDBOX_BACKEND:-docker}"

echo "[1/2] 下载 parquet（domains=$DOMAINS）..."
python3 -m data.download_dataset --domains $DOMAINS

echo "[2/2] 生成 GRPO jsonl（backend=$BACKEND）..."
python3 -m data.prepare_grpo \
  --domains $DOMAINS \
  --backend "$BACKEND" \
  ${SANDBOX_URL:+--sandbox-url "$SANDBOX_URL"} \
  ${JUDGE_BASE_URL:+--judge-base-url "$JUDGE_BASE_URL"} \
  ${JUDGE_MODEL:+--judge-model "$JUDGE_MODEL"} \
  --output-path data/outputs/grpo_train.jsonl

echo "完成。产物见 data/outputs/（grpo_<domain>.jsonl）。"
echo "SFT 数据需先采集成功轨迹：见 scripts/collect_il.sh 或 data/collect_il_demo.py。"
