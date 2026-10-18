#!/usr/bin/env bash
# 下载 MiMo 数据并生成 GRPO / SFT 所需文件。
# 用法：bash scripts/prepare_data.sh [domains...]
#   domains 缺省为全部：code webdev general cyber music
set -euo pipefail
cd "$(dirname "$0")/.."

DOMAINS="${*:-code webdev general cyber music}"
BACKEND="${SANDBOX_BACKEND:-docker}"

echo "[1/3] 下载 parquet（domains=$DOMAINS）..."
python3 -m data.download_dataset --domains $DOMAINS

echo "[2/3] 下载镜像映射表（image-mapping.jsonl，dataset_image -> dockerhub tag）..."
python3 -m data.image_mapping

if [[ " $DOMAINS " == *" general "* ]]; then
  echo "[2.5/3] general 域任务环境目录（可选，数量/体积大；取样训练可跳过全量）："
  echo "  全量：   python -m data.download_general_envs"
  echo "  按任务： python -m data.download_general_envs --from-jsonl data/outputs/grpo_general.jsonl"
  echo "  （未下载的 env_task_dir 行会被 prepare_grpo 跳过并计数）"
fi

echo "[3/3] 生成 GRPO jsonl（backend=$BACKEND）..."
python3 -m data.prepare_grpo \
  --domains $DOMAINS \
  --backend "$BACKEND" \
  ${SANDBOX_URL:+--sandbox-url "$SANDBOX_URL"} \
  ${JUDGE_BASE_URL:+--judge-base-url "$JUDGE_BASE_URL"} \
  ${JUDGE_MODEL:+--judge-model "$JUDGE_MODEL"} \
  --output-path data/outputs/grpo_train.jsonl

echo "完成。产物见 data/outputs/（grpo_<domain>.jsonl）。"
echo "SFT 数据需先采集成功轨迹：见 data/collect_il_demo.py。"
