#!/usr/bin/env bash
# =====================================================================
# 任务镜像预拉取（在沙箱服务器上运行）。
#
# 背景：code 域 2698 个任务每个是独立镜像 tag（1~2GB+），公共加速器对
# xiaomimimo 仓缓存差、拉取慢。会话创建内同步 pull 会撞 HTTP 超时，
# 正确做法是训练前按取样任务清单提前拉好（docker 分层缓存，断点续传）。
#
# 用法（服务器上，工程根目录）：
#   # 从 GPU 实例传来的 grpo_<domain>.jsonl 提取 dockerhub_image 清单：
#   bash scripts/prefetch_images.sh grpo_code.list      # 每行一个镜像全名
#   # 或直接指定：
#   bash scripts/prefetch_images.sh xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001714
#
# 清单生成（GPU 实例上）：
#   python - <<'PY'
#   import json
#   imgs = []
#   for ln in open('data/outputs/grpo_code.jsonl'):
#       im = json.loads(ln)['env_config'].get('dockerhub_image')
#       if im and im not in imgs: imgs.append(im)
#   open('grpo_code.list','w').write('\n'.join(imgs))
#   PY
#   scp grpo_code.list user@<sandbox-server>:~/mimo-agent-grpo-swift/
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:?用法: prefetch_images.sh <镜像清单文件 | 单个镜像名>}"
PARALLEL="${PARALLEL:-2}"        # 并发 pull 数（2核4G 建议 1~2）
RETRY="${RETRY:-3}"

# 单镜像模式
if [[ "$TARGET" != *.list ]] && [[ ! -f "$TARGET" ]]; then
  exec docker pull "$TARGET"
fi

IMAGES=$(grep -v '^\s*#' "$TARGET" | grep -v '^\s*$' | sort -u)
TOTAL=$(echo "$IMAGES" | wc -l)
echo "[prefetch] $TOTAL 个镜像，并发 $PARALLEL，失败重试 $RETRY 次"

ok=0; fail=0; failed_list=()
pull_one() {
  local img="$1"
  for i in $(seq 1 "$RETRY"); do
    if docker image inspect "$img" >/dev/null 2>&1; then echo "  [cached] $img"; return 0; fi
    if docker pull "$img" >/dev/null 2>&1; then echo "  [ok]     $img"; return 0; fi
    echo "  [retry$i] $img"
    sleep 5
  done
  echo "  [FAIL]   $img"
  return 1
}
export -f pull_one
export RETRY

# xargs 并发控制
echo "$IMAGES" | xargs -P "$PARALLEL" -I{} bash -c 'pull_one "$@"' _ {} \
  > /tmp/prefetch_result.txt || true

# 统计（xargs 子进程 echo 汇总到文件后统计）
while read -r line; do
  case "$line" in
    "[ok]"*|"[cached]"*) ok=$((ok+1)) ;;
    "[FAIL]"*) fail=$((fail+1)); failed_list+=("${line#[FAIL]   }") ;;
  esac
done < /tmp/prefetch_result.txt

echo "===================================================="
echo "[prefetch] 成功 $ok / $TOTAL，失败 $fail"
if [ "$fail" -gt 0 ]; then
  echo "失败清单（可重跑本脚本续传）："
  printf '  %s\n' "${failed_list[@]}"
  exit 1
fi
docker system df | head -4
