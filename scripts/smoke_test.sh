#!/usr/bin/env bash
# 离线静态/闭环冒烟：不依赖 GPU / docker（沙箱用 local 后端 + mock policy）。
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m py_compile $(find . -name "*.py" -not -path "*/research/*")
echo "py_compile OK"
python3 -m pytest -q tests
