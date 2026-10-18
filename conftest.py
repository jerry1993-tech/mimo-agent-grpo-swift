"""pytest 全局：确保工程根在 sys.path（不依赖安装）。"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
