"""MS-Swift GRPO 训练统一入口（支持 yaml 配置 + 自动注册 mimo_env）。

用法（工程根目录）：
    # 单进程 / 单卡
    python swift_agent/train_entry.py configs/grpo_code.yaml
    # 多卡（torchrun；每个 rank 都会注册 mimo_env 并进入 rlhf_main）
    NPROC_PER_NODE=8 python -m torch.distributed.run --nproc_per_node=8 \
        swift_agent/train_entry.py configs/grpo_code.yaml
    # 也可不用 yaml，直接追加任意 swift rlhf 参数：
    python swift_agent/train_entry.py --rlhf_type grpo --model ... --dataset ...

为什么需要这个入口：ms-swift CLI 的 --gym_env 只接受已注册名字，而我们的 mimo_env
是工程内自定义类，必须在调用 rlhf_main 之前 import 触发注册（swift/rollout/gym_env.envs）。
"""
from __future__ import annotations

import os
import sys

# 项目根兜底
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _expand_config():
    """若首个参数是 .yaml/.json，按 ms-swift 约定展开为 --key value，并写回 sys.argv。"""
    argv = sys.argv[1:]
    if not argv:
        return
    first = argv[0].lower()
    if first.endswith(('.yaml', '.yml', '.json')):
        from swift.cli.main import parse_yaml_args
        parse_yaml_args(argv)            # 原地把 yaml 展开成 flag 列表
        sys.argv = [sys.argv[0], *argv]


def main():
    # 先展开配置（需要 swift 已安装）
    _expand_config()

    # 注册自定义多轮沙箱环境（import 即执行 envs['mimo_env']=MiMoAgentEnv）
    from swift_agent import mimo_env  # noqa: F401

    # 进入 ms-swift RLHF 主流程（解析 sys.argv；--rlhf_type grpo）
    from swift.pipelines import rlhf_main
    rlhf_main()


if __name__ == '__main__':
    main()
