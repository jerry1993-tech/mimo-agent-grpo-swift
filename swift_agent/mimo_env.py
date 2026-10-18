"""MS-Swift 对接：MiMo Agent 多轮交互式沙箱环境（gym Env）。

这是本工程与 ms-swift 的核心集成点。我们【不】在 reward func 里重放多轮（那会破坏
on-policy），而是实现 ms-swift 原生的 ``swift.rollout.gym_env.Env`` 并注册为 ``mimo_env``：
- reset：按任务（每行 env_config）启动【该任务专属】的隔离后端，返回 (problem, info, system)；
- 每轮 step：解析 assistant 文本里的 <tool_call>，在专属容器内真实执行，把 observation 回灌；
  中间步 reward=0.0、done=False；
- submit 时：调用领域 verifier（真实执行 hidden 测试 / rubric / 规则），最终步一次性返回
  outcome 且 done=True —— 对齐小米“逐轮 generate+observation 回灌、末位单点 outcome reward”。

API 依据（ms-swift 4.5.3）：
- swift/rollout/gym_env.py 的 Env(reset/step/close) 与 envs 注册表
- swift/rollout/multi_turn.py 的 GYMScheduler（total_reward 累积、observation 以 user 角色回灌）
- swift/rl_core/grpo_algorithm.py（use_gym_env 时读 rollout_infos['total_reward'] 作为 reward）
"""
from __future__ import annotations

import os
import sys

# 项目根 sys.path 兜底（保证在任何入口下都能 import 工程内模块）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any, Dict, Optional  # noqa: E402

from swift.rollout.gym_env import Env, envs  # noqa: E402

from swift_agent.spec import TaskSpec, GYM_ENV_NAME  # noqa: E402
from swift_agent.tool_parser import (  # noqa: E402
    parse_tool_calls, build_system_prompt,
)
from swift_agent.turn_executor import execute_calls, finalize  # noqa: E402
from sandbox.base import get_backend  # noqa: E402

_NO_TOOL_HINT = (
    'No <tool_call> detected. You MUST either take an action with\n'
    '<tool_call>\n{"name": "bash", "arguments": {"command": "..."}}\n</tool_call>\n'
    'or finish with the "submit" tool.'
)


class MiMoAgentEnv(Env):
    """一个实例对应一条任务 trajectory；持有专属后端。"""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.spec: TaskSpec = TaskSpec.from_env_config(env_config)
        self.backend = None
        self.tool_dispatcher = None
        self._closed = False

    # ------------------------------------------------------------------
    async def reset(self, config) -> tuple[str, Dict[str, Any], str]:
        # 启动专属隔离后端（docker/local/remote）
        self.backend = get_backend(self.spec)
        await self.backend.start()

        # general 域：装配 env_task_dir（manifest uploads + MCP 服务 + rubrics）
        if self.spec.domain == 'general':
            try:
                from sandbox.general_env_setup import setup_general_env, GeneralEnvError
                setup_info = await setup_general_env(self.spec, self.backend)
                if isinstance(setup_info.get('rubrics'), (dict, list)):
                    self.spec.rubrics = setup_info['rubrics']
            except GeneralEnvError as e:
                await self._safe_close()
                raise RuntimeError(f'general env setup failed: {e}') from e

        # general 域：尝试装配 MCP 工具分发器（setup_general_env 已写入 spec.extra['mcp_servers']）
        try:
            from sandbox.mcp_client import mcp_dispatcher_factory
            self.tool_dispatcher = await mcp_dispatcher_factory(self.spec)
        except Exception:
            self.tool_dispatcher = None

        system_prompt = build_system_prompt(self.spec, extra_tools=self.spec.tools)
        # 初始 observation 即任务描述（GYMScheduler 会据此重建 [system, user]）
        return self.spec.problem_statement, {'task': self.spec.summary()}, system_prompt

    # ------------------------------------------------------------------
    async def step(self, action) -> tuple[str, float, bool, Dict[str, Any]]:
        last = action[-1] if action else {}
        text = last.get('content', '') if isinstance(last, dict) else str(last)
        text = text or ''

        # ---- music：单轮纯文本 ABC，直接规则打分并结束 ----
        if self.spec.domain == 'music':
            from verifier.music_verifier import MusicVerifier
            res = MusicVerifier(self.spec).score_text(text)
            await self._safe_close()
            return '', float(res.score), True, {
                'is_infra': res.is_infra,
                'components': res.components,
                'stop_reason': 'music',
            }

        # ---- 解析工具调用 ----
        calls = parse_tool_calls(text)
        if not calls:
            # 模型这一轮没有发起任何工具调用：给一次提示，不计 reward、不结束
            return _NO_TOOL_HINT, 0.0, False, {'stop_reason': 'no_tool_call'}

        turn = await execute_calls(self.backend, self.spec, calls, self.tool_dispatcher)

        # ---- 基建故障：标记 is_infra 并结束（调用方应在 GRPO 组内剔除，勿当模型 0 分）----
        if turn.is_infra:
            await self._safe_close()
            return turn.observation, 0.0, True, {
                'is_infra': True,
                'tool_error_count': turn.tool_error_count,
                'stop_reason': 'infra',
            }

        # ---- submit：运行 verifier，一次性给出最终 outcome ----
        if turn.submitted:
            if turn.submit_answer:
                self.spec.extra['answer'] = turn.submit_answer
                try:
                    await self.backend.write_file('answer.md', turn.submit_answer)
                except Exception:
                    pass
            vres = await finalize(self.spec, self.backend)
            await self._safe_close()
            return turn.observation, float(vres.score), True, {
                'is_infra': vres.is_infra,
                'components': vres.components,
                'verifier_message': vres.message,
                'stop_reason': 'submitted',
            }

        # ---- 中间步：reward=0、未结束，回灌 observation ----
        return turn.observation, 0.0, False, {
            'tool_error_count': turn.tool_error_count,
            'stop_reason': 'continue',
        }

    # ------------------------------------------------------------------
    async def close(self):
        await self._safe_close()

    async def _safe_close(self):
        if self._closed:
            return
        self._closed = True
        if self.backend is not None:
            try:
                await self.backend.stop()
            except Exception:
                pass


# 注册到 ms-swift 的 env 注册表（GYMScheduler._create_env 读 env_config['name']）
envs[GYM_ENV_NAME] = MiMoAgentEnv
