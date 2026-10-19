# 接口冻结规范（INTERFACE SPEC）

> 本文件是 `mimo-agent-grpo-swift` 各模块之间的**契约**。所有并行实现必须严格遵守，
> 不得擅自更改类名、方法签名、返回结构与字段名。实现细节（内部算法）可自由发挥。
> 版本基准：ms-swift **4.5.2**；数据集 `XiaomiMiMo/MiMo-V2.6-RL-oss`（Apache-2.0）。

## 0. 总体数据流（与小米 VERL 对齐）

```
parquet(任务环境池)
   └── data/prepare_grpo.py ──► grpo.jsonl  每行: {messages:[user(task)], env_config:{...TaskSpec}}
                                       │
swift rlhf (GRPO, GYMScheduler)        │
   └─ 每条 trajectory:
        env.reset(req)  -> (problem_observation, info, system_prompt)   # 起隔离容器
        循环 t = 1..max_turns:
           vLLM 生成 assistant(可含 <tool_call> JSON)
        env.step(messages) ->
              解析 tool_call -> 在【该任务专属容器】内执行 -> observation
              中间步: reward=0.0, done=False
              submit/到轮次上限: 运行 verifier(真实执行) -> reward=outcome, done=True
        total_reward = 末步 outcome（中间全 0）  # 对齐小米“末位单点 outcome reward”
```

## 1. `swift_agent/spec.py` —— `TaskSpec`（框架无关，禁止 import swift）

跨域统一的任务规格。关键序列化约定：GRPO jsonl 的 `env_config` 列就是
`TaskSpec.to_env_config()` 的结果；训练侧 `TaskSpec.from_env_config()` 还原。

- 必有字段：`instance_id, dataset_type, domain, docker_image, cwd, problem_statement`
- code 域：`test_command, test_patch, verifier_timeout_sec`
- general 域：`num_turns, env_task_dir, rubrics`（verifier_meta 解析结果）
- music 域：`music_meta`（bpm/meter/nvoice_want/tag/...）
- cyber 域：`description`（含期望崩溃函数/sanitizer/error_type 线索）
- 运行期开关放在 env_config 顶层：`name`（gym env 注册名，恒为 `mimo_env`）、
  `backend`（`docker|local|remote`）、`sandbox_url`（remote 时）、
  `judge_base_url/judge_model/judge_api_key`（LLM-as-judge）、
  `tool_timeout_sec, max_tool_output_chars, network`（是否允许联网）。
- `extra`：原始 instance_json 的兜底 dict，任何未识别字段都保留在此。

## 2. `sandbox/base.py` —— 沙箱后端（框架无关）

```python
@dataclass
class ExecResult:
    output: str            # stdout+stderr（已按 max_tool_output_chars 截断，保留头尾）
    returncode: int
    reason: str            # ok | timeout | transport_error | backend_error
    truncated: bool = False

class SandboxBackend(ABC):
    name: str
    def __init__(self, spec: TaskSpec, **opts): ...
    async def start(self) -> None: ...                 # 拉起/准备环境
    async def exec(self, command: str, timeout: float | None = None,
                   cwd: str | None = None) -> ExecResult: ...   # argv 等价 timeout N bash -lc
    async def write_file(self, path: str, content: str) -> None: ...
    async def read_file(self, path: str) -> str: ...
    async def reset_files(self, relpaths: list[str]) -> bool: ...  # code 打分前 git reset（可选）
    async def stop(self) -> None: ...                  # 销毁，用完即毁
    # 必须支持 async context manager: async with get_backend(...) as sb: ...
```
- 注册表 `BACKENDS: dict[str, type]`；工厂 `get_backend(spec) -> SandboxBackend`，
  依据 `spec.env_config['backend']`（默认 docker）选择。
- 三个实现：`docker_backend.DockerBackend`（CLI: docker run/exec/cp/rm）、
  `local_backend.LocalBackend`（临时目录 + asyncio.subprocess，仅供测试/低风险，
  必须在文档与日志显著标注“无隔离”）、`remote_backend.RemoteBackend`（HTTP 调 sandbox server）。

## 3. `verifier/base.py` —— 评判器（框架无关）

```python
@dataclass
class VerificationResult:
    score: float                  # 最终 outcome，[0,1]
    passed: bool                  # score>=阈值（code 二值时 score==1）
    components: dict[str, float]  # 分项（仅记录/可观测，不回传梯度）
    is_infra: bool                # 基建失败=True（绝不作模型 0 分训练）
    message: str                  # 人类可读说明 / 测试输出摘要

class Verifier(ABC):
    domain: str
    def __init__(self, spec: TaskSpec, **opts): ...
    async def verify(self, backend: SandboxBackend) -> VerificationResult: ...
```
- 注册表 `VERIFIERS`；工厂 `get_verifier(spec) -> Verifier`，按 `spec.domain` 选择。
- 五个实现：code（git apply test_patch→test_command，二值）、cyber（崩溃函数+sanitizer+
  error_type 全等匹配，二值）、music（ABC 记谱校验/特征，规则）、general（LLM rubric judge，
  逐条 1/0 均值）、webdev（渲染截图→视觉评分；组级相对排序需外部 grader，未配置时降级）。
- **铁律**：code 的 hidden 测试（test_patch/test_command）只能在 `verify()` 内、agent
  结束后才注入容器；agent 工作期间容器内不得存在测试文件。基建异常置 `is_infra=True`。

## 4. `swift_agent/tool_parser.py` —— 文本协议（框架无关）

- 模型在 assistant 文本中用一个或多个块表达工具调用：
  `<tool_call>{"name": str, "arguments": dict}</tool_call>`（兼容 ```tool_call fenced）。
- 内置工具：`bash{command,timeout?}`、`read_file{path}`、`write_file{path,content}`、
  `submit{answer?}`。general 域的 MCP 工具名透传给 `sandbox/mcp_client.py`。
- `parse_tool_calls(text) -> list[ToolCall]`；`render_observation(name,obs) -> str`
  输出 `<tool_result>...</tool_result>`；`is_submit(...)`；无 tool_call 时视为普通思考。

## 5. `swift_agent/mimo_env.py` —— ms-swift 适配（唯一 import swift 的核心文件）

- `class MiMoAgentEnv(gym_env.Env)`：reset 起后端并返回 (problem, {}, system_prompt)；
  step 解析 tool_call→在后端执行→拼 observation；submit 时 `get_verifier(spec).verify()`，
  中间步返回 reward=0.0，最终步返回 outcome 且 done=True；close 停后端。
- `envs['mimo_env'] = MiMoAgentEnv`。
- 基建失败：step 返回 done=True 且在 info 中置 `is_infra=True`、reward=0.0，
  并把该事实写入 rollout_infos（README 说明如何在 GRPO 组内剔除）。

## 6. 命名与运行约定

- 所有可执行脚本从工程根目录运行，优先 `python -m <pkg>.<module>`；CLI 文件内含
  项目根 sys.path 兜底。测试用根目录 `conftest.py`。
- 密钥/地址一律环境变量，见 `.env.example`，代码不硬编码。
- 对外 API/字段引用在代码注释与 README 中附官方 URL。
