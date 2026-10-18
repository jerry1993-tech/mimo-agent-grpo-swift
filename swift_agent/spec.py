"""TaskSpec: 跨域统一的 MiMo Agent RL 任务规格（框架无关，禁止 import ms-swift）。

数据来源事实（已联网核实，见 research/01_dataset_schema.md）：
- HuggingFace 数据集 ``XiaomiMiMo/MiMo-V2.6-RL-oss``（Apache-2.0），共 5 个 parquet、7780 条任务。
- parquet 是“任务环境池”，不含轨迹。通用列：
    data_source, ability?, agent_name?, prompt(list[{role,content}]),
    reward_model({style,ground_truth}), extra_info(struct, 内含 instance_json: JSON 字符串)。
- music 域 extra_info 为扁平字段（src_id/lang/tag/length/nvoice_want/bpm/meter），无 instance_json。

本模块把异构 parquet 行归一化为 TaskSpec，并负责与 GRPO jsonl 的 env_config 列互转。
官方数据/代码：
- https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss
- https://github.com/XiaomiMiMo/verl  (recipes/*/dataset.py)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# gym env 注册名（swift_agent/mimo_env.py 中注册到 swift.rollout.gym_env.envs）
GYM_ENV_NAME = 'mimo_env'

# dataset_type -> 领域
_DATASET_TYPE_TO_DOMAIN = {
    'opensource-code': 'code',
    'webdev': 'webdev',
    'arvo': 'cyber',
    'general_agent': 'general',
    'terminal_bench': 'general',
    'music': 'music',
}


def _safe_json_loads(value: Any, default: Any = None) -> Any:
    """宽容地解析 JSON 字符串；已是 dict/list 则原样返回；失败返回 default。"""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode('utf-8', errors='ignore')
    if not isinstance(value, str):
        return default
    value = value.strip()
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        # 个别字段可能被单引号包裹；最后兜底
        try:
            return json.loads(value.replace("'", '"'))
        except Exception:
            return default


def _prompt_to_text(prompt: Any) -> str:
    """parquet 的 prompt 为 list[{role,content}]（每行 1 条 user）；兼容纯字符串。"""
    if prompt is None:
        return ''
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        contents = []
        for msg in prompt:
            if isinstance(msg, dict):
                contents.append(str(msg.get('content', '')))
            else:
                contents.append(str(msg))
        return '\n'.join(c for c in contents if c)
    return str(prompt)


@dataclass
class TaskSpec:
    """跨域统一任务规格。"""

    instance_id: str = ''
    dataset_type: str = ''
    domain: str = ''

    # 环境
    docker_image: str = ''
    cwd: str = ''
    network: bool = False

    # 任务描述
    problem_statement: str = ''
    description: str = ''

    # code 域
    test_command: str = ''
    test_patch: str = ''
    verifier_timeout_sec: float = 300.0

    # general 域
    env_task_dir: str = ''
    num_turns: int = 1
    rubrics: Optional[Dict[str, Any]] = None

    # music 域
    music_meta: Optional[Dict[str, Any]] = None

    # 运行期开关（写入 env_config 顶层）
    backend: str = 'docker'          # docker | local | remote
    sandbox_url: str = ''           # remote 模式
    judge_base_url: str = ''
    judge_api_key: str = ''
    judge_model: str = ''
    tool_timeout_sec: float = 30.0
    max_tool_output_chars: int = 8000

    # 显式工具 schema（可选，general MCP）
    tools: Optional[List[Dict[str, Any]]] = None

    # 兜底：原始 instance_json / extra_info
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 构造：parquet 行 -> TaskSpec
    # ------------------------------------------------------------------
    @staticmethod
    def from_parquet_row(row: Dict[str, Any]) -> 'TaskSpec':
        """把一条 parquet 行（pandas 取出的 dict）归一化。对缺失字段健壮。"""
        extra_info = row.get('extra_info') or {}
        if not isinstance(extra_info, dict):
            extra_info = _safe_json_loads(extra_info, default={}) or {}

        dataset_type = str(extra_info.get('dataset_type', '') or row.get('data_source', '') or '')
        domain = _DATASET_TYPE_TO_DOMAIN.get(dataset_type, '')
        if not domain:
            ability = str(row.get('ability', '') or '')
            if 'music' in ability:
                domain = 'music'
                dataset_type = dataset_type or 'music'
            elif 'webdev' in ability:
                domain = 'webdev'
            elif ability == 'swe':
                domain = 'code'
            elif ability == 'agent':
                domain = 'general'

        problem = _prompt_to_text(row.get('prompt'))

        spec = TaskSpec(
            instance_id=str(extra_info.get('instance_id', extra_info.get('src_id', '')) or ''),
            dataset_type=dataset_type,
            domain=domain,
            problem_statement=problem,
        )

        # music 域：extra_info 是扁平元数据，无 instance_json
        if domain == 'music' or (row.get('ability') == 'music_generation'):
            spec.domain = 'music'
            spec.dataset_type = spec.dataset_type or 'music'
            spec.music_meta = {
                k: extra_info.get(k)
                for k in ('src_id', 'lang', 'tag', 'length', 'nvoice_want', 'bpm', 'meter')
                if k in extra_info
            }
            if not spec.instance_id:
                spec.instance_id = str(extra_info.get('src_id', ''))
            return spec

        # 其余域：解析嵌套 instance_json（JSON 字符串，需二次解析）
        instance_json = _safe_json_loads(extra_info.get('instance_json'), default={}) or {}
        spec.extra = dict(instance_json)

        spec.docker_image = str(instance_json.get('docker_image', '') or '')
        spec.cwd = str(instance_json.get('cwd', '') or '')
        spec.problem_statement = str(instance_json.get('problem_statement', '') or problem)
        spec.description = str(instance_json.get('description', '') or '')
        spec.test_command = str(instance_json.get('test_command', '') or '')
        spec.test_patch = str(instance_json.get('test_patch', '') or '')
        if instance_json.get('verifier_timeout_sec') is not None:
            try:
                spec.verifier_timeout_sec = float(instance_json['verifier_timeout_sec'])
            except (TypeError, ValueError):
                pass
        spec.env_task_dir = str(instance_json.get('env_task_dir', '') or '')
        if instance_json.get('num_turns') is not None:
            try:
                spec.num_turns = int(instance_json['num_turns'])
            except (TypeError, ValueError):
                pass
        # terminal_bench 资源规格
        if instance_json.get('allow_internet') is not None:
            spec.network = bool(instance_json['allow_internet'])
        # webdev 行内 task_id
        if not spec.instance_id:
            spec.instance_id = str(instance_json.get('task_id', instance_json.get('instance_id', '')) or '')
        return spec

    # ------------------------------------------------------------------
    # 序列化：TaskSpec <-> env_config(dict)
    # ------------------------------------------------------------------
    def to_env_config(self, runtime_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """产出 GRPO jsonl 的 env_config 列。含 gym env 名与全部任务/运行期字段。"""
        data = asdict(self)
        # gym env 注册名（GYMScheduler._create_env 读 env_config['name']）
        data['name'] = GYM_ENV_NAME
        # 顶层 name 与 dataclass 字段无冲突（dataclass 无 name 字段）
        if runtime_overrides:
            for k, v in runtime_overrides.items():
                if v is not None:
                    data[k] = v
        return data

    @staticmethod
    def from_env_config(env_config: Dict[str, Any]) -> 'TaskSpec':
        """从 GRPO env_config 还原 TaskSpec（忽略 name 键）。"""
        env_config = dict(env_config or {})
        env_config.pop('name', None)
        valid = {f for f in TaskSpec.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in env_config.items() if k in valid}
        spec = TaskSpec(**kwargs)
        # 不认识的运行期字段塞进 extra，避免丢信息
        for k, v in env_config.items():
            if k not in valid:
                spec.extra[k] = v
        return spec

    def summary(self) -> str:
        return (f'TaskSpec(domain={self.domain}, type={self.dataset_type}, '
                f'id={self.instance_id}, image={self.docker_image or "-"})')
