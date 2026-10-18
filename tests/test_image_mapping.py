"""image-mapping 与镜像解析测试。

覆盖：
- data/image_mapping.py：load/resolve（含未收录名、已是全名）；
- TaskSpec.dockerhub_image 字段随 env_config 往返；
- DockerBackend._resolve_image 优先序（dockerhub_image > docker_image）；
- prepare_grpo 生成的 env_config 带 dockerhub_image（用 mock parquet 行）。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from data.image_mapping import load_image_mapping, resolve_dockerhub_image
from swift_agent.spec import TaskSpec


@pytest.fixture
def mapping_file(tmp_path):
    p = tmp_path / 'image-mapping.jsonl'
    p.write_text(
        '{"dataset_image": "format-code-task-001457:latest", '
        '"dockerhub_image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457"}\n'
        '{"dataset_image": "arvo-rl:v1-arvo-35858", '
        '"dockerhub_image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:arvo-v1-35858"}\n'
        '{"dataset_image": "webdev-rl-opensource:v2", '
        '"dockerhub_image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:webdev-rl-opensource"}\n',
        encoding='utf-8',
    )
    return str(p)


def test_load_and_resolve(mapping_file):
    m = load_image_mapping(os.path.dirname(mapping_file))
    assert len(m) == 3
    assert resolve_dockerhub_image('format-code-task-001457:latest', m) == \
        'docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457'
    assert resolve_dockerhub_image('arvo-rl:v1-arvo-35858', m) == \
        'docker.io/xiaomimimo/mimo-v2.6-rl-oss:arvo-v1-35858'


def test_resolve_unknown_returns_none(mapping_file):
    m = load_image_mapping(os.path.dirname(mapping_file))
    assert resolve_dockerhub_image('my-custom-image:dev', m) is None
    assert resolve_dockerhub_image('', m) is None


def test_resolve_already_hubname_passthrough(mapping_file):
    m = load_image_mapping(os.path.dirname(mapping_file))
    full = 'xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000001'
    assert resolve_dockerhub_image(full, m) == full


def test_missing_mapping_file_returns_empty(tmp_path):
    assert load_image_mapping(str(tmp_path)) == {}


def test_spec_roundtrip_with_dockerhub_image():
    spec = TaskSpec(domain='code', dataset_type='opensource-code',
                    docker_image='format-code-task-001457:latest',
                    dockerhub_image='docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457')
    cfg = spec.to_env_config()
    assert cfg['dockerhub_image'].endswith(':format-code-task-001457')
    spec2 = TaskSpec.from_env_config(cfg)
    assert spec2.dockerhub_image == spec.dockerhub_image
    assert spec2.docker_image == spec.docker_image


def test_docker_backend_resolve_image_priority():
    from sandbox.docker_backend import DockerBackend
    spec = TaskSpec(domain='code', docker_image='format-code-task-001457:latest',
                    dockerhub_image='docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457')
    be = DockerBackend(spec)
    assert be._resolve_image() == 'docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457'
    # 无 dockerhub_image -> 原名
    be2 = DockerBackend(TaskSpec(domain='code', docker_image='my-local-img:latest'))
    assert be2._resolve_image() == 'my-local-img:latest'


def test_spec_task_resources_from_instance_json():
    """general 域 instance_json 的 agent_timeout_sec/cpus/memory_mb 进 TaskSpec。"""
    import pyarrow  # noqa: F401
    row = {
        'data_source': 'mimoagent/general_agent',
        'prompt': [{'role': 'user', 'content': 'Do a knowledge-work task.'}],
        'extra_info': {
            'instance_id': 's3k_xxx',
            'dataset_type': 'general_agent',
            'instance_json': json.dumps({
                'dataset_type': 'general_agent',
                'instance_id': 's3k_xxx',
                'env_task_dir': 'envs/s3k_xxx',
                'docker_image': 'general-agent-env-0:oss',
                'cwd': '/work/workspace',
                'problem_statement': 'Do a knowledge-work task.',
                'agent_timeout_sec': 900.0,
                'cpus': 4.0,
                'memory_mb': 8192,
            }),
        },
    }
    spec = TaskSpec.from_parquet_row(row)
    assert spec.domain == 'general'
    assert spec.agent_timeout_sec == 900.0
    assert spec.cpus == 4.0
    assert spec.memory_mb == 8192
    assert spec.env_task_dir == 'envs/s3k_xxx'


def test_docker_backend_uses_task_resources():
    from sandbox.docker_backend import DockerBackend
    spec = TaskSpec(domain='general', docker_image='general-agent-env-0:oss',
                    cpus=4.0, memory_mb=8192)
    be = DockerBackend(spec)
    assert be.cpus == '4.0'
    assert be.memory == '8192m'
    # opts 覆盖任务级
    be2 = DockerBackend(spec, cpus=8.0, memory='16g')
    assert be2.cpus == '8.0'
    assert be2.memory == '16g'
    # 无任务级 -> 环境变量/默认
    be3 = DockerBackend(TaskSpec(domain='code', docker_image='x:1'))
    assert be3.memory == '2g'
    assert be3.cpus == '2.0'
