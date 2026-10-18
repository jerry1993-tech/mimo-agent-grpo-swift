# 通用代码执行沙箱镜像（不依赖小米专用镜像的通用任务用）。
#
# 说明：
# - 小米 MiMo-V2.6-RL-oss 各任务的专用镜像在：
#     https://hub.docker.com/r/xiaomimimo/mimo-v2.6-rl-oss
#   （每行 TaskSpec.docker_image 自带对应镜像 tag，本 Dockerfile 仅作通用兜底。）
# - 该专用镜像已预置 git 历史截断、answer-leak 黑洞等反作弊；本通用镜像不提供这些。
#
# 运行方式（由 DockerBackend 拉起）：
#   docker run -d --network=none --memory=2g --cpus=2 --pids-limit=256 \
#              -w /work <image> sleep infinity
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# python3 + 常用构建工具 + git + coreutils(含 timeout) + bash
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates coreutils git \
        python3 python3-pip python3-venv \
        build-essential curl tzdata \
    && rm -rf /var/lib/apt/lists/*

# node（可选）：失败不阻断通用镜像构建
RUN apt-get update && (apt-get install -y --no-install-recommends nodejs npm || true) \
    && rm -rf /var/lib/apt/lists/*

# 非 root 运行用户
RUN useradd -m -s /bin/bash agent && mkdir -p /work && chown agent:agent /work

WORKDIR /work

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER agent
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# DockerBackend 以 `sleep infinity` 作为 CMD 传入，保持容器常驻供 docker exec
CMD ["sleep", "infinity"]
