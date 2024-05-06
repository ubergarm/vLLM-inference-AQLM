#!/usr/bin/env bash
docker run \
    --rm \
    -it \
    --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$PWD":/workspace \
    -v "$PWD/.cache":/root/.cache \
    localllama/vllm-inference-aqlm \
    /bin/bash
