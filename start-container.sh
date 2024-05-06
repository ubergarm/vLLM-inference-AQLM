#!/usr/bin/env bash
docker run \
    --rm \
    -it \
    --gpus all \
    -v "$PWD":/workspace \
    -v "$PWD/.cache":/root/.cache \
    localllama/vllm-inference-aqlm \
    /bin/bash
