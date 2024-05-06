@echo off
echo Untested docker script for windows...
docker run ^
       --rm ^
       -it ^
      --gpus all ^
      -v "%cd%":"/workspace" ^
      -v "%cd%/.cache":"/root/.cache" ^
      localllama/vllm-inference-aqml ^
      /bin/bash
