vLLM-inference-AQLM
===
Use vLLM and FlashAttention for AQLM quantized Local LLM model inferencing.

## Issues
Seems to require Python <3.11, so bleeding edge 3.12+ can use the
Dockerfile if desired.

## Build Docker Image
The base image `nvcr.io/nvidia/pytorch:24.03-py3` has all the python and
flash attention dependencies ready to go. Just adds python pip and venv.
```bash
docker build -t localllama/vllm-inference-aqlm .
```

## Run Docker Container
It will persist all `huggingface` and `pip` cache in the local `.cache/`
and `.config/` directories across runs to reduce extreneous downloads.
```
# Linux
./start-container.sh
# Windows (untested)
start-container.bat
```

## Inside the Docker Container Context
Once you are inside the container shell, run the following the first time.
```bash
# 1. create pip virtual environment
python3 -m venv ./venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip

# 2. install dependencies
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements-dev.txt

# 3. upgrade and check versions
python3 -m venv --upgrade ./venv
pip freeze
python3 -V
```

## Inference
Now you are ready to go. It will automatically download the models the
first time and persist them in `./cache/` folder across runs.

Remember to get into the python venv by sourcing the activate script
each time you start a new container.
```bash
source ./venv/bin/activate
./inference.py
```

## Benchmarks
[Inferencing Speed Benchmarks](BENCHMARKS.md)

## Eval
Still working on this, didn't get good results due to issues with
evals (e.g. `mmlu`) requiring logliklihood Also the evals available
are not exactly the same as those published by Meta etc (e.g. gsm8k
8-shot CoT) which is a bit different than simply specifying `gsm8k`
from what I can tell. So it isn't even apples to apples without
additional cleaning and voting likley using something like the project
[outlines](https://github.com/outlines-dev/outlines) or just some
regexes hah.

```bash
# Any evals requiring logliklihood e.g. mmlu don't work.. gsm8k seems okay though
source ./venv/bin/activate

# run AQLM evaluation
export MODEL_NAME="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
lm_eval \
    --model vllm \
    --model_args pretrained="$MODEL_NAME",dtype=auto,gpu_memory_utilization=0.99,enforce_eager=True,max_model_len=2048,kv_cache_dtype=auto \
    --tasks gsm8k \
    --batch_size 1

# for comparison run GGUF evaluation via API
# startup LMStudio or KoboldCPP etc and run your GGUF and turn on the API server then:
export OPENAI_API_KEY=nobearertoken
export API_URL="http://172.17.0.1:1234/v1"
export LLM_MODEL="openai/model"
lm_eval \
    --model local-chat-completions \
    --tasks gsm8k \
    --model_args model="$LLM_MODEL",base_url="$API_URL"
```

Additional hacking notes in [EVALS.md](EVALS.md)

## Quantization
Sorry, takes way to long to AQLM quantize a model at home on your 3090TI.

## References
* [r/LocalLLaMA Announcement](https://www.reddit.com/r/LocalLLaMA/comments/1clinlb/bringing_2bit_llms_to_production_new_aqlm_models/)
* [vllm-project/vlm](https://github.com/vllm-project/vllm)
* [Vahe1994/AQLM](https://github.com/Vahe1994/AQLM)
* [AQLM ipynb](https://github.com/Vahe1994/AQLM/blob/main/notebooks/aqlm_vllm.ipynb)
* [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
