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
folder across runs to reduce extreneous downloads.
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
Inside the docker Context
```bash
source ./venv/bin/activate
# install evaluation harness
git clone git@github.com:Vahe1994/AQLM.git
cd AQML
pip install -r lm-evaluation-harness/requirements.txt

export CUDA_VISIBLE_DEVICES=0
#export QUANTZED_MODEL=<PATH_TO_SAVED_QUANTIZED_MODEL_FROM_MAIN.py>
export MODEL_PATH="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
#export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
#export WANDB_PROJECT=MY_AQ_LM_EVAL
#export WANDB_NAME=COOL_EVAL_NAME

python lmeval.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_PATH,dtype=float16,use_accelerate=True \
    --load $QUANTZED_MODEL \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 1

# install evaluation harness
#git clone https://github.com/EleutherAI/lm-evaluation-harness
#cd lm-evaluation-harness
#pip install -e .
#cd ..
##pip install aqlm[gpu]
#
##pip install lm_eval[vllm]
##--model_args pretrained="$MODEL_NAME",tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
#export MODEL_NAME="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
#lm_eval \
#    --model vllm \
#    --model_args pretrained="$MODEL_NAME",dtype=auto,gpu_memory_utilization=0.8 \
#    --tasks mmlu \
#    --batch_size auto
#
## run
#export MODEL_NAME="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
#lm_eval \
#    --model hf \
#    --model_args pretrained="$MODEL_NAME" \
#    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge,gsm8k,mmlu \
#    --device cuda:0 \
#    --batch_size 1
#
#export OPENAI_API_KEY=nobearertoken
#export API_URL="http://172.17.0.1:1234/v1"
#export LLM_MODEL="openai/model"
#export LLM_MODEL="openai/model"
#lm_eval \
#    --model local-chat-completions \
#    --tasks mmlu \
#    --model_args model="$LLM_MODEL",base_url="$API_URL"
```

## Quantization
Sorry, takes way to long to AQLM quantize a model at home on your 3090TI.

## References
* [r/LocalLLaMA Announcement](https://www.reddit.com/r/LocalLLaMA/comments/1clinlb/bringing_2bit_llms_to_production_new_aqlm_models/)
* [vllm-project/vlm](https://github.com/vllm-project/vllm)
* [Vahe1994/AQLM](https://github.com/Vahe1994/AQLM)
* [AQLM ipynb](https://github.com/Vahe1994/AQLM/blob/main/notebooks/aqlm_vllm.ipynb)
* [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
