vLLM-inference-AQML
===
Use vLLM and FlashAttention for AQML quantized Local LLM model inferencing.

## Issues
Seems to require Python <3.11, so bleeding edge 3.12+ can use the
Dockerfile if desired.

## Build Docker Image
The base image `nvcr.io/nvidia/pytorch:24.03-py3` has all the python and
flash attention dependencies ready to go. Just adds python pip and venv.
```bash
docker build -t localllama/vllm-inference-aqml .
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

## Quantization
Sorry, takes way to long to AQML quantize a model at home on your 3090TI.

## References
* [r/LocalLLaMA Announcement](https://www.reddit.com/r/LocalLLaMA/comments/1clinlb/bringing_2bit_llms_to_production_new_aqlm_models/)
* [vllm-project/vlm](https://github.com/vllm-project/vllm)
* [Vahe1994/AQML](https://github.com/Vahe1994/AQLM)
* [AQML ipynb](https://github.com/Vahe1994/AQLM/blob/main/notebooks/aqlm_vllm.ipynb)
