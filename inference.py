#!/usr/bin/env python3

from vllm import LLM, SamplingParams

llm = LLM(
    model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16", # An AQLM model checkpoint
    # model="ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16",
    # model="ISTA-DASLab/c4ai-command-r-v01-AQLM-2Bit-1x16",
    enforce_eager=True,  # Don't compile the graph
    gpu_memory_utilization=0.99,
    max_model_len=1024, # this is the context size
    # kv_cache_dtype="fp8", # enabling this allows a bit more context size
    # use_v2_block_manager=False, # testing experimental stuff below
    # enable_chunked_prefill=True, max_num_batched_tokens=512,
    # next line is not needed as `KV cache scaling factors provided, but the KV cache data type is not FP8. KV cache scaling factors will not be used.`
    # quantization_param_path="/root/.cache/huggingface/hub/models--ISTA-DASLab--Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16/snapshots/f4ca0b50cf3c348d92b60cf98216ae6294f180cf/config.json",
    # quantization="aqlm", # already pulls this from the model
)
tokenizer = llm.get_tokenizer()

conversations = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'Generate a poem about the sun in Spanish'}],
    tokenize=False,
)

outputs = llm.generate(
    [conversations],
    SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    ),
    use_tqdm=False,
)

# print text output
print(outputs[0].outputs[0].text)

# calculate and print tokens per second
num_generated_tokens = len(outputs[0].outputs[0].token_ids)
arrival_time = outputs[0].metrics.arrival_time
finished_time = outputs[0].metrics.finished_time
total_time = finished_time - arrival_time
print("\n\n===")
print(f"Generated {num_generated_tokens} tokens in {total_time:.2f} seconds = {num_generated_tokens/total_time:.2f} tok/sec")
