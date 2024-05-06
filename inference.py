#!/usr/bin/env python3

from vllm import LLM, SamplingParams

llm = LLM(
    model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16", # An AQLM model checkpoint
    # model="ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16",
    enforce_eager=True,  # Don't compile the graph
    gpu_memory_utilization=0.99,
    max_model_len=1024,
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
