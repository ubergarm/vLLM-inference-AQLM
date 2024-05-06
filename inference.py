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

print(outputs[0].outputs[0].text)
