from time import perf_counter

import torch

from .llm import LLM


def measure(llm: LLM, messages, *, max_new_tokens=4096):
    """Generate a response and return it with its token throughput."""
    input_ids = llm.tokenize_messages(messages)
    torch.cuda.synchronize(input_ids.device)
    start_time = perf_counter()
    output_ids = llm.generate_tokens(input_ids, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize(input_ids.device)
    elapsed_seconds = perf_counter() - start_time
    generated_tokens = output_ids.shape[-1] - input_ids.shape[-1]
    response = llm.decode_response(output_ids, input_ids.shape[-1])
    return response, generated_tokens / elapsed_seconds
