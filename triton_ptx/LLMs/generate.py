"""Backend-agnostic text generation."""

from .llm import LLM


def generate(llm: LLM, messages, *, max_new_tokens=4096):
    """Generate a text response from chat messages."""
    input_ids = llm.tokenize_messages(messages)
    output_ids = llm.generate_tokens(input_ids, max_new_tokens=max_new_tokens)
    return llm.decode_response(output_ids, input_ids.shape[-1])
