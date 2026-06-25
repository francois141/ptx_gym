
from .base import * # noqa: F403
from .anthropic_prompt import AnthropicPrompt
from .fake_add_prompt import FakeAddPrompt
from .gemini_prompt import GeminiPrompt
from .openai_prompt import OpenAIPrompt
from .openrouter_prompt import OpenRouterPrompt

def drop_none_values(options: dict) -> dict:
    return {key: value for key, value in options.items() if value is not None}

def _get_llm_endpoint_class(provider: str):
    endpoints = {
        "openai": OpenAIPrompt,
        "openrouter": OpenRouterPrompt,
        "anthropic": AnthropicPrompt,
        "gemini": GeminiPrompt,
        "fake_add": FakeAddPrompt,
    }

    try:
        return endpoints[provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider!r}") from None


def create_llm_endpoint(provider: str, *, model: str | None = None, options: dict | None = None):
    init_kwargs = drop_none_values({"model": model, **(options or {})})
    return _get_llm_endpoint_class(provider)(**init_kwargs)
