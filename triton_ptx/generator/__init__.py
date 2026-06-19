
from .base import * # noqa: F403
from .anthropic_prompt import AnthropicPrompt
from .gemini_prompt import GeminiPrompt
from .openai_prompt import OpenAIPrompt

def drop_none_values(options: dict) -> dict:
    return {key: value for key, value in options.items() if value is not None}

def _get_generator_class(provider: str):
    generators = {
        "openai": OpenAIPrompt,
        "anthropic": AnthropicPrompt,
        "gemini": GeminiPrompt,
    }

    try:
        return generators[provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider!r}") from None


def build_prompter(provider: str, *, model: str | None = None, options: dict | None = None):
    init_kwargs = drop_none_values({"model": model, **(options or {})})
    return _get_generator_class(provider)(**init_kwargs)

