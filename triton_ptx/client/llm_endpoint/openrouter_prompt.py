import os

from openai import OpenAI

from .base import LLMEndpoint, parse_response_text


class OpenRouterPrompt(LLMEndpoint):
    DEFAULT_MODEL = "qwen/qwen3-coder"
    BASE_URL = "https://openrouter.ai/api/v1"

    PRICING_PER_1M_TOKENS = {
        # Price estimates in USD per 1M tokens.
        # Keep these aligned with https://openrouter.ai/models.
        "qwen/qwen3-coder": {
            "input": 0.22,
            "output": 1.80,
        },
        "deepseek/deepseek-v4-pro": {
            "input": 0.435,
            "cached_input": 0.003625,
            "output": 0.87,
        },
    }

    def __init__(
        self,
        model=None,
        api_key=None,
        site_url=None,
        app_name=None,
    ):
        self.model = model or self.DEFAULT_MODEL

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter support requires OPENROUTER_API_KEY or an api_key option."
            )

        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-OpenRouter-Title"] = app_name

        self.client = OpenAI(
            api_key=key,
            base_url=self.BASE_URL,
            default_headers=default_headers or None,
        )

    def _estimate_cost(self, response):
        pricing = self.PRICING_PER_1M_TOKENS.get(self.model)
        usage = getattr(response, "usage", None)

        if pricing is None or usage is None:
            return None

        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_input_tokens = 0

        if prompt_details is not None:
            cached_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

        uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
        cached_input_price = pricing.get("cached_input", pricing["input"])

        return (
            uncached_input_tokens * pricing["input"]
            + cached_input_tokens * cached_input_price
            + output_tokens * pricing["output"]
        ) / 1_000_000

    def generate_response(self, prompt, *, num_answers=None):
        requested = int(num_answers) if num_answers is not None else 1

        if requested <= 0:
            raise ValueError("num_answers must be positive when provided.")

        prompt = (
            f"{prompt}\n\n"
            "Generate exactly one answer dictionary as `ptx_kernel`."
        )

        all_answers = []
        total_cost = 0.0
        cost_available = True

        while len(all_answers) < requested:
            remaining = requested - len(all_answers)

            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "n": remaining,
            }

            response = self.client.chat.completions.create(**request_kwargs)

            cost = self._estimate_cost(response)
            if cost is None:
                cost_available = False
                print(f"Estimated query cost: unavailable for model {self.model!r}")
            else:
                total_cost += cost
                print(f"Estimated query cost: ${cost:.6f}")

            for choice in response.choices:
                text = choice.message.content or ""

                try:
                    parsed = parse_response_text(text)

                    if len(parsed) != 1:
                        raise ValueError(
                            "Expected exactly one answer per completion, "
                            f"got {len(parsed)}."
                        )

                    all_answers.extend(parsed)
                except Exception:
                    # Bad JSON / malformed response. Ignore this one;
                    # the while loop will re-query the missing answer.
                    pass

        if cost_available:
            print(f"Estimated total query cost: ${total_cost:.6f}")
        else:
            print(f"Estimated total query cost: unavailable for model {self.model!r}")

        return all_answers
