from openai import OpenAI

from .base import LLMEndpoint, parse_response_text


class OpenAIPrompt(LLMEndpoint):
    DEFAULT_MODEL = "gpt-5"
    ALLOWED_REASONING_EFFORTS = {None, "minimal", "low", "medium", "high"}

    PRICING_PER_1M_TOKENS = {
        # Price estimates in USD per 1M tokens.
        # Keep these aligned with https://platform.openai.com/pricing.
        "gpt-5": {
            "input": 2.50,
            "cached_input": 0.25,
            "output": 15.00,
        },
        "gpt-5-mini": {
            "input": 0.25,
            "cached_input": 0.025,
            "output": 2.00,
        },
        "gpt-5-nano": {
            "input": 0.05,
            "cached_input": 0.005,
            "output": 0.40,
        },
        "gpt-4.1": {
            "input": 2.00,
            "cached_input": 0.20,
            "output": 8.00,
        },
        "gpt-4.1-mini": {
            "input": 0.40,
            "cached_input": 0.04,
            "output": 1.60,
        },
        "gpt-4.1-nano": {
            "input": 0.10,
            "cached_input": 0.01,
            "output": 0.40,
        },
        "o4-mini": {
            "input": 1.10,
            "cached_input": 0.275,
            "output": 4.40,
        },
        "o3": {
            "input": 10.00,
            "cached_input": 2.50,
            "output": 40.00,
        },
    }

    def __init__(
        self,
        model=None,
        reasoning_effort="medium",
    ):
        self.model = model or self.DEFAULT_MODEL

        if reasoning_effort not in self.ALLOWED_REASONING_EFFORTS:
            allowed = sorted(
                v for v in self.ALLOWED_REASONING_EFFORTS if v is not None
            )
            raise ValueError(
                f"Invalid reasoning_effort. Expected one of {allowed} or None."
            )

        self.reasoning_effort = reasoning_effort

        self.client = OpenAI()

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
            cached_input_tokens = (
                getattr(prompt_details, "cached_tokens", 0) or 0
            )

        uncached_input_tokens = max(
            input_tokens - cached_input_tokens, 0
        )

        return (
            uncached_input_tokens * pricing["input"]
            + cached_input_tokens * pricing["cached_input"]
            + output_tokens * pricing["output"]
        ) / 1_000_000

    def generate_response(self, prompt, *, num_answers=None):
        requested = (
            int(num_answers) if num_answers is not None else 1
        )

        if requested <= 0:
            raise ValueError(
                "num_answers must be positive when provided."
            )

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

            if self.reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            response = self.client.chat.completions.create(**request_kwargs)

            cost = self._estimate_cost(response)
            if cost is None:
                cost_available = False
                print(
                    f"Estimated query cost: unavailable for "
                    f"model {self.model!r}"
                )
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

        if not cost_available:
            print(
                f"Estimated query cost: unavailable for "
                f"model {self.model!r}"
            )
        else:
            print(
                f"Estimated total query cost: ${total_cost:.6f}"
            )

        return all_answers
