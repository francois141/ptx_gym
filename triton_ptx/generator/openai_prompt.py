from pathlib import Path

from openai import OpenAI

from triton_ptx.generator import ResponseGenerator, parse_response_text


class OpenAIPrompt(ResponseGenerator):
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
        api_key_path="openai_api_key.txt",
        reasoning_effort=None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.api_key_path = Path(api_key_path)
        if reasoning_effort not in self.ALLOWED_REASONING_EFFORTS:
            raise ValueError(
                "Invalid reasoning_effort. Expected one of: "
                f"{sorted(v for v in self.ALLOWED_REASONING_EFFORTS if v is not None)} or None."
            )
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=self._read_api_key())

    def _read_api_key(self):
        api_key = self.api_key_path.read_text(encoding="utf-8").strip()

        if not api_key:
            raise ValueError("OpenAI API key file is empty")

        return api_key

    def _estimate_cost(self, response):
        pricing = self.PRICING_PER_1M_TOKENS.get(self.model)

        if pricing is None:
            return None

        usage = response.usage

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        input_details = getattr(usage, "input_tokens_details", None)
        cached_input_tokens = 0
        if input_details is not None:
            cached_input_tokens = getattr(input_details, "cached_tokens", 0) or 0

        uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)

        cost = (
            uncached_input_tokens * pricing["input"]
            + cached_input_tokens * pricing["cached_input"]
            + output_tokens * pricing["output"]
        ) / 1_000_000

        return cost

    def generate_response(self, prompt):
        request_kwargs = {
            "model": self.model,
            "input": prompt,
        }
        if self.reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": self.reasoning_effort}

        response = self.client.responses.create(**request_kwargs)

        cost = self._estimate_cost(response)
        if cost is not None:
            print(f"Estimated query cost: ${cost:.6f}")
        else:
            print(f"Estimated query cost: unavailable for model {self.model!r}")

        return parse_response_text(response.output_text)
