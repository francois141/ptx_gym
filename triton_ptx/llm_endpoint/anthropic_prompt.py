from anthropic import Anthropic

from .base import LLMEndpoint, parse_response_text


class AnthropicPrompt(LLMEndpoint):
    DEFAULT_MODEL = "claude-opus-4-8"

    PRICING_PER_1M_TOKENS = {
        # Price estimates in USD per 1M tokens.
        # Keep these aligned with Anthropic API pricing.
        "claude-opus-4-8": {
            "input": 15.00,
            "output": 75.00,
        },
        "claude-opus-4": {
            "input": 15.00,
            "output": 75.00,
        },
        "claude-sonnet-4": {
            "input": 3.00,
            "output": 15.00,
        },
    }

    def __init__(self, model=None, max_tokens=8192):
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.client = Anthropic()

    def _estimate_cost(self, response):
        pricing = self.PRICING_PER_1M_TOKENS.get(self.model)
        usage = getattr(response, "usage", None)

        if pricing is None or usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        return (
            input_tokens * pricing["input"]
            + output_tokens * pricing["output"]
        ) / 1_000_000

    def _extract_text(self, response):
        parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts)

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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            cost = self._estimate_cost(response)
            if cost is None:
                cost_available = False
                print(f"Estimated query cost: unavailable for model {self.model!r}")
            else:
                total_cost += cost
                print(f"Estimated query cost: ${cost:.6f}")

            text = self._extract_text(response)

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
