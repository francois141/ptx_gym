import json
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from .base import LLMEndpoint, parse_response_text


class OpenAIPrompt(LLMEndpoint):
    DEFAULT_MODEL = "gpt-5"
    ALLOWED_REASONING_EFFORTS = {None, "minimal", "low", "medium", "high"}

    PRICING_PER_1M_TOKENS = {
        "gpt-5": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
        "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
        "gpt-4.1": {"input": 2.00, "cached_input": 0.20, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.04, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.01, "output": 0.40},
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        "o3": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
    }

    WEB_SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for up-to-date documentation, Triton APIs, "
                "CUDA specifications, PTX references, or implementation details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, model=None, reasoning_effort="medium"):
        self.model = model or self.DEFAULT_MODEL

        if reasoning_effort not in self.ALLOWED_REASONING_EFFORTS:
            allowed = sorted(v for v in self.ALLOWED_REASONING_EFFORTS if v is not None)
            raise ValueError(
                f"Invalid reasoning_effort. Expected one of {allowed} or None."
            )

        self.reasoning_effort = reasoning_effort
        self.client = OpenAI()

    def web_search(self, query, max_results=5):
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.select(".result")[:max_results]:
            title_node = result.select_one(".result__title a")
            snippet_node = result.select_one(".result__snippet")

            if not title_node:
                continue

            results.append(
                {
                    "title": title_node.get_text(" ", strip=True),
                    "url": title_node.get("href"),
                    "snippet": (
                        snippet_node.get_text(" ", strip=True)
                        if snippet_node
                        else ""
                    ),
                }
            )

        return {
            "query": query,
            "results": results,
        }

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

        return (
            uncached_input_tokens * pricing["input"]
            + cached_input_tokens * pricing["cached_input"]
            + output_tokens * pricing["output"]
        ) / 1_000_000

    def _create_completion(self, messages, *, n=1):
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": [self.WEB_SEARCH_TOOL],
            "tool_choice": "auto",
            "n": n,
            "response_format": {"type": "json_object"},
        }

        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort

        return self.client.chat.completions.create(**request_kwargs)

    def _run_tool_loop(self, messages, *, max_tool_rounds=50):
        total_cost = 0.0
        cost_available = True

        for _ in range(max_tool_rounds):
            response = self._create_completion(messages)

            cost = self._estimate_cost(response)
            if cost is None:
                cost_available = False
            else:
                total_cost += cost
                print(f"Estimated query cost: ${cost:.6f}")

            choice = response.choices[0]
            message = choice.message

            if not message.tool_calls:
                return message.content or "", total_cost, cost_available

            messages.append(message)

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")

                if function_name != "web_search":
                    tool_result = {
                        "error": f"Unknown tool: {function_name}",
                    }
                else:
                    tool_result = self.web_search(
                        query=arguments["query"],
                        max_results=arguments.get("max_results", 5),
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        raise RuntimeError("Model kept requesting tools and did not produce final JSON.")

    def generate_response(self, prompt, *, num_answers=None):
        requested = int(num_answers) if num_answers is not None else 1

        if requested <= 0:
            raise ValueError("num_answers must be positive when provided.")

        base_prompt = (
            f"{prompt}\n\n"
            "You may call web_search when current docs or facts are needed. "
            "Use the search results as context, then answer.\n\n"
            "Return JSON only. No markdown, no prose outside JSON. " 
            "Tensor cores must contribute to the answer and if they are not relevant to the final answer this is not a valid response." \
            "Dummy tensor cores aren't allowed"
        )

        all_answers = []
        total_cost = 0.0
        cost_available = True

        while len(all_answers) < requested:
            print("Again in the loop " + str(len(all_answers)))

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code generation assistant. "
                        "When you need fresh or external facts, call web_search. "
                        "Final output must be valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": base_prompt,
                },
            ]

            try:
                text, cost, available = self._run_tool_loop(messages)
                total_cost += cost
                cost_available = cost_available and available

                print(text)

                parsed = parse_response_text(text)

                if len(parsed) != 1:
                    raise ValueError(
                        "Expected exactly one answer per completion, "
                        f"got {len(parsed)}."
                    )

                all_answers.extend(parsed)

            except Exception as exc:
                print("Failed to parse json or complete request")

                print(str(exc))

        if not cost_available:
            print(f"Estimated query cost: unavailable for model {self.model!r}")
        else:
            print(f"Estimated total query cost: ${total_cost:.6f}")

        return all_answers