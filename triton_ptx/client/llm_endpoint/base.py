from abc import ABC, abstractmethod
import ast
import json


class LLMEndpoint(ABC):

    @abstractmethod
    def generate_response(self, prompt: str, *, num_answers: int | None = None) -> list[dict]:
        """Generate a response from a prompt."""
        pass


def parse_response_text(text: str) -> list[dict]:
    # Remove markdown fences/backticks if pasted from ChatGPT
    text = text.replace("```json", "").replace("```python", "")
    text = text.replace("```", "").strip()

    obj = None

    # First try strict JSON
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: support Python-like values with triple-quoted strings
        # Example: ptx_kernel = [{"ptx": """..."""}]
        if "=" in text:
            text = text.split("=", 1)[1].strip()
        try:
            obj = ast.literal_eval(text)
        except Exception as e:
            raise ValueError(f"Invalid JSON/Python-like response: {e}") from e

    # Ensure result can be serialized as strict JSON
    try:
        json.dumps(obj)
    except TypeError as e:
        raise ValueError(f"Parsed response is not JSON-serializable: {e}") from e

    if isinstance(obj, dict):
        if "ptx" in obj:
            return [obj]
        answers = obj.get("answers")
        if not isinstance(answers, list):
            raise ValueError("Response dictionary must contain either a 'ptx' key or an 'answers' list.")
        obj = answers

    if not isinstance(obj, list):
        raise ValueError("Response must be a top-level list of answer dictionaries.")

    for index, answer in enumerate(obj, start=1):
        if not isinstance(answer, dict):
            raise ValueError(f"Answer #{index} is not a dictionary.")

    return obj
