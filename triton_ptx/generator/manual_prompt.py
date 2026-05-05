from abc import ABC, abstractmethod
import ast
import json
from pathlib import Path


class ResponseGenerator(ABC):

    @abstractmethod
    def generate_response(self, prompt: str) -> dict:
        """Generate a response from a prompt."""
        pass


def parse_response_text(text: str) -> dict:
    # Remove markdown fences/backticks if pasted from ChatGPT
    text = text.replace("```json", "").replace("```python", "")
    text = text.replace("```", "").strip()

    # First try strict JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: support Python-like dicts with triple-quoted strings
    # Example: ptx_kernel = {"answers": [{"ptx": """..."""}]}
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

    return obj


class ManualPrompt(ResponseGenerator):

    def generate_response(self, prompt: str) -> dict:
        file_path = Path("input.json")

        # input("Write anything when you are done copying the answer in input.json")

        try:
            text = file_path.read_text(encoding="utf-8")
            text = text.strip()
            return parse_response_text(text)
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
