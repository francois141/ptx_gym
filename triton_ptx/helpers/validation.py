from __future__ import annotations

from typing import Any


def positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'Candidate payload field "{name}" must be a positive integer.')
    return value
