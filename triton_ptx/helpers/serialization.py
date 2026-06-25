from __future__ import annotations

from typing import Any

import torch


def tensor_summary(x: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    }


def tensor_bytes(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()

    if isinstance(x, (list, tuple)):
        return sum(tensor_bytes(v) for v in x)

    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())

    return 0


def dump_nested(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return tensor_summary(x)

    if isinstance(x, (list, tuple)):
        return [dump_nested(v) for v in x]

    if isinstance(x, dict):
        return {k: dump_nested(v) for k, v in x.items()}

    return x
