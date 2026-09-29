"""Serializable public API for driving Triton PTX from a remote client.

Every function here accepts and returns only JSON-serializable values
(``str``, ``int``, ``float``, ``bool``, ``None``, ``list``, ``dict``) so the
boundary can be exposed over HTTP without leaking live Python objects.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from ptx_gym.evaluation.evaluate import TritonPTXCandidateEvaluator
from ptx_gym.evaluation.types import Payload
from ptx_gym.helpers.environment import (
    get_ptx_system_config,
    is_gpu_available as _is_gpu_available,
)
from ptx_gym.helpers.kernels import extract_specification_from_operator
from ptx_gym.helpers.ptx import parse_ptx_signature
from ptx_gym.helpers.triton import dump_kernel_ptx as _dump_kernel_ptx
from ptx_gym.kernels import kernel_list, resolve_kernel
from ptx_gym.kernels.base import TritonPTXKernel


def list_kernels() -> list[str]:
    """Return the identifiers of every available kernel."""
    return [operator.__name__ for operator in kernel_list]


def is_gpu_available() -> bool:
    """Return whether a CUDA-capable GPU is available on the host."""
    return _is_gpu_available()


@cache
def _operator_for(kernel_id: str) -> TritonPTXKernel:
    """Return a cached, tuned operator so every call shares one tuning config.

    Autotuning is not deterministic when configurations tie, so building a
    fresh operator per call can report one launch configuration while
    compiling PTX for another.
    """
    return resolve_kernel(kernel_id)()


def dump_kernel_ptx(kernel_id: str) -> str:
    """Return the Triton-generated PTX for a kernel."""
    return _dump_kernel_ptx(_operator_for(kernel_id))


@cache
def _evaluator_for(kernel_id: str) -> TritonPTXCandidateEvaluator:
    """Return a cached evaluator for a kernel to avoid rebuilding it."""
    return TritonPTXCandidateEvaluator(
        resolve_kernel(kernel_id), operator=_operator_for(kernel_id)
    )


def _scalar_or_repr(value: Any) -> Any:
    """Keep JSON-native scalars intact and fall back to ``repr`` otherwise."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def get_kernel_data(kernel_id: str) -> dict[str, Any]:
    """Return the serializable data a client needs to prompt for a kernel.

    Bundles the kernel specification, the Triton-generated PTX signature, and
    the host PTX system configuration into a single JSON-serializable payload.
    """
    operator = _operator_for(kernel_id)
    spec = extract_specification_from_operator(operator)
    version, target, address_size = get_ptx_system_config()
    ptx_signature = parse_ptx_signature(_dump_kernel_ptx(operator))

    return {
        "kernel_id": kernel_id,
        "operator_name": spec.operator_name,
        "kernel_name": spec.kernel_name,
        "source": spec.source,
        "parameters": [
            {"name": parameter.name, "annotation": str(parameter.annotation)}
            for parameter in spec.parameters
        ],
        "constexpr_values": {
            name: _scalar_or_repr(value)
            for name, value in spec.constexpr_values.items()
        },
        "num_warps": spec.num_warps,
        "ptx_signature": [
            {"name": parameter.name, "ptx_type": parameter.ptx_type}
            for parameter in ptx_signature
        ],
        "system": {
            "version": version,
            "target": target,
            "address_size": address_size,
        },
    }


def evaluate_candidate(kernel_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Compile, verify, and benchmark a candidate; return a serializable result.

    ``payload`` is a launch mapping such as
    ``{"ptx": ..., "num_threads_x": ..., ...}``. The returned dict mirrors
    ``EvaluatedCandidate.to_dict`` and includes the computed ``passed`` flag.
    """
    result = _evaluator_for(kernel_id).evaluate(Payload.from_input(payload))
    return json.loads(result.to_json())
