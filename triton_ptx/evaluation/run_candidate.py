from __future__ import annotations

import inspect
from typing import Protocol

import torch

from triton_ptx.evaluation.types import Payload
from triton_ptx.kernels import resolve_kernel


class _CandidateOperator(Protocol):
    """Kernel methods required for a single PTX launch."""

    def get_random_input(self, **kwargs: int) -> object:
        """Create inputs for one candidate launch."""

    def forward_triton(self, inputs: object, ptx: bool = False) -> object:
        """Launch the operator's Triton or PTX implementation."""


def _input_kwargs(operator: _CandidateOperator, input_size: int) -> dict[str, int]:
    signature = inspect.signature(operator.get_random_input)
    size_names = {"size", "m", "n", "k", "h", "w", "length"}
    return {
        name: input_size
        for name, parameter in signature.parameters.items()
        if name.lower() in size_names and parameter.default is not inspect.Parameter.empty
    }


def run_candidate(
    kernel_name: str,
    candidate: dict[str, object],
    input_size: int = 128,
) -> None:
    """Launch a PTX candidate once and wait for CUDA completion.

    This function intentionally performs no correctness checks or benchmarks. It
    provides a minimal launch target for external tools such as Compute Sanitizer.

    Args:
        kernel_name: Registered kernel class used to construct inputs and the grid.
        candidate: PTX text and launch dimensions accepted by ``Payload``.
        input_size: Representative size for configurable random inputs.

    Raises:
        RuntimeError: If CUDA is unavailable.
        ValueError: If the candidate payload or input size is invalid.
    """
    if input_size <= 0:
        raise ValueError("input_size must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run a PTX candidate.")

    payload = Payload.from_input(candidate)
    operator = resolve_kernel(kernel_name)(ptx=payload.to_launch_dict())
    inputs = operator.get_random_input(**_input_kwargs(operator, input_size))
    operator.forward_triton(inputs, ptx=True)
    torch.cuda.synchronize()
