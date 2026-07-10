from __future__ import annotations

from collections.abc import Mapping
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


def run_candidate(
    kernel_name: str,
    candidate: dict[str, object],
    input_kwargs: Mapping[str, int] | None = None,
) -> None:
    """Launch a PTX candidate once and wait for CUDA completion.

    This function intentionally performs no correctness checks or benchmarks. It
    provides a minimal launch target for external tools such as Compute Sanitizer.

    Args:
        kernel_name: Registered kernel class used to construct inputs and the grid.
        candidate: PTX text and launch dimensions accepted by ``Payload``.
        input_kwargs: Optional integer keyword arguments passed to
            ``get_random_input``.

    Raises:
        RuntimeError: If CUDA is unavailable.
        ValueError: If the candidate payload is invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run a PTX candidate.")

    payload = Payload.from_input(candidate)
    operator = resolve_kernel(kernel_name)(ptx=payload.to_launch_dict())
    inputs = operator.get_random_input(**(dict(input_kwargs) if input_kwargs else {}))
    operator.forward_triton(inputs, ptx=True)
    torch.cuda.synchronize()
