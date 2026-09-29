from __future__ import annotations

import pytest
import torch

from ptx_gym.evaluation import OutputVerifier
from ptx_gym.helpers.triton import extract_ptx
from ptx_gym.kernels import kernel_list
from ptx_gym.kernels.base import TritonPTXKernel


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this system.",
)


@pytest.mark.parametrize("kernel_cls", kernel_list, ids=lambda cls: cls.__name__)
def test_extracted_ptx_can_be_injected_and_matches_triton_output(
    kernel_cls: type[TritonPTXKernel],
) -> None:
    """Inject extracted PTX and verify it matches the Triton kernel output."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    verifier = OutputVerifier(iterations=1, seed=0)
    kernel = kernel_cls()
    inputs = kernel.get_random_input()

    _, compiled_kernel = kernel.forward_triton(inputs)
    ptx = extract_ptx(compiled_kernel)

    assert ptx is not None
    assert ".entry" in ptx

    ptx_kernel = kernel_cls(ptx=ptx)
    assert verifier.verify(ptx_kernel)
