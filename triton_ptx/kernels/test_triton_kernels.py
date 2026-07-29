from __future__ import annotations

import pytest
import torch

from triton_ptx.evaluation import OutputVerifier
from triton_ptx.kernels import kernel_list


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this system.",
)


@pytest.mark.parametrize("kernel_cls", kernel_list, ids=lambda cls: cls.__name__)
def test_kernel_matches_torch(kernel_cls: type) -> None:
    kernel = kernel_cls()
    verifier = OutputVerifier(iterations=50, seed=42)

    assert verifier.verify_triton_vs_torch(kernel), (
        f"{kernel_cls.__name__} failed against PyTorch: {verifier.last_report}"
    )
