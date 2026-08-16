from __future__ import annotations

import pytest
import torch

from triton_ptx.evaluation import OutputVerifier
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.kernels import kernel_list


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this system.",
)


@pytest.fixture(autouse=True)
def disable_autotuning(monkeypatch: pytest.MonkeyPatch) -> None:
    original_init_compiled_kernels = TritonPTXKernel.init_compiled_kernels

    def init_compiled_kernels_without_autotuning(
        self,
        *,
        ptx,
        autotune=True,
        tuning_options=None,
    ):
        return original_init_compiled_kernels(
            self,
            ptx=ptx,
            autotune=False,
            tuning_options=tuning_options,
        )

    monkeypatch.setattr(
        TritonPTXKernel,
        "init_compiled_kernels",
        init_compiled_kernels_without_autotuning,
    )


@pytest.mark.parametrize("kernel_cls", kernel_list, ids=lambda cls: cls.__name__)
def test_kernel_matches_torch(kernel_cls: type) -> None:
    kernel = kernel_cls()
    tolerance = 1e-2
    verifier = OutputVerifier(
        iterations=250,
        seed=42,
        rtol=tolerance,
        atol=tolerance,
    )

    assert verifier.verify_triton_vs_torch(kernel), (
        f"{kernel_cls.__name__} failed against PyTorch: {verifier.last_report}"
    )
