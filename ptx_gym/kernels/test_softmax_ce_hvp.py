"""HVP-specific numerical and tail-mask coverage beyond the shared kernel tests."""

import pytest
import torch
import torch.nn.functional as F

from ptx_gym.kernels.common.softmax_ce_hvp import SOFTMAX_CE_HVP_KERNELS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for HVP kernel tests"
)


@pytest.mark.parametrize("kernel_cls", SOFTMAX_CE_HVP_KERNELS)
@pytest.mark.parametrize(
    "classes", [1, 2, 31, 32, 33, 255, 256, 257, 1000, 1024, 1025, 4096, 8192]
)
def test_hvp_tail_masks_and_input_preservation(kernel_cls, classes):
    torch.manual_seed(42)
    op = kernel_cls(batch_size=3, num_classes=classes, autotune=False)
    inputs = op.get_random_input()
    originals = [value.clone() for value in inputs]
    actual, _ = op.forward_triton(inputs)
    assert actual.dtype == torch.float32
    assert actual.shape == (3, classes)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, op.forward_torch(inputs), rtol=1e-6, atol=1e-6)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("kernel_cls", SOFTMAX_CE_HVP_KERNELS)
def test_hvp_matches_independent_float64_autograd(kernel_cls):
    torch.manual_seed(7)
    op = kernel_cls(batch_size=2, num_classes=33, autotune=False)
    logits = torch.randn(2, 33, device="cuda", dtype=op.vector_dtype)
    vector = torch.randn_like(logits)
    first = logits.float().softmax(-1) if op.cached_probabilities else logits
    actual, _ = op.forward_triton((first, vector))

    # Use CPU float64 for the independent loss derivative, including on GPUs
    # with limited FP64 throughput. Cached probabilities have FP32 roundoff.
    z = logits.cpu().double().requires_grad_()
    labels = torch.tensor([0, 1])
    loss = F.cross_entropy(z, labels, reduction="sum")
    gradient = torch.autograd.grad(loss, z, create_graph=True)[0]
    expected = torch.autograd.grad((gradient * vector.cpu().double()).sum(), z)[0]
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-5, atol=1e-6)
    assert not torch.allclose(actual, torch.zeros_like(actual), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kernel_cls", SOFTMAX_CE_HVP_KERNELS)
def test_hvp_extreme_logits_and_nullspace(kernel_cls):
    op = kernel_cls(batch_size=2, num_classes=33, autotune=False)
    logits = torch.full((2, 33), -10000.0, device="cuda", dtype=op.vector_dtype)
    logits[0, 0] = 10000.0
    first = logits.float().softmax(-1) if op.cached_probabilities else logits
    for vector in (
        torch.zeros_like(logits),
        torch.ones_like(logits),
        torch.randn_like(logits),
    ):
        actual, _ = op.forward_triton((first, vector))
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual, op.forward_torch((first, vector)), rtol=1e-6, atol=1e-6
        )
        if torch.all(vector == 0) or torch.all(vector == 1):
            torch.testing.assert_close(
                actual, torch.zeros_like(actual), rtol=0, atol=1e-6
            )
