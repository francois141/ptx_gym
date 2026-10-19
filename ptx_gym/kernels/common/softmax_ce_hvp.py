"""Row-wise cross-entropy Hessian actions: p * (v - sum(p * v)).

The loss is unweighted cross-entropy summed over examples, without batch-mean
scaling. Fused variants compute p = softmax(logits) in FP32. Cached variants
accept normalized FP32 probabilities. All reductions and outputs use FP32.
"""

from typing import ClassVar

import torch
import triton
import triton.language as tl

from ptx_gym.kernels.base import TritonPTXKernel

DEFAULT_BATCH_SIZE = 4096
DEFAULT_NUM_CLASSES = 1000
MAX_CLASSES = 8192


def _validate_inputs(first, vector, *, cached):
    if not isinstance(first, torch.Tensor) or not isinstance(vector, torch.Tensor):
        raise TypeError("HVP inputs must be Torch tensors.")
    if first.ndim != 2 or vector.ndim != 2:
        raise ValueError("HVP inputs must have rank 2: [batch, classes].")
    if first.shape != vector.shape or first.numel() == 0:
        raise ValueError("HVP inputs must have matching, nonempty shapes.")
    if first.device != vector.device:
        raise ValueError("HVP inputs must be on the same device.")
    if not first.is_contiguous() or not vector.is_contiguous():
        raise ValueError("HVP inputs must be contiguous.")
    if vector.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("HVP vectors must use float32, float16 or bfloat16.")
    if cached:
        if first.dtype != torch.float32:
            raise TypeError("Cached probabilities must use float32.")
    elif first.dtype != vector.dtype:
        raise TypeError("Fused HVP logits and vectors must have matching dtypes.")


def _torch_hvp(first, vector, *, cached):
    _validate_inputs(first, vector, cached=cached)
    probability = first if cached else torch.softmax(first.float(), dim=-1)
    vector_fp32 = vector.float()
    projection = torch.sum(probability * vector_fp32, dim=-1, keepdim=True)
    return probability * (vector_fp32 - projection)


class _SoftmaxCrossEntropyHVPBase(TritonPTXKernel):
    vector_dtype = torch.float32
    cached_probabilities = False
    tuning_options: ClassVar = {"num_warps": (1, 2, 4, 8, 16)}
    autotune_tolerance = 1e-6
    verification_tolerance = 1e-6

    def __init__(
        self,
        *,
        ptx=None,
        batch_size=DEFAULT_BATCH_SIZE,
        num_classes=DEFAULT_NUM_CLASSES,
        autotune=True,
    ):
        for name, value in (("batch_size", batch_size), ("num_classes", num_classes)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if num_classes > MAX_CLASSES:
            raise ValueError(f"num_classes must be at most {MAX_CLASSES}.")
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.block_size = triton.next_power_of_2(num_classes)
        self.num_warps = 4
        self.num_stages = 1
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx, autotune=autotune)

    def get_random_input(self, fixed=False):
        del fixed  # Each registered task keeps its canonical shape on every trial.
        shape = (self.batch_size, self.num_classes)
        if self.cached_probabilities:
            first = torch.randn(shape, device="cuda", dtype=torch.float32).softmax(-1)
        else:
            first = torch.randn(shape, device="cuda", dtype=self.vector_dtype)
        vector = torch.randn(shape, device="cuda", dtype=self.vector_dtype)
        return first, vector

    def get_shape_information(self):
        shape = (self.batch_size, self.num_classes)
        dtype = str(self.vector_dtype).removeprefix("torch.")
        input_dtype = "float32" if self.cached_probabilities else dtype
        meaning = (
            "cached softmax probabilities" if self.cached_probabilities else "logits"
        )
        return (
            f"- output_ptr: float32 tensor with shape {shape}\n"
            f"- input_ptr: {input_dtype} tensor with shape {shape} ({meaning})\n"
            f"- vector_ptr: {dtype} tensor with shape {shape}\n"
            "All tensors are contiguous; the loss uses sum reduction, not batch mean."
        )

    def forward_triton(self, inputs, ptx=False):
        first, vector = inputs
        _validate_inputs(first, vector, cached=self.cached_probabilities)
        if first.device.type != "cuda":
            raise ValueError("Triton HVP inputs must be on CUDA.")
        if first.shape != (self.batch_size, self.num_classes):
            raise ValueError(
                "Input shape must match this operator's fixed problem shape."
            )
        if vector.dtype != self.vector_dtype:
            raise TypeError("Vector dtype must match this registered kernel variant.")
        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        if launch is None:
            raise RuntimeError("No PTX has been installed on this operator.")
        output = torch.empty_like(first, dtype=torch.float32)
        kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        compiled = launch[(self.batch_size,)](
            output,
            first,
            vector,
            self.num_classes,
            BLOCK_SIZE=self.block_size,
            **kwargs,
        )
        return output, compiled

    def forward_torch(self, inputs):
        first, vector = inputs
        return _torch_hvp(first, vector, cached=self.cached_probabilities)


class SoftmaxCrossEntropyHVPFloat32Kernel(_SoftmaxCrossEntropyHVPBase):
    """Fused FP32 logits/vector -> FP32 softmax cross-entropy HVP."""

    @staticmethod
    def kernel(output_ptr, input_ptr, vector_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK_SIZE)
        offset = row * n_cols + col
        mask = col < n_cols
        logits = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        vector = tl.load(vector_ptr + offset, mask=mask, other=0).to(tl.float32)
        numerator = tl.exp(logits - tl.max(logits, axis=0))
        probability = numerator / tl.sum(numerator, axis=0)
        projection = tl.sum(probability * vector, axis=0)
        tl.store(output_ptr + offset, probability * (vector - projection), mask=mask)


class SoftmaxCrossEntropyHVPFloat16Kernel(SoftmaxCrossEntropyHVPFloat32Kernel):
    """Fused FP16 logits/vector, with FP32 softmax, reductions and output."""

    vector_dtype = torch.float16


class SoftmaxCrossEntropyHVPBFloat16Kernel(SoftmaxCrossEntropyHVPFloat32Kernel):
    """Fused BF16 logits/vector, with FP32 softmax, reductions and output."""

    vector_dtype = torch.bfloat16


class SoftmaxCrossEntropyHVPCachedFloat32Kernel(_SoftmaxCrossEntropyHVPBase):
    """Cached FP32 probabilities/vector -> FP32 Hessian action."""

    cached_probabilities = True

    @staticmethod
    def kernel(output_ptr, input_ptr, vector_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK_SIZE)
        offset = row * n_cols + col
        mask = col < n_cols
        probability = tl.load(input_ptr + offset, mask=mask, other=0).to(tl.float32)
        vector = tl.load(vector_ptr + offset, mask=mask, other=0).to(tl.float32)
        projection = tl.sum(probability * vector, axis=0)
        tl.store(output_ptr + offset, probability * (vector - projection), mask=mask)


class SoftmaxCrossEntropyHVPCachedFloat16Kernel(
    SoftmaxCrossEntropyHVPCachedFloat32Kernel
):
    """Cached FP32 probabilities and FP16 vector; FP32 reductions/output."""

    vector_dtype = torch.float16


class SoftmaxCrossEntropyHVPCachedBFloat16Kernel(
    SoftmaxCrossEntropyHVPCachedFloat32Kernel
):
    """Cached FP32 probabilities and BF16 vector; FP32 reductions/output."""

    vector_dtype = torch.bfloat16


SOFTMAX_CE_HVP_KERNELS = (
    SoftmaxCrossEntropyHVPFloat32Kernel,
    SoftmaxCrossEntropyHVPFloat16Kernel,
    SoftmaxCrossEntropyHVPBFloat16Kernel,
    SoftmaxCrossEntropyHVPCachedFloat32Kernel,
    SoftmaxCrossEntropyHVPCachedFloat16Kernel,
    SoftmaxCrossEntropyHVPCachedBFloat16Kernel,
)
