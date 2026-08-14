"""Fixed-shape Triton RMS normalization kernels for Apertus."""

import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


def _initialize(kernel, ptx):
    kernel.block_size = 128
    kernel.num_warps = 4
    kernel.constexpr_values = {"BLOCK_SIZE": kernel.block_size}
    kernel.init_compiled_kernels(ptx=ptx)


def _get_random_input(hidden_size):
    return (
        torch.rand((1, hidden_size), device="cuda", dtype=torch.float16),
        torch.rand(hidden_size, device="cuda", dtype=torch.float16),
        1e-6,
    )


def _get_shape_information(hidden_size):
    return (
        f"- input_ptr: float16 tensor with shape (1, {hidden_size})\n"
        f"- weight_ptr: float16 tensor with shape ({hidden_size},)\n"
        f"- output_ptr: float16 tensor with shape (1, {hidden_size})"
    )


def _forward_triton(kernel, inputs, ptx, hidden_size):
    hidden_states, weight, eps = inputs
    assert hidden_states.is_cuda, "RMSNorm requires CUDA input."
    assert weight.is_cuda, "RMSNorm requires CUDA weights."
    assert hidden_states.device == weight.device, (
        "RMSNorm input and weights must be on the same device."
    )
    assert weight.is_contiguous(), "RMSNorm weights must be contiguous."
    assert weight.shape == (hidden_size,), (
        f"RMSNorm weights must have shape ({hidden_size},)."
    )
    assert weight.dtype == hidden_states.dtype, (
        "RMSNorm weights must match the input dtype."
    )
    assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
        "RMSNorm input must use float16 or bfloat16."
    )
    assert hidden_states.is_contiguous(), "RMSNorm input must be contiguous."
    assert hidden_states.shape[-1] == hidden_size, (
        f"RMSNorm kernel is fixed to hidden size {hidden_size}; "
        f"received {hidden_states.shape[-1]}."
    )
    flattened_input = hidden_states.reshape(-1, hidden_size)
    output = torch.empty_like(flattened_input)
    launch_kernel = kernel.compiled_kernel_ptx if ptx else kernel.compiled_kernel
    launch_kwargs = (
        kernel.ptx_launch_kwargs() if ptx else {"num_warps": kernel.num_warps}
    )
    compiled_kernel = launch_kernel[(flattened_input.shape[0],)](
        flattened_input,
        weight,
        output,
        eps=eps,
        BLOCK_SIZE=kernel.block_size,
        **launch_kwargs,
    )
    return output.reshape_as(hidden_states), compiled_kernel


def _forward_torch(inputs):
    hidden_states, weight, eps = inputs
    return (
        hidden_states
        * torch.rsqrt(
            hidden_states.float().square().mean(dim=-1, keepdim=True) + eps
        ).to(hidden_states.dtype)
        * weight
    )


class ApertusRMSNorm128Kernel(TritonPTXKernel):
    """RMSNorm specialized for Apertus attention-head dimensions."""

    def __init__(self, *, ptx=None):
        _initialize(self, ptx)

    def get_random_input(self, fixed: bool = False):
        return _get_random_input(128)

    def get_shape_information(self) -> str:
        return _get_shape_information(128)

    def forward_triton(self, inputs, ptx=False):
        return _forward_triton(self, inputs, ptx, 128)

    def forward_torch(self, inputs):
        return _forward_torch(inputs)

    @staticmethod
    def kernel(input_ptr, weight_ptr, output_ptr, eps, BLOCK_SIZE: tl.constexpr):
        row_offset = tl.program_id(axis=0) * 128
        offsets = tl.arange(0, BLOCK_SIZE)
        values = tl.load(input_ptr + row_offset + offsets).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / 128 + eps)
        weights = tl.load(weight_ptr + offsets)
        normalized = (values * inverse_rms).to(input_ptr.dtype.element_ty)
        tl.store(output_ptr + row_offset + offsets, normalized * weights)


class ApertusRMSNorm4096Kernel(TritonPTXKernel):
    """RMSNorm specialized for Apertus's model hidden dimension."""

    def __init__(self, *, ptx=None):
        _initialize(self, ptx)

    def get_random_input(self, fixed: bool = False):
        return _get_random_input(4096)

    def get_shape_information(self) -> str:
        return _get_shape_information(4096)

    def forward_triton(self, inputs, ptx=False):
        return _forward_triton(self, inputs, ptx, 4096)

    def forward_torch(self, inputs):
        return _forward_torch(inputs)

    @staticmethod
    def kernel(input_ptr, weight_ptr, output_ptr, eps, BLOCK_SIZE: tl.constexpr):
        row_offset = tl.program_id(axis=0) * 4096
        offsets = tl.arange(0, BLOCK_SIZE)
        squared_sum = 0.0
        for block_offset in tl.range(0, 4096, BLOCK_SIZE):
            values = tl.load(input_ptr + row_offset + block_offset + offsets).to(
                tl.float32
            )
            squared_sum += tl.sum(values * values, axis=0)

        inverse_rms = tl.rsqrt(squared_sum / 4096 + eps)
        for block_offset in tl.range(0, 4096, BLOCK_SIZE):
            values = tl.load(input_ptr + row_offset + block_offset + offsets).to(
                tl.float32
            )
            weights = tl.load(weight_ptr + block_offset + offsets)
            normalized = (values * inverse_rms).to(input_ptr.dtype.element_ty)
            tl.store(
                output_ptr + row_offset + block_offset + offsets, normalized * weights
            )
