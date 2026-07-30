"""Triton RMS normalization kernel for Apertus."""

import random

import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class ApertusRMSNormKernel(TritonPTXKernel):
    def __init__(self, *, block_size=128, num_warps=4, ptx=None):
        self.block_size = block_size
        self.num_warps = num_warps
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        row_offset = row * hidden_size
        squared_sum = 0.0
        for block_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            values = tl.load(input_ptr + row_offset + block_offset + offsets).to(
                tl.float32
            )
            squared_sum += tl.sum(values * values, axis=0)

        inverse_rms = tl.rsqrt(squared_sum / hidden_size + eps)
        for block_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            values = tl.load(input_ptr + row_offset + block_offset + offsets).to(
                tl.float32
            )
            normalized = (values * inverse_rms).to(input_ptr.dtype.element_ty)
            weights = tl.load(weight_ptr + block_offset + offsets)
            tl.store(
                output_ptr + row_offset + block_offset + offsets, normalized * weights
            )

    def get_random_input(self):
        hidden_size = random.choice((128, 4096))
        return (
            torch.rand((1, hidden_size), device="cuda", dtype=torch.float16),
            torch.rand(hidden_size, device="cuda", dtype=torch.float16),
            1e-6,
        )

    def get_shape_information(self) -> str:
        return (
            "- input_ptr: float16 tensor with shape (1, hidden_size), where "
            "hidden_size is 128 or 4096\n"
            "- weight_ptr: float16 tensor with shape (hidden_size,)\n"
            "- output_ptr: float16 tensor with shape (1, hidden_size)"
        )

    def forward_triton(self, inputs, ptx=False):
        hidden_states, weight, eps = inputs
        hidden_size = hidden_states.shape[-1]
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
        assert hidden_size in (128, 4096), (
            "RMSNorm hidden size must be 128 or 4096; "
            f"received {hidden_size}."
        )
        flattened_input = hidden_states.reshape(-1, hidden_size)
        output = torch.empty_like(flattened_input)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(flattened_input.shape[0],)](
            flattened_input,
            weight,
            output,
            hidden_size=hidden_size,
            eps=eps,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output.reshape_as(hidden_states), kernel

    def forward_torch(self, inputs):
        hidden_states, weight, eps = inputs
        return (
            hidden_states
            * torch.rsqrt(
                hidden_states.float().square().mean(dim=-1, keepdim=True) + eps
            ).to(hidden_states.dtype)
            * weight
        )
