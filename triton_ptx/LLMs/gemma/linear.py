"""Triton linear kernel for Gemma."""

import torch
import triton
import triton.language as tl
from torch.nn import functional

from triton_ptx.kernels.base import TritonPTXKernel


class GemmaLinearKernel(TritonPTXKernel):
    """Gemma's bias-free linear projection kernel."""

    def __init__(self, *, ptx=None):
        self.block_m = 1
        self.block_n = 128
        self.block_k = 32
        self.num_warps = 4
        self.num_stages = 4
        self.constexpr_values = {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
        }
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        input_features,
        input_row_stride,
        weight_output_stride,
        weight_input_stride,
        output_row_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        row_offsets = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        output_offsets = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        input_offsets = tl.arange(0, BLOCK_K)
        input_ptrs = (
            input_ptr + row_offsets[:, None] * input_row_stride + input_offsets[None, :]
        )
        weight_ptrs = (
            weight_ptr
            + output_offsets[None, :] * weight_output_stride
            + input_offsets[:, None] * weight_input_stride
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in tl.range(0, input_features, BLOCK_K):
            accumulator = tl.dot(
                tl.load(input_ptrs), tl.load(weight_ptrs), acc=accumulator
            )
            input_ptrs += BLOCK_K
            weight_ptrs += BLOCK_K * weight_input_stride

        output_ptrs = (
            output_ptr
            + row_offsets[:, None] * output_row_stride
            + output_offsets[None, :]
        )
        tl.store(output_ptrs, accumulator.to(output_ptr.dtype.element_ty))

    def get_random_input(self, fixed: bool = False):
        return (
            torch.rand((1, 2560), device="cuda", dtype=torch.bfloat16),
            torch.rand((2560, 2560), device="cuda", dtype=torch.bfloat16),
        )

    def get_shape_information(self) -> str:
        return (
            "- input_ptr: bfloat16 tensor with shape (1, 2560)\n"
            "- weight_ptr: bfloat16 tensor with shape (2560, 2560)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 2560)"
        )

    def forward_triton(self, inputs, ptx=False):
        hidden_states, weight = inputs
        input_features = hidden_states.shape[-1]
        output_features, weight_input_features = weight.shape
        assert hidden_states.is_cuda, "Linear requires CUDA input."
        assert hidden_states.is_contiguous(), "Linear input must be contiguous."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
            "Linear input must use float16 or bfloat16."
        )
        assert weight.is_cuda, "Linear weight must be on CUDA."
        assert weight.is_contiguous(), "Linear weight must be contiguous."
        assert weight.dtype == hidden_states.dtype, "Weight must match input dtype."
        assert weight.device == hidden_states.device, (
            "Weight must be on the input device."
        )
        assert weight_input_features == input_features, (
            "Weight input features must match input."
        )
        assert input_features % self.block_k == 0, (
            f"Input features must be divisible by {self.block_k}."
        )
        assert output_features % self.block_n == 0, (
            f"Output features must be divisible by {self.block_n}."
        )
        flattened_input = hidden_states.reshape(-1, input_features).contiguous()
        output = torch.empty(
            (flattened_input.shape[0], output_features),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        kernel = launch_kernel[
            (
                triton.cdiv(flattened_input.shape[0], self.block_m),
                triton.cdiv(output_features, self.block_n),
            )
        ](
            flattened_input,
            weight,
            output,
            input_features,
            flattened_input.stride(0),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output.reshape(*hidden_states.shape[:-1], output_features), kernel

    def forward_torch(self, inputs):
        hidden_states, weight = inputs
        return functional.linear(hidden_states, weight)
