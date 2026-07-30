"""Triton linear kernel for Apertus."""

import random

import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel


class LinearKernel(TritonPTXKernel):
    def __init__(
        self, *, block_n=128, block_k=32, num_warps=4, num_stages=4, ptx=None
    ):
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.constexpr_values = {
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        }
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        input_features,
        output_features,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        vector_index = tl.program_id(0)
        program_n = tl.program_id(1)
        output_offsets = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        input_offsets = tl.arange(0, BLOCK_K)
        input_ptrs = input_ptr + vector_index * input_features + input_offsets
        weight_ptrs = (
            weight_ptr
            + output_offsets[None, :] * input_features
            + input_offsets[:, None]
        )
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for _ in tl.range(0, input_features, BLOCK_K):
            accumulator += tl.sum(
                tl.load(input_ptrs).to(tl.float32)[:, None]
                * tl.load(weight_ptrs).to(tl.float32),
                axis=0,
            )
            input_ptrs += BLOCK_K
            weight_ptrs += BLOCK_K

        output_ptrs = output_ptr + vector_index * output_features + output_offsets
        tl.store(output_ptrs, accumulator.to(output_ptr.dtype.element_ty))

    def get_random_input(self):
        input_features = random.randrange(1, 5) * 1024
        output_features = random.randrange(1, 5) * 1024
        return (
            torch.rand((1, input_features), device="cuda", dtype=torch.float16),
            torch.rand(
                (output_features, input_features),
                device="cuda",
                dtype=torch.float16,
            ),
        )

    def get_shape_information(self) -> str:
        return (
            "- input_ptr: float16 tensor with shape "
            "(1, input_features), where input_features is a multiple of 1024\n"
            "- weight_ptr: float16 tensor with shape "
            "(output_features, input_features), where output_features is a "
            "multiple of 1024\n"
            "- output_ptr: float16 tensor with shape (..., output_features)"
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
        assert input_features % 1024 == 0, (
            "Input features must be a multiple of 1024."
        )
        assert output_features % 1024 == 0, (
            "Output features must be a multiple of 1024."
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
                flattened_input.shape[0],
                triton.cdiv(output_features, self.block_n),
            )
        ](
            flattened_input,
            weight,
            output,
            input_features,
            output_features,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output.reshape(*hidden_states.shape[:-1], output_features), kernel

    def forward_torch(self, inputs):
        hidden_states, weight = inputs
        return functional.linear(hidden_states, weight)
