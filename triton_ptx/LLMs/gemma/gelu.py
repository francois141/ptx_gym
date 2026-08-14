"""Triton GELU activation kernel for Gemma."""

import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class GemmaGELUKernel(TritonPTXKernel):
    """Gemma's tanh-approximated GELU activation kernel."""

    def __init__(self, *, ptx=None):
        self.block_size = 1024
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        block = tl.program_id(axis=0)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inner = 0.7978845608028654 * (values + 0.044715 * values * values * values)
        output = 0.5 * values * (1.0 + tl.extra.libdevice.tanh(inner))
        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    def get_random_input(self, fixed: bool = False):
        return torch.rand(10_240, device="cuda", dtype=torch.bfloat16)

    def get_shape_information(self) -> str:
        return (
            "- input_ptr: bfloat16 tensor with shape (1, 10240)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 10240)"
        )

    def forward_triton(self, inputs, ptx=False):
        assert inputs.is_cuda, "GELU requires CUDA input."
        assert inputs.is_contiguous(), "GELU input must be contiguous."
        assert inputs.dtype in (torch.float16, torch.bfloat16), (
            "GELU input must use float16 or bfloat16."
        )
        assert inputs.numel() > 0, "GELU input must not be empty."
        flattened_input = inputs.reshape(-1)
        output = torch.empty_like(flattened_input)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = (triton.cdiv(flattened_input.numel(), self.block_size),)
        kernel = launch_kernel[grid](
            flattened_input,
            output,
            flattened_input.numel(),
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output.reshape_as(inputs), kernel

    def forward_torch(self, inputs):
        return torch.nn.functional.gelu(inputs, approximate="tanh")
