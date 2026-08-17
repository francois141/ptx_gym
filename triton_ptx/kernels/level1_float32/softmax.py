import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.kernels.vector_workload import (
    BATCH_SIZE,
    VECTOR_SIZE,
    VECTOR_SIZE_CONSTEXPR,
)


class SoftmaxKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.size = VECTOR_SIZE
        self.batch_size = BATCH_SIZE
        self.block_size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        batch_index = tl.program_id(axis=0)
        vector_offset = batch_index * VECTOR_SIZE_CONSTEXPR
        offsets = tl.arange(0, BLOCK_SIZE)
        maximum = -float("inf")
        for block_offset in tl.range(0, VECTOR_SIZE_CONSTEXPR, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            maximum = tl.maximum(maximum, tl.max(values, axis=0))
        denominator = 0.0
        for block_offset in tl.range(0, VECTOR_SIZE_CONSTEXPR, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            denominator += tl.sum(tl.exp(values - maximum), axis=0)
        for block_offset in tl.range(0, VECTOR_SIZE_CONSTEXPR, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            tl.store(
                output_ptr + vector_offset + block_offset + offsets,
                tl.exp(values - maximum) / denominator,
            )

    def get_random_input(self, fixed: bool = False):
        return torch.randn(
            (self.batch_size, self.size), device="cuda", dtype=torch.float32
        )

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float32 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float32 tensor with shape ({self.batch_size}, {self.size})"
        )

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (self.batch_size,)
        launched_kernel = launch_kernel[grid](
            x,
            output,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, x):
        return torch.softmax(x, dim=1)
