import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class SoftmaxFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.size = 4096
        self.batch_size = 256
        self.block_size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        batch_index = tl.program_id(axis=0)
        vector_offset = batch_index * 4096
        offsets = tl.arange(0, BLOCK_SIZE)
        maximum = -float("inf")
        for block_offset in tl.range(0, 4096, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            maximum = tl.maximum(maximum, tl.max(values, axis=0))
        denominator = 0.0
        for block_offset in tl.range(0, 4096, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            denominator += tl.sum(tl.exp(values - maximum), axis=0)
        for block_offset in tl.range(0, 4096, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            tl.store(
                output_ptr + vector_offset + block_offset + offsets,
                (tl.exp(values - maximum) / denominator).to(tl.float16),
            )

    def get_random_input(self, fixed: bool = False):
        return torch.rand(
            (self.batch_size, self.size), device="cuda", dtype=torch.float16
        )

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float16 tensor with shape ({self.batch_size}, {self.size})"
        )

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(self.batch_size,)](
            x, output, BLOCK_SIZE=self.block_size, **launch_kwargs
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.softmax(x, dim=1)
