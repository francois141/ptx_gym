import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.kernels.vector_workload import (
    BATCH_SIZE,
    VECTOR_SIZE,
    random_softmax_input,
)


class SoftmaxFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.size = VECTOR_SIZE
        self.batch_size = BATCH_SIZE
        self.block_size = VECTOR_SIZE
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.num_warps = 8
        self.num_stages = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_rows,
        BLOCK_SIZE: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        row_start = tl.program_id(axis=0)
        row_step = tl.num_programs(axis=0)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
            input_row_ptr = input_ptr + row_idx * input_row_stride
            values = tl.load(input_row_ptr + col_offsets).to(tl.float32)
            values -= tl.max(values, axis=0)
            numerator = tl.exp(values)
            denominator = tl.sum(numerator, axis=0)
            output_row_ptr = output_ptr + row_idx * output_row_stride
            tl.store(output_row_ptr + col_offsets, numerator / denominator)

    def get_random_input(self, fixed: bool = False):
        return random_softmax_input(self.batch_size, self.size, torch.float16)

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
            output,
            x,
            x.stride(0),
            output.stride(0),
            self.batch_size,
            BLOCK_SIZE=self.block_size,
            num_stages=self.num_stages,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.softmax(x, dim=1)
