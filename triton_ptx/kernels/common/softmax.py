import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class SoftmaxFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.size = 256
        self.batch_size = 4096
        self.block_size = 256
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
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        row_start = tl.program_id(0)
        row_step = tl.num_programs(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
            row_ptr = input_ptr + row_idx * input_row_stride
            row = tl.load(row_ptr + col_offsets)
            row_minus_max = row - tl.max(row, axis=0)
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator, axis=0)

            output_row_ptr = output_ptr + row_idx * output_row_stride
            tl.store(
                output_row_ptr + col_offsets,
                numerator / denominator
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
        n_rows, n_cols = x.shape
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(self.batch_size,)](
            output,
            x,
            x.stride(0),
            output.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=self.block_size,
            num_stages=self.num_stages,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.softmax(x, dim=1)
