from pathlib import Path

import torch
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin

SOFTMAX_SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "softmax.spec"


class SoftmaxFloat16Kernel(TritonPTXKernel):
    input_element_width = 2
    tuning_options = {
        "num_warps": (4,8, 16),
        "num_stages": (2, 3),
    }

    def __init__(self, *, ptx=None):
        self.size = 256
        self.batch_size = 4096
        self.block_size = 256
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
            tl.store(output_row_ptr + col_offsets, numerator / denominator)

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

    def volta_arguments(self):
        elements = self.batch_size * self.size
        return SOFTMAX_SPEC_PATH, [
            "-g",
            str(self.batch_size),
            "--array",
            f"input:0x100000000:{self.input_element_width}:{elements}:in",
            "--array",
            f"output:0x200000000:2:{elements}:out",
            "--param",
            "ptr:output",
            "--param",
            "ptr:input",
            "--param",
            f"int:{self.size}",
            "--param",
            f"int:{self.size}",
            "--param",
            f"int:{self.batch_size}",
            "--param",
            f"int:{self.size}",
            "--param",
            "int:0",
            "--param",
            "int:0",
            "--dim",
            f"B={self.batch_size}",
            "--dim",
            f"D={self.size}",
        ]


class SoftmaxFloat8Kernel(Float8KernelMixin, SoftmaxFloat16Kernel):
    input_element_width = 1
    autotune_tolerance = 1e-1
    verification_tolerance = 1e-1

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 1)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 1)

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x, dtype=torch.float16)
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
        return torch.softmax(x.to(torch.float32), dim=1).to(torch.float16)
