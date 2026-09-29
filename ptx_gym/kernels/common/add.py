from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin


class AddFloat16Kernel(TritonPTXKernel):
    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_size": (32, 64, 128, 256, 512, 1024, 2048),
        "num_warps": (4, 8, 16),
    }

    def __init__(self, *, ptx=None):
        self.block_size = 1024
        self.size = 4096
        self.batch_size = 256
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        left_ptr,
        right_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        left = tl.load(left_ptr + offsets).to(tl.float16)
        right = tl.load(right_ptr + offsets).to(tl.float16)
        output = left + right
        tl.store(output_ptr + offsets, output)

    def get_random_input(self, fixed: bool = False):
        return (
            torch.rand(
                (self.batch_size, self.size), device="cuda", dtype=torch.float16
            ),
            torch.rand(
                (self.batch_size, self.size), device="cuda", dtype=torch.float16
            ),
        )

    def get_shape_information(self) -> str:
        return (
            f"- left_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- right_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float16 tensor with shape ({self.batch_size}, {self.size})"
        )

    def forward_triton(self, inputs, ptx=False):
        left, right = inputs
        output = torch.empty_like(left)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(triton.cdiv(left.numel(), self.block_size),)](
            left,
            right,
            output,
            left.numel(),
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        left, right = inputs
        return left + right


class AddFloat8Kernel(Float8KernelMixin, AddFloat16Kernel):
    autotune_tolerance = 1e-2
    verification_tolerance = 1e-2

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 2)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 2)

    def forward_triton(self, inputs, ptx=False):
        left, right = inputs
        output = torch.empty_like(left, dtype=torch.float16)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(triton.cdiv(left.numel(), self.block_size),)](
            left,
            right,
            output,
            left.numel(),
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        left, right = self.float32_inputs(inputs)
        return (left + right).to(torch.float16)
