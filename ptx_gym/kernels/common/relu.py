from pathlib import Path

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin

RELU_SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "relu.spec"


class ReLUFloat16Kernel(TritonPTXKernel):
    input_element_width = 2
    tuning_options = {
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
    def kernel(x_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(output_ptr + offsets, tl.maximum(tl.load(x_ptr + offsets), 0.0))

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
        kernel = launch_kernel[(triton.cdiv(x.numel(), self.block_size),)](
            x, output, x.numel(), BLOCK_SIZE=self.block_size, **launch_kwargs
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.relu(x)

    def volta_arguments(self):
        n_elements = self.batch_size * self.size
        return RELU_SPEC_PATH, [
            "-g",
            str(triton.cdiv(n_elements, self.block_size)),
            "--array",
            f"x:0x100000000:{self.input_element_width}:{n_elements}:in",
            "--array",
            f"output:0x200000000:2:{n_elements}:out",
            "--param",
            "ptr:x",
            "--param",
            "ptr:output",
            "--param",
            "int:0",
            "--param",
            "int:0",
            "--dim",
            f"N={n_elements}",
        ]


class ReLUFloat8Kernel(Float8KernelMixin, ReLUFloat16Kernel):
    input_element_width = 1
    autotune_tolerance = 1e-2
    verification_tolerance = 1e-2

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 1)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 1)

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x, dtype=torch.float16)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(triton.cdiv(x.numel(), self.block_size),)](
            x, output, x.numel(), BLOCK_SIZE=self.block_size, **launch_kwargs
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.relu(x.to(torch.float32)).to(torch.float16)
