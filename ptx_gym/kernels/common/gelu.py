import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin


class GELUFloat16Kernel(TritonPTXKernel):
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
        x = tl.load(x_ptr + offsets).to(tl.float32)
        output = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        tl.store(output_ptr + offsets, output.to(tl.float16))

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
        return torch.nn.functional.gelu(x)


class GELUFloat8Kernel(Float8KernelMixin, GELUFloat16Kernel):
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
        return torch.nn.functional.gelu(x.to(torch.float32)).to(torch.float16)
