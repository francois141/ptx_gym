import torch
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin


class RMSNormFloat16Kernel(TritonPTXKernel):
    tuning_options = {
        "block_size": (128, 256, 512, 1024, 2048, 4096),
        "num_warps": (4, 8, 16),
    }

    def __init__(self, *, eps=1e-6, ptx=None):
        self.size = 4096
        self.batch_size = 256
        self.block_size = 4096
        self.eps = eps
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx, autotune=True)

    @staticmethod
    def kernel(
        x_ptr, weight_ptr, output_ptr, BLOCK_SIZE: tl.constexpr, EPS: tl.constexpr
    ):
        pid = tl.program_id(axis=0)

        offsets = tl.arange(0, BLOCK_SIZE)
        row_offset = pid * BLOCK_SIZE

        x = tl.load(x_ptr + row_offset + offsets).to(tl.float32)

        squared_sum = tl.sum(x * x, axis=0)
        inverse_rms = tl.rsqrt(squared_sum * (1.0 / BLOCK_SIZE) + EPS)

        weight = tl.load(weight_ptr + row_offset + offsets).to(tl.float32)

        y = x * inverse_rms * weight

        tl.store(
            output_ptr + row_offset + offsets,
            y.to(tl.float16),
        )

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
            f"- x_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- weight_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float16 tensor with shape ({self.batch_size}, {self.size})"
        )

    def forward_triton(self, inputs, ptx=False):
        x, weight = inputs
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(self.batch_size,)](
            x,
            weight,
            output,
            BLOCK_SIZE=self.block_size,
            EPS=self.eps,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x, weight = inputs
        inverse_rms = torch.rsqrt(
            torch.mean(x.square(), dim=1, keepdim=True) + self.eps
        )
        return x * inverse_rms * weight


class RMSNormFloat8Kernel(Float8KernelMixin, RMSNormFloat16Kernel):
    autotune_tolerance = 1e-2
    verification_tolerance = 1e-2

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 2)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 2)

    def forward_triton(self, inputs, ptx=False):
        x, weight = inputs
        output = torch.empty_like(x, dtype=torch.float16)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(self.batch_size,)](
            x,
            weight,
            output,
            BLOCK_SIZE=self.block_size,
            EPS=self.eps,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x, weight = self.float32_inputs(inputs)
        inverse_rms = torch.rsqrt(
            torch.mean(x.square(), dim=1, keepdim=True) + self.eps
        )
        return (x * inverse_rms * weight).to(torch.float16)
