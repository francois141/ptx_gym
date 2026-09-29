import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin


class ReductionSumFloat16Kernel(TritonPTXKernel):
    tuning_options = {
        "block_size": (32, 64, 128, 256, 512, 1024, 2048),
        "num_warps": (4, 8, 16),
    }

    def __init__(self, *, ptx=None):
        self.size = 4096
        self.batch_size = 256
        self.block_size = 1024
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        batch_index = tl.program_id(axis=1)
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + batch_index * 4096 + offsets).to(tl.float32)
        tl.atomic_add(output_ptr + batch_index, tl.sum(values, axis=0))

    def get_random_input(self, fixed: bool = False):
        return torch.rand(
            (self.batch_size, self.size), device="cuda", dtype=torch.float16
        )

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float32 tensor with shape ({self.batch_size},)"
        )

    def forward_triton(self, x, ptx=False):
        output = torch.zeros(self.batch_size, device=x.device, dtype=torch.float32)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (
            triton.cdiv(self.size, meta["BLOCK_SIZE"]),
            self.batch_size,
        )
        launched_kernel = launch_kernel[grid](
            x,
            output,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, x):
        return torch.sum(x, dim=1, dtype=torch.float32)


class ReductionSumFloat8Kernel(Float8KernelMixin, ReductionSumFloat16Kernel):
    autotune_tolerance = 1e-2
    verification_tolerance = 1e-2

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 1)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 1)

    def forward_torch(self, x):
        return torch.sum(x.to(torch.float32), dim=1, dtype=torch.float32)
