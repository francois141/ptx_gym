import torch
import torch.nn.functional as functional
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class SiLUFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.size}
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + offsets)
        values = values.to(tl.float32)
        tl.store(output_ptr + offsets, values / (1.0 + tl.exp(-values)))

    def get_random_input(self, fixed: bool = False):
        return torch.rand(self.size, device="cuda", dtype=torch.float16)

    def get_shape_information(self) -> str:
        return (
            "- x_ptr: float16 tensor with shape (4096,)\n"
            "- output_ptr: float16 tensor with shape (4096,)"
        )

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(1,)](x, output, BLOCK_SIZE=self.size, **launch_kwargs)
        return output, kernel

    def forward_torch(self, x):
        return functional.silu(x)
