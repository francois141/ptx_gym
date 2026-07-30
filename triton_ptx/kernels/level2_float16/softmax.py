import torch
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class SoftmaxFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, num_warps=8, ptx=None):
        self.size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + offsets).to(tl.float32)
        values -= tl.max(values, axis=0)
        output = tl.exp(values)
        tl.store(output_ptr + offsets, (output / tl.sum(output, axis=0)).to(tl.float16))

    def get_random_input(self):
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
        return torch.softmax(x, dim=0)
