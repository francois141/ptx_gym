import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class SoftmaxKernel(TritonPTXKernel):
    def __init__(self, *, num_warps=8, ptx=None):
        self.size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + offsets).to(tl.float32)
        values -= tl.max(values, axis=0)
        numerators = tl.exp(values)
        output = numerators / tl.sum(numerators, axis=0)
        tl.store(output_ptr + offsets, output)

    def get_random_input(self):
        return torch.randn(self.size, device="cuda", dtype=torch.float32)

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (triton.cdiv(self.size, meta["BLOCK_SIZE"]),)
        launched_kernel = launch_kernel[grid](
            x,
            output,
            BLOCK_SIZE=self.size,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, x):
        return torch.softmax(x, dim=0)
