import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class ReLUKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, num_warps=4, ptx=None):
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

        self.size = 4096

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        tl.store(output_ptr + offsets, tl.maximum(x, 0.0))

    def get_random_input(self):
        return self._rand_1d(self.size)

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            output,
            n_elements,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.relu(x)
