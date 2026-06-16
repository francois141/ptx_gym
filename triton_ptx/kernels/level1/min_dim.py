import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MinDimKernel(TritonPTXKernel):

    def __init__(self, dim=0, block_size=1024, num_warps=4, ptx=None):
        self.dim = dim
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        if tl.program_id(0) != 0:
            return
        output = float("inf")
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
            output = tl.minimum(output, tl.min(inputs, axis=0))
        tl.store(out_ptr, output)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        out = torch.empty((1,), device=inputs.device, dtype=inputs.dtype)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            inputs,
            out,
            n_elements,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return out[0], kernel
        return torch.min(inputs, dim=self.dim)[0], kernel

    def forward_torch(self, inputs):
        return torch.min(inputs, dim=self.dim)[0]
