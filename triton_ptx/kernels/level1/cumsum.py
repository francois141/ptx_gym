import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class CumsumKernel(TritonPTXKernel):

    def __init__(self, block_size=1024, num_warps=4, ptx=None):
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(out_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        if tl.program_id(0) != 0:
            return
        carry = 0.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            output = tl.cumsum(inputs, axis=0) + carry
            tl.store(out_ptr + offsets, output, mask=mask)
            carry += tl.sum(inputs, axis=0)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        out = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if not ptx:
            kernel = self.compiled_kernel[grid](out, inputs, n_elements, BLOCK_SIZE=self.block_size, num_warps=self.num_warps,)
        else:
            kernel = self.require_compiled_ptx()[grid](
                out,
                inputs,
                n_elements,
                BLOCK_SIZE=self.block_size,
                **self.ptx_launch_kwargs(),
            )
        return out, kernel

    def forward_torch(self, inputs):
        return torch.cumsum(inputs, dim=0)
