import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {"ptx": None, "BLOCK_SIZE": None}


class CumprodKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=_ptx_kernel):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(out_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        if tl.program_id(0) != 0:
            return
        carry = 1.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(x_ptr + offsets, mask=mask, other=1.0)
            output = tl.cumprod(inputs, axis=0) * carry
            tl.store(out_ptr + offsets, output, mask=mask)
            carry = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == BLOCK_SIZE - 1, output, 0.0), axis=0)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size) + 1e-3

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        out = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if not ptx:
            kernel = self.compiled_kernel[grid](out, inputs, n_elements, BLOCK_SIZE=self.block_size)
        else:
            kernel = self.require_compiled_ptx()[grid](
                out,
                inputs,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                num_warps=self.ptx["num_warps"],
            )
        return out, kernel

    def forward_torch(self, inputs):
        return torch.cumprod(inputs, dim=0)
