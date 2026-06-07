import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class SumDimKernel(TritonPTXKernel):
    def __init__(self, keepdim=True, block_size=1024, ptx=None):
        self.keepdim = keepdim
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        if tl.program_id(0) != 0:
            return
        output = 0.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            output += tl.sum(inputs, axis=0)
        tl.store(out_ptr, output)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        out = torch.empty((1,), device=inputs.device, dtype=inputs.dtype)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](inputs, out, n_elements, BLOCK_SIZE=self.block_size)
        else:
            kernel = self.require_compiled_ptx()[grid](
                inputs,
                out,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                **self.ptx_launch_kwargs(),
            )

        if not self.keepdim:
            out = out[0]
        return out, kernel

    def forward_torch(self, inputs):
        return torch.sum(inputs, dim=0, keepdim=self.keepdim)
