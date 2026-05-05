import torch
import triton
import triton.language as tl

from triton_ptx.helpers import get_ptx_constexpr
from triton_ptx.kernels.base import TritonPTXOperator

_ptx_kernel = {
    "ptx": None,
    "BLOCK_SIZE": None,
}




class SigmoidOperator(TritonPTXOperator):
    def __init__(self, block_size=1024, ptx=_ptx_kernel):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    def get_random_input(self, size=10_000_000):
        return torch.randn(size, device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx=False):
        x = inputs
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                x,
                output,
                n_elements,
                BLOCK_SIZE=(get_ptx_constexpr(self.ptx, "BLOCK_SIZE") or self.block_size),
            )
        return output, kernel

    def forward_torch(self, inputs):
        x = inputs
        return torch.sigmoid(x)
