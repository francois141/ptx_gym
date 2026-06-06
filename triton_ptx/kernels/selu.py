import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {
    "ptx": None,
    "BLOCK_SIZE": None,
}


class SELUKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=_ptx_kernel):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        x32 = x.to(tl.float32)
        output = scale * tl.where(x32 > 0, x32, alpha * (tl.exp(x32) - 1.0))
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        output = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                inputs, output, n_elements, BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                inputs,
                output,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                num_warps=self.ptx["num_warps"],
            )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.nn.functional.selu(inputs)
