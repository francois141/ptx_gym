import torch
import triton
import triton.language as tl

from helpers import jit_fixed_parameters

class AddOperator:
    def __init__(self, *, size=100_000, block_size=1024, ptx=None):
        self.size = size
        self.block_size = block_size
        self.compiled_kernel = jit_fixed_parameters(self.kernel)
        self.ptx = ptx
        if self.ptx is not None:
            self.compiled_kernel_ptx = jit_fixed_parameters(self.kernel, ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        KERNEL_BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * KERNEL_BLOCK_SIZE
        offsets = block_start + tl.arange(0, KERNEL_BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def get_random_input(self):
        x = torch.randn(self.size, device='cuda')
        y = torch.randn(self.size, device='cuda')
        return x, y

    def forward_triton(self, inputs, ptx=False, use_ptx=None):
        x, y = inputs
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['KERNEL_BLOCK_SIZE']),)

        if use_ptx is not None:
            ptx = use_ptx

        if not ptx:
            kernel = self.compiled_kernel[grid](
                x, y, output, n_elements, KERNEL_BLOCK_SIZE=self.block_size
            )
        else:
            assert self.ptx is not None
            kernel = self.compiled_kernel_ptx[grid](
                x, y, output, n_elements, KERNEL_BLOCK_SIZE=self.block_size
            )
        return output, kernel

    def forward_torch(self, inputs):
        x, y = inputs
        return x + y
