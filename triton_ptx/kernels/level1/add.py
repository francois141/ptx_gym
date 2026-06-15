
import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

class AddKernel(TritonPTXKernel):

    def __init__(self, *, block_size=1024, num_warps=4, ptx=None):
        self.block_size = block_size
        self.constexpr_values = {"KERNEL_BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

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

    def get_random_input(self, size=10_000_000):
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")
        return x, y

    def forward_triton(self, inputs, ptx=False):
        x, y = inputs
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta["KERNEL_BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                x, y, output, n_elements, KERNEL_BLOCK_SIZE=self.block_size,
                num_warps=self.num_warps,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                x,
                y,
                output,
                n_elements,
                KERNEL_BLOCK_SIZE=self.block_size,
                **self.ptx_launch_kwargs(),
            )
        return output, kernel

    def forward_torch(self, inputs):
        x, y = inputs
        return x + y
