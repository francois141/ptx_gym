import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

class LeakyReLUKernel(TritonPTXKernel):
    def __init__(self, negative_slope=0.01, block_size=1024, ptx=None):
        self.negative_slope = negative_slope
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = tl.where(x >= 0, x, x * negative_slope)
        tl.store(output_ptr + offsets, output, mask=mask)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        output = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                inputs,
                output,
                n_elements,
                self.negative_slope,
                BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                inputs,
                output,
                n_elements,
                self.negative_slope,
                **self.ptx_launch_kwargs(),
            )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.nn.functional.leaky_relu(inputs, negative_slope=self.negative_slope)
