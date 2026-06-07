import torch
import triton
import triton.language as tl

from triton_ptx.helpers import jit_fixed_parameters
from triton_ptx.kernels.base import TritonPTXKernel

class FancyFusedKernel(TritonPTXKernel):
    def __init__(self, *, block_size=1024, ptx=None):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx, jit=jit_fixed_parameters)

    @staticmethod
    def kernel(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        s_x = tl.sin(x)
        r_s_x = tl.maximum(0.0, s_x)
        c_x = tl.cos(r_s_x)
        e_c_x = tl.exp(c_x)
        x_sq = x * x
        sig_x2 = 1.0 / (1.0 + tl.exp(-x_sq))
        result = e_c_x * sig_x2
        tl.store(output_ptr + offsets, result, mask=mask)

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
                **self.ptx_launch_kwargs(),
            )
        return output, kernel

    def forward_torch(self, inputs):
        inner = torch.relu(torch.sin(inputs))
        left = torch.exp(torch.cos(inner))
        right = torch.sigmoid(inputs**2)
        return left * right
