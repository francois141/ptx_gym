import torch
import triton
import triton.language as tl

from helpers import jit_fixed_parameters


class FancyFusedOperator:
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

    def get_random_input(self):
        return torch.randn(self.size, device="cuda")

    def forward_triton(self, inputs, ptx=False, use_ptx=None):
        x = inputs
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if use_ptx is not None:
            ptx = use_ptx

        if not ptx:
            kernel = self.compiled_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=self.block_size
            )
        else:
            assert self.ptx is not None
            kernel = self.compiled_kernel_ptx[grid](
                x, output, n_elements, BLOCK_SIZE=self.block_size
            )
        return output, kernel

    def forward_torch(self, inputs):
        x = inputs
        inner = torch.relu(torch.sin(x))
        left = torch.exp(torch.cos(inner))
        right = torch.sigmoid(x**2)
        return left * right