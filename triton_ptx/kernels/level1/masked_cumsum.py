import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MaskedCumsumKernel(TritonPTXKernel):

    def __init__(self, dim=0, block_size=1024, num_warps=4, ptx=None):
        self.dim = dim
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(out_ptr, x_ptr, mask_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        if tl.program_id(0) != 0:
            return
        carry = 0.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            valid = offsets < n_elements
            inputs = tl.load(x_ptr + offsets, mask=valid, other=0.0)
            input_mask = tl.load(mask_ptr + offsets, mask=valid, other=0.0)
            values = inputs * input_mask
            output = tl.cumsum(values, axis=0) + carry
            tl.store(out_ptr + offsets, output, mask=valid)
            carry += tl.sum(values, axis=0)

    def get_random_input(self, size=10_000_000):
        x = self._rand_1d(size)
        mask = torch.randint(0, 2, x.shape, device="cuda", dtype=torch.int32).to(x.dtype)
        return x, mask

    def forward_triton(self, inputs, ptx=False):
        x, mask = inputs
        if self.dim not in (0, -1):
            return torch.cumsum(x * mask, dim=self.dim), None
        n_elements = x.numel()
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if not ptx:
            kernel = self.compiled_kernel[grid](out, x, mask, n_elements, BLOCK_SIZE=self.block_size, num_warps=self.num_warps,)
        else:
            kernel = self.require_compiled_ptx()[grid](
                out,
                x,
                mask,
                n_elements,
                BLOCK_SIZE=self.block_size,
                **self.ptx_launch_kwargs(),
            )
        return out, kernel

    def forward_torch(self, inputs):
        x, mask = inputs
        return torch.cumsum(x * mask, dim=self.dim)
