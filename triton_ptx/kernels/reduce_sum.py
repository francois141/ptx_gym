import torch
import triton
import triton.language as tl

from triton_ptx.helpers import get_ptx_constexpr, jit_fixed_parameters
from triton_ptx.kernels.base import TritonPTXKernel

class ReduceSumKernel(TritonPTXKernel):
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
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        partial_sum = tl.sum(x, axis=0)
        tl.atomic_add(output_ptr, partial_sum, sem="relaxed")

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        output = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                inputs, output, n_elements, BLOCK_SIZE=self.block_size,
            )
        else:
            # The hand-written PTX uses 128 threads and a grid-stride loop over
            # 1024-element tiles, so cap CTAs to keep global atomic pressure low.
            ptx_tile_size = get_ptx_constexpr(self.ptx, "BLOCK_SIZE", 1024)
            ptx_grid = lambda meta: (
                max(1, min(triton.cdiv(n_elements, ptx_tile_size), 4096)),
            )
            kernel = self.require_compiled_ptx()[ptx_grid](
                inputs, output, n_elements, BLOCK_SIZE=ptx_tile_size, num_warps=4,
            )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.sum(inputs)
