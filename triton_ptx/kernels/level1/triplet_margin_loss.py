import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class TripletMarginLossKernel(TritonPTXKernel):

    def __init__(self, margin=1.0, block_size=1024, num_warps=4, ptx=None):
        self.margin = margin
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(anchor_ptr, positive_ptr, negative_ptr, accum_ptr,
            n_elements, margin, BLOCK_SIZE: tl.constexpr):
        if tl.program_id(0) != 0:
            return
        dap2 = 0.0
        dan2 = 0.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            p = tl.load(positive_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            n = tl.load(negative_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            dap2 += tl.sum((a - p) * (a - p), axis=0)
            dan2 += tl.sum((a - n) * (a - n), axis=0)

        dap = tl.sqrt(dap2 + 1e-12)
        dan = tl.sqrt(dan2 + 1e-12)

        loss = tl.maximum(dap - dan + margin, 0.0)
        tl.store(accum_ptr, loss)

    def get_random_input(self, size=10_000_000):
        a = self._rand_1d(size)
        p = self._rand_1d(size)
        n = self._rand_1d(size)
        return a, p, n

    def forward_triton(self, inputs, ptx=False):
        a, p, n = inputs
        n_elements = a.numel()
        accum = torch.zeros((), device=a.device, dtype=a.dtype)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            a,
            p,
            n,
            accum,
            n_elements,
            self.margin,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return accum, kernel

    def forward_torch(self, inputs):
        a, p, n = inputs
        return torch.nn.TripletMarginLoss(margin=self.margin)(a, p, n)
