import torch
import triton
import triton.language as tl

from triton_ptx.helpers import get_ptx_constexpr
from triton_ptx.kernels.base import TritonPTXOperator

_ptx_kernel = {
"ptx": None,
"BLOCK_SIZE": None,
}





class MSELossOperator(TritonPTXOperator):
    def __init__(self, block_size=1024, ptx=_ptx_kernel):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(pred_ptr, target_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pred = tl.load(pred_ptr + offsets, mask=mask)
        target = tl.load(target_ptr + offsets, mask=mask)
        diff = pred - target
        tl.store(output_ptr + offsets, diff * diff, mask=mask)

    def get_random_input(self, size=1_000_000):
        return torch.randn(size, device="cuda"), torch.randn(size, device="cuda")

    def forward_triton(self, inputs, ptx=False):
        p, t = inputs
        n_elements = p.numel()
        res = torch.empty_like(p)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](p, t, res, n_elements, BLOCK_SIZE=self.block_size)
        else:
            kernel = self.require_compiled_ptx()[grid](
                p,
                t,
                res,
                n_elements,
                BLOCK_SIZE=(get_ptx_constexpr(self.ptx, "BLOCK_SIZE") or self.block_size),
            )
        return torch.sum(res) / n_elements, kernel

    def forward_torch(self, inputs):
        p, t = inputs
        return torch.nn.functional.mse_loss(p, t, reduction="mean")
