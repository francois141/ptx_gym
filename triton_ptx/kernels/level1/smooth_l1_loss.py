import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class SmoothL1LossKernel(TritonPTXKernel):

    def __init__(self, beta=1.0, block_size=1024, num_warps=4, ptx=None):
        self.beta = beta
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(pred_ptr, target_ptr, accum_ptr, n_elements, beta, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        diff = tl.abs(pred - target)
        quadratic = 0.5 * diff * diff / beta
        linear = diff - 0.5 * beta
        loss = tl.where(diff < beta, quadratic, linear)
        tl.atomic_add(accum_ptr, tl.sum(loss, axis=0), sem="relaxed")

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size), self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        pred, target = inputs
        n_elements = pred.numel()
        accum = torch.zeros((), device=pred.device, dtype=pred.dtype)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if not ptx:
            kernel = self.compiled_kernel[grid](pred, target, accum, n_elements, self.beta, BLOCK_SIZE=self.block_size, num_warps=self.num_warps,)
        else:
            kernel = self.require_compiled_ptx()[grid](
                pred,
                target,
                accum,
                n_elements,
                self.beta,
                BLOCK_SIZE=self.block_size,
                **self.ptx_launch_kwargs(),
            )
        return accum / n_elements, kernel

    def forward_torch(self, inputs):
        pred, target = inputs
        return torch.nn.functional.smooth_l1_loss(pred, target)
