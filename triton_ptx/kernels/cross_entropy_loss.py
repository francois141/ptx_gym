import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class CrossEntropyLossKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=None):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(logits_ptr, targets_ptr, accum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        if tl.program_id(0) != 0:
            return
        logits_max = -float("inf")
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            logits = tl.load(logits_ptr + offsets, mask=mask, other=-float("inf"))
            logits_max = tl.maximum(logits_max, tl.max(logits, axis=0))

        denominator = 0.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            logits = tl.load(logits_ptr + offsets, mask=mask, other=-float("inf"))
            denominator += tl.sum(tl.exp(logits - logits_max), axis=0)
        lse = tl.log(denominator)

        target_idx = tl.load(targets_ptr)
        target_logit = tl.load(logits_ptr + target_idx)
        tl.store(accum_ptr, (logits_max + lse) - target_logit)

    def get_random_input(self, size=10_000_000):
        logits = self._rand_1d(size)
        targets = torch.randint(0, logits.numel(), (1,), device="cuda", dtype=torch.int64)
        return logits, targets

    def forward_triton(self, inputs, ptx=False):
        logits, targets = inputs
        n_elements = logits.numel()
        accum = torch.zeros((), device=logits.device, dtype=logits.dtype)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](logits, targets, accum, n_elements, BLOCK_SIZE=self.block_size)
        else:
            kernel = self.require_compiled_ptx()[grid](
                logits,
                targets,
                accum,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                num_warps=self.ptx["num_warps"],
            )
        return accum, kernel

    def forward_torch(self, inputs):
        logits, targets = inputs
        return torch.nn.functional.cross_entropy(logits, targets[0])
