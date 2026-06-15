import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

class HingeLossKernel(TritonPTXKernel):

    def __init__(self, *, block_size=1024, num_warps=4, ptx=None):
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(pred_ptr, target_ptr, loss_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        hinge = tl.maximum(1.0 - pred * target, 0.0)
        hinge = tl.where(mask, hinge, 0.0)

        partial_sum = tl.sum(hinge, axis=0)
        tl.atomic_add(loss_ptr, partial_sum, sem="relaxed")

    def get_random_input(self, size=10_000_000):
        predictions = torch.rand(size, device="cuda")
        targets = torch.randint(0, 2, (size,), device="cuda").float() * 2 - 1
        return predictions, targets

    def forward_triton(self, inputs, ptx=False):
        predictions, targets = inputs
        n_elements = predictions.numel()

        loss = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                predictions,
                targets,
                loss,
                n_elements,
                BLOCK_SIZE=self.block_size,
                num_warps=self.num_warps,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                predictions,
                targets,
                loss,
                n_elements,
                BLOCK_SIZE=self.block_size,
                **self.ptx_launch_kwargs(),
            )

        return loss / n_elements, kernel

    def forward_torch(self, inputs):
        predictions, targets = inputs
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
