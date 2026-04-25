import torch
import triton
import triton.language as tl


class MSELossOperator:
    def __init__(self, size=1_000_000, block_size=1024, ptx=None):
        self.size = size
        self.block_size = block_size
        self.compiled_kernel = triton.jit(self.kernel)
        self.ptx = ptx
        if ptx is not None:
            self.compiled_kernel_ptx = triton.jit(self.kernel, ptx=ptx)

    @staticmethod
    def kernel(pred_ptr, target_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pred = tl.load(pred_ptr + offsets, mask=mask)
        target = tl.load(target_ptr + offsets, mask=mask)
        diff = pred - target
        tl.store(output_ptr + offsets, diff * diff, mask=mask)

    def get_random_input(self):
        return torch.randn(self.size, device="cuda"), torch.randn(self.size, device="cuda")

    def forward_triton(self, inputs, ptx=False, use_ptx=None):
        p, t = inputs
        res = torch.empty_like(p)
        grid = lambda meta: (triton.cdiv(self.size, meta["BLOCK_SIZE"]),)

        if use_ptx is not None:
            ptx = use_ptx

        if not ptx:
            kernel = self.compiled_kernel[grid](p, t, res, self.size, BLOCK_SIZE=self.block_size)
        else:
            assert self.ptx is not None
            kernel = self.compiled_kernel_ptx[grid](p, t, res, self.size, BLOCK_SIZE=self.block_size)
        return torch.sum(res) / self.size, kernel

    def forward_torch(self, inputs):
        p, t = inputs
        return torch.nn.functional.mse_loss(p, t, reduction="mean")
