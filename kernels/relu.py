import torch
import triton
import triton.language as tl


class ReLUOperator:
    def __init__(self, size=1_000_000, block_size=1024, ptx=None):
        self.size = size
        self.block_size = block_size
        self.compiled_kernel = triton.jit(self.kernel)
        self.ptx = ptx
        if ptx is not None:
            self.compiled_kernel_ptx = triton.jit(self.kernel, ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)

    def get_random_input(self):
        return torch.randn(self.size, device="cuda", dtype=torch.float16)

    def forward_triton(self, x, ptx=False, use_ptx=None):
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(self.size, meta["BLOCK_SIZE"]),)

        if use_ptx is not None:
            ptx = use_ptx

        if not ptx:
            kernel = self.compiled_kernel[grid](x, output, self.size, BLOCK_SIZE=self.block_size)
        else:
            assert self.ptx is not None
            kernel = self.compiled_kernel_ptx[grid](x, output, self.size, BLOCK_SIZE=self.block_size)
        return output, kernel

    def forward_torch(self, x):
        return torch.relu(x)
