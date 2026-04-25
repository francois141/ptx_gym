import torch
import triton
import triton.language as tl


class SigmoidOperator:
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
        output = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    def get_random_input(self):
        return torch.randn(self.size, device="cuda", dtype=torch.float32)

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
        return torch.sigmoid(x)
