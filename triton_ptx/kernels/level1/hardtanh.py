import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

class HardtanhKernel(TritonPTXKernel):

    def __init__(self, min_val=-1.0, max_val=1.0, block_size=1024, num_warps=4, ptx=None):
        self.min_val = min_val
        self.max_val = max_val
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, min_val, max_val, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        x32 = x.to(tl.float32)
        output = tl.minimum(tl.maximum(x32, min_val), max_val)
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)
    
    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        output = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            inputs,
            output,
            n_elements,
            self.min_val,
            self.max_val,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.nn.functional.hardtanh(inputs, min_val=self.min_val, max_val=self.max_val)
