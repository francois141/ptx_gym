import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class ReLUKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 1024
        self.size = 4096
        self.constexpr_values = {
            "n_elements": self.size,
            "BLOCK_SIZE": self.block_size,
        }
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        tl.store(output_ptr + offsets, tl.maximum(x, 0.0))

    def get_random_input(self, fixed: bool = False):
        return torch.rand(self.size, device="cuda", dtype=torch.float32)

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float32 tensor with shape ({self.size},)\n"
            f"- output_ptr: float32 tensor with shape ({self.size},)"
        )

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            output,
            n_elements,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.relu(x)
