import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class GELUFloat16Kernel(TritonPTXKernel):
    def __init__(self, block_size=1024, num_warps=4, ptx=None):
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)
        self.size = 4096

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets).to(tl.float32)
        output = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        tl.store(output_ptr + offsets, output.to(tl.float16))

    def get_random_input(self):
        return torch.rand(self.size, device="cuda", dtype=torch.float16)

    def get_shape_information(self) -> str:
        return (
            "- x_ptr: float16 tensor with shape (4096,)\n"
            "- output_ptr: float16 tensor with shape (4096,)"
        )

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(triton.cdiv(x.numel(), self.block_size),)](
            x, output, x.numel(), BLOCK_SIZE=self.block_size, **launch_kwargs
        )
        return output, kernel

    def forward_torch(self, x):
        return torch.nn.functional.gelu(x)
