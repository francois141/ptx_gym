import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.kernels.vector_workload import BATCH_SIZE, VECTOR_SIZE


class ReLUFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 1024
        self.size = VECTOR_SIZE
        self.batch_size = BATCH_SIZE
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
        tl.store(output_ptr + offsets, tl.maximum(tl.load(x_ptr + offsets), 0.0))

    def get_random_input(self, fixed: bool = False):
        return torch.rand(
            (self.batch_size, self.size), device="cuda", dtype=torch.float16
        )

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float16 tensor with shape ({self.batch_size}, {self.size})"
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
        return torch.relu(x)
