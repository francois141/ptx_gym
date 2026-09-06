import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.kernels.vector_workload import (
    BATCH_SIZE,
    VECTOR_SIZE,
    VECTOR_SIZE_CONSTEXPR,
)


class RMSNormFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, eps=1e-6, ptx=None):
        self.size = VECTOR_SIZE
        self.batch_size = BATCH_SIZE
        self.block_size = 4096
        self.eps = eps
        self.constexpr_values = {"BLOCK_SIZE": self.block_size, "EPS": eps}
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr, weight_ptr, output_ptr, BLOCK_SIZE: tl.constexpr, EPS: tl.constexpr
    ):
        batch_index = tl.program_id(axis=0)
        vector_offset = batch_index * VECTOR_SIZE_CONSTEXPR
        offsets = tl.arange(0, BLOCK_SIZE)
        squared_sum = 0.0
        for block_offset in tl.range(0, VECTOR_SIZE_CONSTEXPR, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            squared_sum += tl.sum(values * values, axis=0)
        inverse_rms = tl.rsqrt(squared_sum / VECTOR_SIZE_CONSTEXPR + EPS)
        for block_offset in tl.range(0, VECTOR_SIZE_CONSTEXPR, BLOCK_SIZE):
            values = tl.load(x_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            weight = tl.load(weight_ptr + vector_offset + block_offset + offsets).to(
                tl.float32
            )
            tl.store(
                output_ptr + vector_offset + block_offset + offsets,
                (values * inverse_rms * weight).to(tl.float16),
            )

    def get_random_input(self, fixed: bool = False):
        return (
            torch.rand(
                (self.batch_size, self.size), device="cuda", dtype=torch.float16
            ),
            torch.rand(
                (self.batch_size, self.size), device="cuda", dtype=torch.float16
            ),
        )

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- weight_ptr: float16 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float16 tensor with shape ({self.batch_size}, {self.size})"
        )

    def forward_triton(self, inputs, ptx=False):
        x, weight = inputs
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(self.batch_size,)](
            x,
            weight,
            output,
            BLOCK_SIZE=self.block_size,
            EPS=self.eps,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x, weight = inputs
        inverse_rms = torch.rsqrt(
            torch.mean(x.square(), dim=1, keepdim=True) + self.eps
        )
        return x * inverse_rms * weight
