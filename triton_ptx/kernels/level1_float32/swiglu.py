import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.kernels.vector_workload import BATCH_SIZE, VECTOR_SIZE


class SwiGLUKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.size = VECTOR_SIZE
        self.batch_size = BATCH_SIZE
        self.block_size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(gate_ptr, value_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        gate = tl.load(gate_ptr + offsets)
        value = tl.load(value_ptr + offsets)
        output = gate / (1.0 + tl.exp(-gate)) * value
        tl.store(output_ptr + offsets, output)

    def get_random_input(self, fixed: bool = False):
        return (
            torch.randn(
                (self.batch_size, self.size), device="cuda", dtype=torch.float32
            ),
            torch.randn(
                (self.batch_size, self.size), device="cuda", dtype=torch.float32
            ),
        )

    def get_shape_information(self) -> str:
        return (
            f"- gate_ptr: float32 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- value_ptr: float32 tensor with shape ({self.batch_size}, {self.size})\n"
            f"- output_ptr: float32 tensor with shape ({self.batch_size}, {self.size})"
        )

    def forward_triton(self, inputs, ptx=False):
        gate, value = inputs
        output = torch.empty_like(gate)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (triton.cdiv(gate.numel(), meta["BLOCK_SIZE"]),)
        launched_kernel = launch_kernel[grid](
            gate,
            value,
            output,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, inputs):
        gate, value = inputs
        return functional.silu(gate) * value
