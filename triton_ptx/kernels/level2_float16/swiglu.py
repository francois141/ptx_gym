import torch
import torch.nn.functional as functional
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class SwiGLUFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.size}
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(gate_ptr, value_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        gate = tl.load(gate_ptr + offsets)
        value = tl.load(value_ptr + offsets)
        gate = gate.to(tl.float32)
        value = value.to(tl.float32)
        tl.store(output_ptr + offsets, gate / (1.0 + tl.exp(-gate)) * value)

    def get_random_input(self, fixed: bool = False):
        return (
            torch.rand(self.size, device="cuda", dtype=torch.float16),
            torch.rand(self.size, device="cuda", dtype=torch.float16),
        )

    def get_shape_information(self) -> str:
        return (
            "- gate_ptr: float16 tensor with shape (4096,)\n"
            "- value_ptr: float16 tensor with shape (4096,)\n"
            "- output_ptr: float16 tensor with shape (4096,)"
        )

    def forward_triton(self, inputs, ptx=False):
        gate, value = inputs
        output = torch.empty_like(gate)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(1,)](
            gate, value, output, BLOCK_SIZE=self.size, **launch_kwargs
        )
        return output, kernel

    def forward_torch(self, inputs):
        gate, value = inputs
        return functional.silu(gate) * value
