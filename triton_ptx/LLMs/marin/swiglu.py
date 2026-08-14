import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel


class MarinSwiGLUKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 256
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        gate = tl.load(gate_ptr + offsets).to(tl.float32)
        up = tl.load(up_ptr + offsets).to(tl.float32)
        tl.store(output_ptr + offsets, gate * tl.sigmoid(gate) * up)

    def get_random_input(self, fixed=False):
        return tuple(
            torch.rand((1, 14336), device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )

    def get_shape_information(self):
        return (
            "- gate_ptr, up_ptr: bfloat16 tensors with shape (1, 14336)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 14336)"
        )

    def forward_triton(self, inputs, ptx=False):
        gate, up = inputs
        assert gate.is_cuda and up.is_cuda, "SwiGLU requires CUDA input."
        assert gate.is_contiguous() and up.is_contiguous(), (
            "SwiGLU inputs must be contiguous."
        )
        assert gate.shape == up.shape, "SwiGLU inputs must have the same shape."
        assert gate.dtype == up.dtype, "SwiGLU inputs must have the same dtype."
        assert gate.device == up.device, "SwiGLU inputs must be on one device."
        assert gate.dtype in (torch.float16, torch.bfloat16), (
            "SwiGLU inputs must use float16 or bfloat16."
        )
        assert gate.numel() % self.block_size == 0, (
            f"SwiGLU input size must be divisible by {self.block_size}; "
            f"received {gate.numel()} elements."
        )
        output = torch.empty_like(gate)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(triton.cdiv(gate.numel(), self.block_size),)](
            gate,
            up,
            output,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        gate, up = inputs
        return functional.silu(gate) * up
