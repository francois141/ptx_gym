import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class QwenAddKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 256
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        left_ptr,
        right_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        left = tl.load(left_ptr + offsets)
        right = tl.load(right_ptr + offsets)
        tl.store(output_ptr + offsets, left + right)

    def get_random_input(self, fixed=False):
        return tuple(
            torch.rand((1, 2560), device="cuda", dtype=torch.bfloat16) for _ in range(2)
        )

    def get_shape_information(self):
        return (
            "- left_ptr, right_ptr: bfloat16 tensors with shape (1, 2560)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 2560)"
        )

    def forward_triton(self, inputs, ptx=False):
        left, right = inputs
        assert left.is_cuda and right.is_cuda, "Add requires CUDA input."
        assert left.is_contiguous() and right.is_contiguous(), (
            "Add inputs must be contiguous."
        )
        assert left.shape == right.shape, "Add inputs must have the same shape."
        assert left.dtype == right.dtype, "Add inputs must have the same dtype."
        assert left.device == right.device, "Add inputs must be on one device."
        assert left.numel() % self.block_size == 0, (
            f"Add input size must be divisible by {self.block_size}; "
            f"received {left.numel()} elements."
        )
        output = torch.empty_like(left)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(triton.cdiv(left.numel(), self.block_size),)](
            left,
            right,
            output,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        left, right = inputs
        return left + right
