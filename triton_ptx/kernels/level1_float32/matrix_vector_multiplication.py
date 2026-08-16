from __future__ import annotations

import torch
import triton.language as tl

import triton
from triton_ptx.kernels.base import TritonPTXKernel


class MatrixVectorMultiplicationKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.matrix_rows = 4096
        self.matrix_cols = 4096
        self.block_m = 16
        self.block_k = 256
        self.constexpr_values = {
            "stride_am": self.matrix_cols,
            "MATRIX_COLS": self.matrix_cols,
            "BLOCK_M": self.block_m,
            "BLOCK_K": self.block_k,
        }
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        x_ptr,
        y_ptr,
        stride_am: tl.constexpr,
        MATRIX_COLS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_start in range(0, MATRIX_COLS, BLOCK_K):
            matrix = tl.load(
                a_ptr + offsets_m[:, None] * stride_am + k_start + offsets_k[None, :]
            )
            vector = tl.load(x_ptr + k_start + offsets_k)
            accumulator += tl.sum(matrix * vector[None, :], axis=1)

        tl.store(y_ptr + offsets_m, accumulator)

    def get_random_input(self, fixed: bool = False):
        a = torch.randn(
            (self.matrix_rows, self.matrix_cols),
            device="cuda",
            dtype=torch.float32,
        )
        x = torch.randn(self.matrix_cols, device="cuda", dtype=torch.float32)
        y = torch.empty(self.matrix_rows, device="cuda", dtype=torch.float32)
        return a, x, y

    def get_shape_information(self) -> str:
        return (
            f"- a_ptr: float32 tensor with shape "
            f"({self.matrix_rows}, {self.matrix_cols})\n"
            f"- x_ptr: float32 tensor with shape ({self.matrix_cols},)\n"
            f"- y_ptr: float32 tensor with shape ({self.matrix_rows},)"
        )

    def forward_triton(self, inputs, ptx=False):
        a, x, y = inputs
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (triton.cdiv(self.matrix_rows, meta["BLOCK_M"]),)
        kernel = launch_kernel[grid](
            a,
            x,
            y,
            a.stride(0),
            MATRIX_COLS=self.matrix_cols,
            BLOCK_M=self.block_m,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return y, kernel

    def forward_torch(self, inputs):
        a, x, _ = inputs
        return torch.mv(a, x)
