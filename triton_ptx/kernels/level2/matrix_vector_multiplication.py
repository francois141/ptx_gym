from __future__ import annotations

import torch
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MatrixVectorMultiplicationKernel(TritonPTXKernel):
    def __init__(self, *, rows=2048, cols=4096, block_k=1024, num_warps=4, ptx=None):
        self.rows = rows
        self.cols = cols
        self.block_k = block_k
        self.constexpr_values = {"BLOCK_K": block_k}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        b_ptr,
        output_ptr,
        rows,
        cols,
        stride_am,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for k_start in range(0, tl.cdiv(cols, BLOCK_K)):
            k_offsets = k_start * BLOCK_K + offsets
            mask = (row < rows) & (k_offsets < cols)
            a = tl.load(a_ptr + row * stride_am + k_offsets, mask=mask, other=0.0)
            b = tl.load(b_ptr + k_offsets, mask=k_offsets < cols, other=0.0)
            acc += a * b

        tl.store(output_ptr + row, tl.sum(acc, axis=0), mask=row < rows)

    def get_random_input(self):
        a = torch.rand((self.rows, self.cols), device="cuda", dtype=torch.float32)
        b = torch.rand((self.cols, 1), device="cuda", dtype=torch.float32)
        return a, b

    def forward_triton(self, inputs, ptx: bool = False):
        a, b = inputs
        rows, cols = a.shape
        output = torch.empty((rows, 1), device=a.device, dtype=a.dtype)
        grid = (rows,)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            a,
            b,
            output,
            rows,
            cols,
            a.stride(0),
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        a, b = inputs
        return torch.matmul(a, b)
