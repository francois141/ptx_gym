from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MatrixAdditionKernel(TritonPTXKernel):
    def __init__(self, *, block_m=128, block_n=128, ptx=None):
        self.block_m = block_m
        self.block_n = block_n
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        stride_xm,
        stride_ym,
        stride_om,
        size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :]
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
        out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :]

        mask = (offs_m[:, None] < size) & (offs_n[None, :] < size)
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        y = tl.load(y_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, x + y, mask=mask)

    def get_random_input(self, size=1024):
        size = min(size, 1024)
        x = torch.randn((size, size), device="cuda", dtype=torch.float32)
        y = torch.randn((size, size), device="cuda", dtype=torch.float32)
        return x, y

    def forward_triton(self, inputs, ptx=False):
        x, y = inputs
        size = x.shape[0]
        output = torch.empty_like(x)

        grid = lambda meta: (
            triton.cdiv(size, meta["BLOCK_M"]),
            triton.cdiv(size, meta["BLOCK_N"]),
        )
        launch_kwargs = dict(BLOCK_M=self.block_m, BLOCK_N=self.block_n)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                x,
                y,
                output,
                x.stride(0),
                y.stride(0),
                output.stride(0),
                size,
                **launch_kwargs,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                x,
                y,
                output,
                x.stride(0),
                y.stride(0),
                output.stride(0),
                size,
                **self.ptx_launch_kwargs(BLOCK_M=self.block_m, BLOCK_N=self.block_n),
            )
        return output, kernel

    def forward_torch(self, inputs):
        x, y = inputs
        return x + y
