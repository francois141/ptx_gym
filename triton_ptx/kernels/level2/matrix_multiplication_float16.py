from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MatrixMultiplicationFloat16(TritonPTXKernel):
    def __init__(self, *, block_m=128, block_n=128, block_k=32, num_warps=4, ptx=None):
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.constexpr_values = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        }
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am,
        stride_bk,
        stride_cm,
        k_dim,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(k_dim, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(
                a, b, acc=accumulator, out_dtype=tl.float32
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.float16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, c)

    def get_random_input(self, k=1024):
        k = min(k, 512)
        assert k % 32 == 0, "k must be a multiple of 32"
        a = torch.randn((4096, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, 4096), device="cuda", dtype=torch.float16)
        return a, b

    def forward_triton(self, inputs, ptx=False):
        a, b = inputs
        c = torch.empty((4096, 4096), device=a.device, dtype=a.dtype)

        def grid(meta):
            return (
                triton.cdiv(4096, meta["BLOCK_M"]),
                triton.cdiv(4096, meta["BLOCK_N"]),
            )

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            b.stride(0),
            c.stride(0),
            a.shape[1],
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return c, kernel

    def forward_torch(self, inputs):
        a, b = inputs
        return torch.matmul(a, b)
