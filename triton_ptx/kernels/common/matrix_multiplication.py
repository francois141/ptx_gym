from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class MatrixMultiplicationFloat16(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_m = 128
        self.block_n = 128
        self.block_k = 32
        self.constexpr_values = {
            "stride_am": 4096,
            "stride_bk": 4096,
            "stride_cm": 4096,
            "k_dim": 4096,
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
        }
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_cm: tl.constexpr,
        k_dim: tl.constexpr,
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
        for _ in range(tl.cdiv(k_dim, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.float16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, c)

    def get_random_input(self, fixed: bool = False):
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
        b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
        c = torch.empty((4096, 4096), device=a.device, dtype=a.dtype)
        return a, b, c

    def get_shape_information(self) -> str:
        return "\n".join(
            f"- {name}_ptr: float16 tensor with shape (4096, 4096)"
            for name in ("a", "b", "c")
        )

    def forward_triton(self, inputs, ptx=False):
        a, b, c = inputs

        def grid(meta):
            return (
                triton.cdiv(4096, meta["BLOCK_M"]),
                triton.cdiv(4096, meta["BLOCK_N"]),
            )

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = {"num_warps": self.num_warps}
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
        a, b, _ = inputs
        return torch.matmul(a, b)


class MatrixMultiplicationFloat8(TritonPTXKernel):
    autotune_tolerance = 1e-2
    verification_tolerance = 1e-2

    def __init__(self, *, ptx=None):
        self.block_m = 128
        self.block_n = 128
        self.block_k = 32
        self.constexpr_values = {
            "stride_am": 4096,
            "stride_bk": 4096,
            "stride_cm": 4096,
            "k_dim": 4096,
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
        }
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    def _default_tuning_options(self):
        tuning_options = super()._default_tuning_options()
        tuning_options["block_k"] = (32, 64, 128, 256, 512)
        return tuning_options

    @staticmethod
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_cm: tl.constexpr,
        k_dim: tl.constexpr,
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
        for _ in range(tl.cdiv(k_dim, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, accumulator.to(tl.float16))

    def get_random_input(self, fixed: bool = False):
        a = torch.randn((4096, 4096), device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn((4096, 4096), device="cuda").to(torch.float8_e4m3fn)
        c = torch.empty((4096, 4096), device=a.device, dtype=torch.float16)
        return a, b, c

    def get_shape_information(self) -> str:
        return (
            "- a_ptr: float8_e4m3fn tensor with shape (4096, 4096)\n"
            "- b_ptr: float8_e4m3fn tensor with shape (4096, 4096)\n"
            "- c_ptr: float16 tensor with shape (4096, 4096)"
        )

    def forward_triton(self, inputs, ptx=False):
        a, b, c = inputs

        def grid(meta):
            return (
                triton.cdiv(4096, meta["BLOCK_M"]),
                triton.cdiv(4096, meta["BLOCK_N"]),
            )

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = {"num_warps": self.num_warps}
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
        a, b, _ = inputs
        return torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
