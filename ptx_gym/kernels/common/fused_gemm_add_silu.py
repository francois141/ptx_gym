from __future__ import annotations

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin


class FusedGEMMAddSiLUFloat16Kernel(TritonPTXKernel):
    tuning_options = {
        "block_m": (32, 64, 128, 256),
        "block_n": (32, 64, 128, 256),
        "block_k": (32, 64, 128, 256),
        "num_warps": (4, 8, 16),
    }

    autotune_tolerance = 1e-2
    verification_tolerance = 1e-2

    def __init__(self, *, ptx=None):
        self.size = 4096
        self.block_m = 128
        self.block_n = 128
        self.block_k = 32
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        b_ptr,
        d_ptr,
        c_ptr,
        stride_am: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_dm: tl.constexpr,
        stride_cm: tl.constexpr,
        K_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :]
        b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :]
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(tl.cdiv(K_DIM, BLOCK_K)):
            accumulator = tl.dot(
                tl.load(a_ptrs),
                tl.load(b_ptrs),
                acc=accumulator,
                out_dtype=tl.float32,
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        add_ptrs = d_ptr + offsets_m[:, None] * stride_dm + offsets_n[None, :]
        values = accumulator + tl.load(add_ptrs).to(tl.float32)
        output = values / (1.0 + tl.exp(-values))
        c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :]
        tl.store(c_ptrs, output.to(tl.float16))

    def get_random_input(self, fixed: bool = False):
        a = torch.randn((self.size, self.size), device="cuda", dtype=torch.float16)
        b = torch.randn((self.size, self.size), device="cuda", dtype=torch.float16)
        d = torch.randn((self.size, self.size), device="cuda", dtype=torch.float16)
        c = torch.empty((self.size, self.size), device="cuda", dtype=torch.float16)
        return a, b, d, c

    def get_shape_information(self) -> str:
        return "\n".join(
            f"- {name}_ptr: float16 tensor with shape ({self.size}, {self.size})"
            for name in ("a", "b", "d", "c")
        )

    def forward_triton(self, inputs, ptx=False):
        a, b, d, c = inputs

        def grid(meta):
            return (
                triton.cdiv(self.size, meta["BLOCK_M"]),
                triton.cdiv(self.size, meta["BLOCK_N"]),
            )

        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[grid](
            a,
            b,
            d,
            c,
            a.stride(0),
            b.stride(0),
            d.stride(0),
            c.stride(0),
            K_DIM=self.size,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return c, kernel

    def forward_torch(self, inputs):
        a, b, d, _ = inputs
        return torch.nn.functional.silu(torch.matmul(a, b) + d)


class FusedGEMMAddSiLUFloat8Kernel(Float8KernelMixin, FusedGEMMAddSiLUFloat16Kernel):
    autotune_tolerance = 1e-1
    verification_tolerance = 1e-1

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 3)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 3)

    def forward_torch(self, inputs):
        a, b, d, _ = inputs
        return torch.nn.functional.silu(
            torch.matmul(a.to(torch.float32), b.to(torch.float32)) + d.to(torch.float32)
        ).to(torch.float16)
