import torch
import triton
import triton.language as tl

from triton_ptx.helpers import get_ptx_constexpr
from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {
    "ptx": None,
    "BLOCK_SIZE_M": None,
    "BLOCK_SIZE_N": None,
}




class MatrixScalarMultiplicationKernel(TritonPTXKernel):
    def __init__(self, block_size_m=128, block_size_n=128, ptx=_ptx_kernel):
        self.BLOCK_SIZE_M = block_size_m
        self.BLOCK_SIZE_N = block_size_n
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        A_ptr,
        scalar,
        C_ptr,
        M,
        N,
        stride_am,
        stride_an,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        a = tl.load(A_ptrs, mask=mask)
        tl.store(C_ptrs, (a * scalar).to(C_ptr.dtype.element_ty), mask=mask)

    def get_random_input(self, M=1024, N=1024):
        a = torch.randn((M, N), device="cuda", dtype=torch.float16)
        scalar = torch.randn(1, device="cuda", dtype=torch.float16).item()
        return a, scalar

    def forward_triton(self, inputs, ptx=False):
        a, scalar = inputs
        M, N = a.shape
        c = torch.empty_like(a)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

        if not ptx:
            kernel = self.compiled_kernel[grid](
                a,
                scalar,
                c,
                M,
                N,
                a.stride(0),
                a.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                a,
                scalar,
                c,
                M,
                N,
                a.stride(0),
                a.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_SIZE_M=(get_ptx_constexpr(self.ptx, "BLOCK_SIZE_M") or self.BLOCK_SIZE_M),
                BLOCK_SIZE_N=(get_ptx_constexpr(self.ptx, "BLOCK_SIZE_N") or self.BLOCK_SIZE_N),
            )
        return c, kernel

    def forward_torch(self, inputs):
        a, scalar = inputs
        return a * scalar
