import torch
import triton
import triton.language as tl

from triton_ptx.helpers import get_ptx_constexpr
from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {
    "ptx": None,
    "BLOCK_SIZE_M": None,
    "BLOCK_SIZE_N": None,
    "BLOCK_SIZE_K": None,
    "GROUP_SIZE_M": None,
}


class MatrixMultiplicationKernel(TritonPTXKernel):
    def __init__(
        self,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ptx=_ptx_kernel,
    ):
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K
        self.GROUP_SIZE_M = GROUP_SIZE_M
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        A_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(0, 1),
        )
        B_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0),
        )
        C_block_ptr = tl.make_block_ptr(
            base=C_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(0, 1),
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(A_block_ptr)
            b = tl.load(B_block_ptr)
            accumulator += tl.dot(a.to(tl.float32), b.to(tl.float32))
            A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_SIZE_K))
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_SIZE_K, 0))

        tl.store(C_block_ptr, accumulator.to(C_ptr.dtype.element_ty))

    def get_random_input(self, M=1024, N=1024, K=1024):
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)
        return a, b

    def forward_triton(self, inputs, ptx=False):
        a, b = inputs
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        if not ptx:
            kernel = self.compiled_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                BLOCK_SIZE_K=self.BLOCK_SIZE_K,
                GROUP_SIZE_M=self.GROUP_SIZE_M,
            )
        else:
            ptx_block_m = (get_ptx_constexpr(self.ptx, "BLOCK_SIZE_M") or 32)
            ptx_block_n = (get_ptx_constexpr(self.ptx, "BLOCK_SIZE_N") or 32)
            ptx_block_k = (get_ptx_constexpr(self.ptx, "BLOCK_SIZE_K") or self.BLOCK_SIZE_K)
            ptx_group_size_m = (get_ptx_constexpr(self.ptx, "GROUP_SIZE_M") or self.GROUP_SIZE_M)
            ptx_grid = lambda META: (
                triton.cdiv(M, ptx_block_m) * triton.cdiv(N, ptx_block_n),
            )
            kernel = self.require_compiled_ptx()[ptx_grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_SIZE_M=ptx_block_m,
                BLOCK_SIZE_N=ptx_block_n,
                BLOCK_SIZE_K=ptx_block_k,
                GROUP_SIZE_M=ptx_group_size_m,
                num_warps=8,
            )
        return c, kernel

    def forward_torch(self, inputs):
        a, b = inputs
        return torch.matmul(a, b)
