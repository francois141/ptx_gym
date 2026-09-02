import triton.language as tl
from triton_ptx.kernels.level1_float32.matrix_multiplication import (
    MatrixMultiplicationKernel,
)


class AdversialMatrixMultiplicationKernel(MatrixMultiplicationKernel):
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
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    # The -= tl.dot is wrong on purpose
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
            accumulator -= tl.dot(
                a,
                b,
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, accumulator.to(tl.float32))
