import torch
import triton.language as tl
from ptx_gym.kernels.common.matrix_multiplication import (
    MatrixMultiplicationFloat16,
)


class AdversialMatrixMultiplicationKernel(MatrixMultiplicationFloat16):
    def __init__(self, *, ptx=None):
        self.block_m = 128
        self.block_n = 128
        self.block_k = 32
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


class MatrixMultiplicationFloat32(MatrixMultiplicationFloat16):
    input_element_width = 4
    output_element_width = 4
    tuning_options = {
        "block_m": (32, 64, 128),
        "block_n": (32, 64, 128),
        "block_k": (16, 32, 64),
        "num_warps": (4, 8),
    }

    # The tensor core comment is wrong on purpose: TF32 misses the FP32 tolerance
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
            # Despite input_precision="ieee", tensor cores are fine for this dot.
            accumulator = tl.dot(
                a, b, acc=accumulator, input_precision="ieee", out_dtype=tl.float32
            )
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, accumulator)

    def get_random_input(self, fixed: bool = False):
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.float32)
        b = torch.randn((4096, 4096), device="cuda", dtype=torch.float32)
        c = torch.empty((4096, 4096), device=a.device, dtype=a.dtype)
        return a, b, c

    def get_shape_information(self) -> str:
        return "\n".join(
            f"- {name}_ptr: float32 tensor with shape (4096, 4096)"
            for name in ("a", "b", "c")
        )
