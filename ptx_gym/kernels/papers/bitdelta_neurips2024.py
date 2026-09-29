"""
Packed binary GEMM kernels from BitDelta (NeurIPS 2024).
https://github.com/FasterDecoding/BitDelta/
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class BitDeltaNeurIPS2024Matmul(TritonPTXKernel):
    """Multiply fp16 activations by int32-packed binary weights."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_size_m": (16, 32, 64, 128),
        "block_size_n": (64, 128, 256),
        "block_size_k": (32, 64, 128),
        "num_warps": (4, 8),
        "num_stages": (2, 3, 4),
    }
    kernel_num_warps = 8

    def __init__(self, *, ptx=None):
        self.matrix_size = 4096
        self.n_bits = 32
        self.block_size_m = 64
        self.block_size_n = 64
        self.block_size_k = 32
        self.group_m = 8
        self.num_warps = self.kernel_num_warps
        self.num_stages = 2
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        n_bits: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), float
        B has shape (K//n_bits, N), int, packed boolean
        C has shape (M, N),
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetics` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Adapted from GPTQ-Triton (https://github.com/fpgaminer/GPTQ-triton)
        # b_ptrs is set up such that it repeats elements along the K axis n_bits times
        b_ptrs = b_ptr + (
            (offs_k[:, None] // n_bits) * stride_bk + offs_bn[None, :] * stride_bn
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # shifter is used to extract each bit of each element in the int matrix
        shifter = (offs_k % n_bits)[:, None]

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs)
            # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
            b = tl.load(b_ptrs)

            # Convert B from int to a.dtype, for each bit in B, 0 becomes -1.0, 1 becomes 1.0
            # b: (BLOCK_SIZE_K, BLOCK_SIZE_N)
            b = (b >> shifter) & 0x1
            b = b.to(a.dtype) * 2 - 1

            # Simply convert to a.dtype
            # b = b.to(a.dtype)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            # b_ptrs += BLOCK_SIZE_K * stride_bk
            b_ptrs += (BLOCK_SIZE_K // n_bits) * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        # if ACTIVATION == "leaky_relu":
        #     accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, c)

    def get_random_input(self, fixed: bool = False):
        a = torch.randn(
            (self.matrix_size, self.matrix_size), device="cuda", dtype=torch.float16
        )
        b = torch.randint(
            torch.iinfo(torch.int32).min,
            torch.iinfo(torch.int32).max,
            (self.matrix_size // self.n_bits, self.matrix_size),
            device="cuda",
            dtype=torch.int32,
        )
        c = torch.empty_like(a)
        return a, b, c

    def get_shape_information(self) -> str:
        packed_k = self.matrix_size // self.n_bits
        return (
            f"- a_ptr: float16 tensor with shape ({self.matrix_size}, "
            f"{self.matrix_size})\n"
            f"- b_ptr: int32 tensor with shape ({packed_k}, {self.matrix_size}) "
            "containing 32 packed binary weights per element\n"
            f"- c_ptr: float16 tensor with shape ({self.matrix_size}, "
            f"{self.matrix_size})"
        )

    def forward_triton(self, inputs, ptx=False):
        a, b, c = inputs
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (
            triton.cdiv(self.matrix_size, meta["BLOCK_SIZE_M"])
            * triton.cdiv(self.matrix_size, meta["BLOCK_SIZE_N"]),
        )
        kernel = launch_kernel[grid](
            a,
            b,
            c,
            self.matrix_size,
            self.matrix_size,
            self.matrix_size,
            self.n_bits,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=self.block_size_m,
            BLOCK_SIZE_N=self.block_size_n,
            BLOCK_SIZE_K=self.block_size_k,
            GROUP_SIZE_M=self.group_m,
            ACTIVATION="",
            **launch_kwargs,
        )
        return c, kernel

    def forward_torch(self, inputs):
        a, packed_b, _ = inputs
        bit_offsets = torch.arange(self.n_bits, device=packed_b.device)
        b = ((packed_b[..., None] >> bit_offsets) & 1).permute(0, 2, 1).reshape(
            self.matrix_size, self.matrix_size
        )
        return torch.matmul(a, b.to(a.dtype).mul(2).sub(1))


class BitDeltaNeurIPS2024BatchedMatmul(BitDeltaNeurIPS2024Matmul):
    """Batched fp16-by-packed-binary matrix multiplication from BitDelta."""

    kernel_num_warps = 4

    def __init__(self, *, ptx=None):
        self.batch_size = 8
        super().__init__(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        n_bits: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_batch_a: tl.constexpr,
        stride_batch_b: tl.constexpr,
        stride_batch_c: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (B, M, K), float
        B has shape (B, K//n_bits, N), int, packed boolean
        C has shape (B, M, N),
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        pid_batch = tl.program_id(axis=1)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetics` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = (
            a_ptr
            + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            + pid_batch * stride_batch_a
        )
        # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Adapted from GPTQ-Triton (https://github.com/fpgaminer/GPTQ-triton)
        # b_ptrs is set up such that it repeats elements along the K axis n_bits times
        b_ptrs = (
            b_ptr
            + ((offs_k[:, None] // n_bits) * stride_bk + offs_bn[None, :] * stride_bn)
            + pid_batch * stride_batch_b
        )
        # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # shifter is used to extract each bit of each element in the int matrix
        shifter = (offs_k % n_bits)[:, None]

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs)
            # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
            b = tl.load(b_ptrs)

            # Convert B from int to a.dtype, for each bit in B, 0 becomes -1.0, 1 becomes 1.0
            # b: (BLOCK_SIZE_K, BLOCK_SIZE_N)
            b = (b >> shifter) & 0x1
            # b = b.to(a.dtype) * 2 - 1
            b = (2 * b - 1).to(a.dtype)

            # Simply convert to a.dtype
            # b = b.to(a.dtype)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            # b_ptrs += BLOCK_SIZE_K * stride_bk
            b_ptrs += (BLOCK_SIZE_K // n_bits) * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        # if ACTIVATION == "leaky_relu":
        #     accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_batch * stride_batch_c
        )
        tl.store(c_ptrs, c)

    def get_random_input(self, fixed: bool = False):
        a = torch.randn(
            (self.batch_size, self.matrix_size, self.matrix_size),
            device="cuda",
            dtype=torch.float16,
        )
        b = torch.randint(
            torch.iinfo(torch.int32).min,
            torch.iinfo(torch.int32).max,
            (
                self.batch_size,
                self.matrix_size // self.n_bits,
                self.matrix_size,
            ),
            device="cuda",
            dtype=torch.int32,
        )
        c = torch.empty_like(a)
        return a, b, c

    def get_shape_information(self) -> str:
        packed_k = self.matrix_size // self.n_bits
        return (
            f"- a_ptr: float16 tensor with shape ({self.batch_size}, "
            f"{self.matrix_size}, {self.matrix_size})\n"
            f"- b_ptr: int32 tensor with shape ({self.batch_size}, {packed_k}, "
            f"{self.matrix_size}) containing 32 packed binary weights per element\n"
            f"- c_ptr: float16 tensor with shape ({self.batch_size}, "
            f"{self.matrix_size}, {self.matrix_size})"
        )

    def forward_triton(self, inputs, ptx=False):
        a, b, c = inputs
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (
            triton.cdiv(self.matrix_size, meta["BLOCK_SIZE_M"])
            * triton.cdiv(self.matrix_size, meta["BLOCK_SIZE_N"]),
            a.shape[0],
        )
        kernel = launch_kernel[grid](
            a,
            b,
            c,
            self.matrix_size,
            self.matrix_size,
            self.matrix_size,
            self.n_bits,
            a.stride(1),
            a.stride(2),
            b.stride(1),
            b.stride(2),
            c.stride(1),
            c.stride(2),
            a.stride(0),
            b.stride(0),
            c.stride(0),
            BLOCK_SIZE_M=self.block_size_m,
            BLOCK_SIZE_N=self.block_size_n,
            BLOCK_SIZE_K=self.block_size_k,
            GROUP_SIZE_M=self.group_m,
            ACTIVATION="",
            **launch_kwargs,
        )
        return c, kernel

    def forward_torch(self, inputs):
        a, packed_b, _ = inputs
        bit_offsets = torch.arange(self.n_bits, device=packed_b.device)
        b = ((packed_b[..., None] >> bit_offsets) & 1).permute(0, 1, 3, 2).reshape(
            a.shape[0], self.matrix_size, self.matrix_size
        )
        return torch.matmul(a, b.to(a.dtype).mul(2).sub(1))
