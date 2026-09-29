"""
INT8 per-block QK attention following SageAttention's Triton convention.
https://github.com/thu-ml/SageAttention
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class SageAttentionICLR2025(TritonPTXKernel):
    """GQA attention with INT8 Q/K and one dequantization scale per block."""

    verification_tolerance = 1e-2
    autotune_tolerance = 1e-2

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_m": [128],
        "block_n": [64],
        "num_warps": (4, 8),
        "num_stages": (2, 3, 4),
    }

    def __init__(self, *, ptx=None):
        self.batch_size = 16
        self.query_heads = 16
        self.key_value_heads = 2
        self.query_length = 1024
        self.key_value_length = 128
        self.head_dim = 64
        self.block_m = 128
        self.block_n = 64
        self.num_warps = 4
        self.num_stages = 3
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Q_scale_ptr,
        K_scale_ptr,
        Out_ptr,
        mask_ptr,
        Lse_ptr,
        stride_qz: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qn: tl.constexpr,
        stride_kz: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_vz: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_oz: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_on: tl.constexpr,
        stride_maskz: tl.constexpr,
        stride_maskh: tl.constexpr,
        stride_maskm: tl.constexpr,
        stride_maskn: tl.constexpr,
        qo_len: tl.constexpr,
        kv_len: tl.constexpr,
        H: tl.constexpr,
        num_kv_groups: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        RETURN_LSE: tl.constexpr,
    ):
        start_m = tl.program_id(0)

        off_z = tl.program_id(2).to(tl.int64)
        off_h = tl.program_id(1).to(tl.int64)

        q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
        k_scale_offset = (
            off_z * (H // num_kv_groups) + off_h // num_kv_groups
        ) * tl.cdiv(kv_len, BLOCK_N)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)
        Q_ptrs = (
            Q_ptr
            + (off_z * stride_qz + off_h * stride_qh)
            + offs_m[:, None] * stride_qn
            + offs_k[None, :]
        )
        q_scale_ptr = Q_scale_ptr + q_scale_offset + start_m
        K_ptrs = (
            K_ptr
            + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh)
            + offs_n[None, :] * stride_kn
            + offs_k[:, None]
        )
        k_scale_ptr = K_scale_ptr + k_scale_offset
        V_ptrs = (
            V_ptr
            + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh)
            + offs_n[:, None] * stride_vn
            + offs_k[None, :]
        )
        O_block_ptr = (
            Out_ptr
            + (off_z * stride_oz + off_h * stride_oh)
            + offs_m[:, None] * stride_on
            + offs_k[None, :]
        )
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q = tl.load(Q_ptrs)
        q_scale = tl.load(q_scale_ptr)
        for start_n in range(0, kv_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = tl.load(K_ptrs)
            k_scale = tl.load(k_scale_ptr)

            qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = tl.load(V_ptrs)
            acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
            m_i = m_ij
            K_ptrs += BLOCK_N * stride_kn
            k_scale_ptr += 1
            V_ptrs += BLOCK_N * stride_vn
        acc = acc / l_i[:, None]
        tl.store(O_block_ptr, acc.to(Out_ptr.type.element_ty))

        if RETURN_LSE:
            lse_ptrs = Lse_ptr + (off_z * qo_len * H + off_h * qo_len) + offs_m
            l_i = tl.log2(l_i) + m_i
            tl.store(lse_ptrs, l_i)

    def get_random_input(self, fixed=False):
        del fixed
        device = "cuda"
        query = torch.randint(
            -127,
            128,
            (self.batch_size, self.query_heads, self.query_length, self.head_dim),
            device=device,
            dtype=torch.int8,
        )
        key = torch.randint(
            -127,
            128,
            (
                self.batch_size,
                self.key_value_heads,
                self.key_value_length,
                self.head_dim,
            ),
            device=device,
            dtype=torch.int8,
        )
        value = torch.randn(
            self.batch_size,
            self.key_value_heads,
            self.key_value_length,
            self.head_dim,
            device=device,
            dtype=torch.float16,
        )
        query_scale = torch.rand(
            self.batch_size,
            self.query_heads,
            triton.cdiv(self.query_length, self.block_m),
            device=device,
        )
        key_scale = torch.rand(
            self.batch_size,
            self.key_value_heads,
            triton.cdiv(self.key_value_length, self.block_n),
            device=device,
        )
        return query, key, value, query_scale, key_scale

    def get_shape_information(self):
        return (
            f"- query_ptr: int8 tensor with shape ({self.batch_size}, "
            f"{self.query_heads}, {self.query_length}, {self.head_dim})\n"
            f"- key_ptr: int8 tensor with shape ({self.batch_size}, "
            f"{self.key_value_heads}, {self.key_value_length}, {self.head_dim})\n"
            f"- value_ptr/output_ptr: float16 tensors with HND layout\n"
            "- query_scale_ptr/key_scale_ptr: float32 scales per Q/K block"
        )

    def forward_triton(self, inputs, ptx=False):
        query, key, value, query_scale, key_scale = inputs
        if (
            query.shape[2] % 32
            or key.shape[2] % 32
            or query.shape[2] % self.block_m
            or key.shape[2] % self.block_n
            or query.shape[3] % 32
        ):
            raise ValueError(
                "Unmasked SageAttention requires query/key-value lengths aligned "
                f"to BLOCK_M/BLOCK_N ({self.block_m}/{self.block_n}) and a "
                "head dimension divisible by 32."
            )
        output = torch.empty_like(query, dtype=value.dtype)
        mask = torch.empty(1, device=query.device, dtype=torch.int8)
        lse = torch.empty(1, device=query.device, dtype=torch.float32)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        kernel = launch_kernel[
            (query.shape[2] // self.block_m, query.shape[1], query.shape[0])
        ](
            query,
            key,
            value,
            query_scale,
            key_scale,
            output,
            mask,
            lse,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            0,
            0,
            0,
            0,
            query.shape[2],
            key.shape[2],
            H=query.shape[1],
            num_kv_groups=query.shape[1] // key.shape[1],
            HEAD_DIM=query.shape[3],
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            STAGE=4,
            RETURN_LSE=False,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        query, key, value, query_scale, key_scale = inputs
        groups = self.query_heads // self.key_value_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        query_factors = query_scale.repeat_interleave(self.block_m, dim=2)[
            ..., : query.shape[2]
        ]
        key_factors = key_scale.repeat_interleave(groups, dim=1).repeat_interleave(
            self.block_n, dim=2
        )[..., : key.shape[2]]
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        scores *= query_factors[..., :, None] * key_factors[..., None, :]
        probabilities = torch.softmax(scores * 0.6931471805599453, dim=-1)
        return probabilities.to(value.dtype) @ value
