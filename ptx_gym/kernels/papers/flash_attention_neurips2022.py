"""Causal FlashAttention forward and query-backward kernels (NeurIPS 2022)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel

FLASH_ATTENTION_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "specs" / "flash_attention.spec"
)


class FlashAttentionNeurIPS2022Forward(TritonPTXKernel):
    """Tiled online-softmax attention following the original FlashAttention kernel."""

    verification_tolerance: ClassVar[float] = 1e-2
    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_m": (16, 32, 64),
        "block_n": (16, 32, 64),
        "num_warps": (4, 8),
        "num_stages": (2, 3, 4),
    }

    def __init__(self, *, ptx=None):
        self.batch_size, self.num_heads = 64, 64
        self.sequence_length, self.head_dim = 128, 64
        self.block_m, self.block_n = 32, 32
        self.num_warps, self.num_stages = 4, 3
        self.scale = self.head_dim**-0.5
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Bias_ptr,
        Out_ptr,
        Lse_ptr,
        TMP_ptr,
        softmax_scale: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_bb: tl.constexpr,
        stride_bh: tl.constexpr,
        stride_bm: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        nheads: tl.constexpr,
        seqlen_q: tl.constexpr,
        seqlen_k: tl.constexpr,
        seqlen_q_rounded: tl.constexpr,
        headdim: tl.constexpr,
        CACHE_KEY_SEQLEN_Q: tl.constexpr,
        CACHE_KEY_SEQLEN_K: tl.constexpr,
        BIAS_TYPE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        tl.static_assert(seqlen_q % BLOCK_M == 0)
        tl.static_assert(seqlen_k % BLOCK_N == 0)
        tl.static_assert(headdim == BLOCK_HEADDIM)

        start_m = tl.program_id(0)
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        # off_b = tl.program_id(1)
        # off_h = tl.program_id(2)
        # off_hb = off_b * nheads + off_h
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        # Initialize pointers to Q, K, V
        # Adding parenthesis around indexing might use int32 math instead of int64 math?
        # https://github.com/openai/triton/issues/741
        # I'm seeing a tiny bit of difference (5-7us)
        q_ptrs = (
            Q_ptr
            + off_b * stride_qb
            + off_h * stride_qh
            + (offs_m[:, None] * stride_qm + offs_d[None, :])
        )
        k_ptrs = (
            K_ptr
            + off_b * stride_kb
            + off_h * stride_kh
            + (offs_n[:, None] * stride_kn + offs_d[None, :])
        )
        v_ptrs = (
            V_ptr
            + off_b * stride_vb
            + off_h * stride_vh
            + (offs_n[:, None] * stride_vn + offs_d[None, :])
        )
        if BIAS_TYPE == "vector":
            b_ptrs = Bias_ptr + off_b * stride_bb + off_h * stride_bh + offs_n
        elif BIAS_TYPE == "matrix":
            b_ptrs = (
                Bias_ptr
                + off_b * stride_bb
                + off_h * stride_bh
                + (offs_m[:, None] * stride_bm + offs_n[None, :])
            )
        lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
        q = tl.load(q_ptrs)
        end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k) if IS_CAUSAL else seqlen_k
        for start_n in range(0, end_n, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = tl.load(k_ptrs + start_n * stride_kn)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            if IS_CAUSAL:
                qk = tl.where(
                    offs_m[:, None] >= (start_n + offs_n)[None, :],
                    qk,
                    float("-inf"),
                )
            if BIAS_TYPE != "none":
                if BIAS_TYPE == "vector":
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)[None, :]
                elif BIAS_TYPE == "matrix":
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
                # can then fuse the mult and add into an fma instruction. But if we have bias we need to
                # to multiply with softmax_scale here.
                qk = qk * softmax_scale + bias
                m_ij = tl.maximum(tl.max(qk, 1), lse_i)
                p = tl.exp(qk - m_ij[:, None])
            else:
                m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
                p = tl.exp(qk * softmax_scale - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            # scale acc_o
            acc_o_scale = tl.exp(m_i - m_ij)

            acc_o = acc_o * acc_o_scale[:, None]
            v = tl.load(v_ptrs + start_n * stride_vn)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

            # -- update statistics
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

        o_scale = tl.exp(m_i - lse_i)
        acc_o = acc_o * o_scale[:, None]
        # rematerialize offsets to save registers
        start_m = tl.program_id(0)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # write back l and m
        lse_ptrs = Lse_ptr + off_hb * seqlen_q_rounded + offs_m
        tl.store(lse_ptrs, lse_i)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        out_ptrs = (
            Out_ptr
            + off_b * stride_ob
            + off_h * stride_oh
            + (offs_m[:, None] * stride_om + offs_d[None, :])
        )
        tl.store(out_ptrs, acc_o)

    def get_random_input(self, fixed=False):
        del fixed
        shape = (self.batch_size, self.num_heads, self.sequence_length, self.head_dim)
        return tuple(
            torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)
        )

    def get_shape_information(self):
        return "- q/k/v/output: [B, H, T, D]; lse: [B, H, T]"

    def forward_triton(self, inputs, ptx=False):
        q, k, v = inputs
        output, lse = (
            torch.empty_like(q),
            torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32),
        )
        bias = torch.empty(1, device=q.device, dtype=q.dtype)
        seqlen_q, seqlen_k = q.shape[-2], k.shape[-2]
        headdim = q.shape[-1]
        seqlen_q_rounded = triton.cdiv(seqlen_q, self.block_m) * self.block_m
        tmp = torch.empty(
            self.batch_size * self.num_heads * seqlen_q_rounded,
            device=q.device,
            dtype=torch.float32,
        )
        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        compiled = launch[
            (triton.cdiv(seqlen_q, self.block_m), self.batch_size * self.num_heads)
        ](
            q,
            k,
            v,
            bias,
            output,
            lse,
            tmp,
            self.scale,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            0,
            0,
            0,
            *output.stride()[:3],
            self.num_heads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            headdim,
            seqlen_q,
            seqlen_k,
            BIAS_TYPE="none",
            IS_CAUSAL=False,
            BLOCK_HEADDIM=triton.next_power_of_2(headdim),
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            **kwargs,
        )
        return (output, lse), compiled

    def forward_torch(self, inputs):
        q, k, v = inputs
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(
            ~torch.tril(
                torch.ones(
                    self.sequence_length,
                    self.sequence_length,
                    device=q.device,
                    dtype=torch.bool,
                )
            ),
            -torch.inf,
        )
        return torch.softmax(scores, dim=-1).to(v.dtype) @ v, torch.logsumexp(
            scores, dim=-1
        )

    def volta_arguments(self):
        heads = self.batch_size * self.num_heads
        seqlen_q_rounded = (
            triton.cdiv(self.sequence_length, self.block_m) * self.block_m
        )
        elements = heads * self.sequence_length * self.head_dim
        arrays = (
            ("q", 2, elements, "in"),
            ("k", 2, elements, "in"),
            ("v", 2, elements, "in"),
            ("bias", 2, 1, "in"),
            ("o", 2, elements, "out"),
            ("lse", 4, heads * self.sequence_length, "out"),
            ("tmp", 4, heads * seqlen_q_rounded, "out"),
        )
        return FLASH_ATTENTION_SPEC_PATH, [
            "-g",
            f"{triton.cdiv(self.sequence_length, self.block_m)},{heads}",
            *(
                argument
                for index, (name, width, length, kind) in enumerate(arrays, start=1)
                for argument in (
                    "--array",
                    f"{name}:{index * 0x100000000:#x}:{width}:{length}:{kind}",
                )
            ),
            *(
                argument
                for name, *_ in arrays
                for argument in ("--param", f"ptr:{name}")
            ),
            "--param",
            "int:0",
            "--param",
            "int:0",
            "--dim",
            f"B={self.batch_size}",
            "--dim",
            f"H={self.num_heads}",
            "--dim",
            f"T={self.sequence_length}",
            "--dim",
            f"D={self.head_dim}",
        ]


class FlashAttentionNeurIPS2022Backward(TritonPTXKernel):
    """Tiled query gradient paired with the original FlashAttention forward pass."""

    verification_tolerance: ClassVar[float] = 1e-1
    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_m": (16, 32, 64),
        "block_n": (16, 32, 64),
        "num_warps": (4, 8),
        "num_stages": (2, 3, 4),
    }

    def __init__(self, *, ptx=None):
        self.batch_size, self.num_heads = 64, 64
        self.sequence_length, self.head_dim = 128, 64
        self.block_m, self.block_n = 32, 32
        self.num_warps, self.num_stages = 4, 3
        self.scale = self.head_dim**-0.5
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Bias_ptr,
        DO_ptr,
        DQ_ptr,
        DK_ptr,
        DV_ptr,
        LSE_ptr,
        D_ptr,
        softmax_scale: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_bb: tl.constexpr,
        stride_bh: tl.constexpr,
        stride_bm: tl.constexpr,
        stride_dob: tl.constexpr,
        stride_doh: tl.constexpr,
        stride_dom: tl.constexpr,
        stride_dqb: tl.constexpr,
        stride_dqh: tl.constexpr,
        stride_dqm: tl.constexpr,
        stride_dkb: tl.constexpr,
        stride_dkh: tl.constexpr,
        stride_dkn: tl.constexpr,
        stride_dvb: tl.constexpr,
        stride_dvh: tl.constexpr,
        stride_dvn: tl.constexpr,
        nheads: tl.constexpr,
        seqlen_q: tl.constexpr,
        seqlen_k: tl.constexpr,
        seqlen_q_rounded: tl.constexpr,
        headdim: tl.constexpr,
        CACHE_KEY_SEQLEN_Q: tl.constexpr,
        CACHE_KEY_SEQLEN_K: tl.constexpr,
        BIAS_TYPE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        tl.static_assert(seqlen_q % BLOCK_M == 0)
        tl.static_assert(seqlen_k % BLOCK_N == 0)
        tl.static_assert(headdim == BLOCK_HEADDIM)

        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        # offset pointers for batch/head
        Q_ptr += off_b * stride_qb + off_h * stride_qh
        K_ptr += off_b * stride_kb + off_h * stride_kh
        V_ptr += off_b * stride_vb + off_h * stride_vh
        DO_ptr += off_b * stride_dob + off_h * stride_doh
        DQ_ptr += off_b * stride_dqb + off_h * stride_dqh
        DK_ptr += off_b * stride_dkb + off_h * stride_dkh
        DV_ptr += off_b * stride_dvb + off_h * stride_dvh
        if BIAS_TYPE != "none":
            Bias_ptr += off_b * stride_bb + off_h * stride_bh
        # pointer to row-wise quantities in value-like data
        D_ptr += off_hb * seqlen_q_rounded
        LSE_ptr += off_hb * seqlen_q_rounded
        for start_n in range(0, seqlen_k, BLOCK_N):
            begin_m = (start_n // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0
            offs_qm = begin_m + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            offs_m = tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, BLOCK_HEADDIM)
            q_ptrs = Q_ptr + (offs_qm[:, None] * stride_qm + offs_d[None, :])
            k_ptrs = K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :])
            v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :])
            do_ptrs = DO_ptr + (offs_qm[:, None] * stride_dom + offs_d[None, :])
            dq_ptrs = DQ_ptr + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
            if BIAS_TYPE == "vector":
                b_ptrs = Bias_ptr + offs_n
            elif BIAS_TYPE == "matrix":
                b_ptrs = Bias_ptr + (offs_qm[:, None] * stride_bm + offs_n[None, :])
            dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
            dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            for start_m in range(begin_m, seqlen_q, BLOCK_M):
                start_m = tl.multiple_of(start_m, BLOCK_M)
                offs_m_curr = start_m + offs_m
                q = tl.load(q_ptrs)
                qk = tl.dot(q, tl.trans(k))
                if IS_CAUSAL:
                    qk = tl.where(
                        offs_m_curr[:, None] >= offs_n[None, :],
                        qk,
                        float("-inf"),
                    )
                lse_i = tl.load(LSE_ptr + offs_m_curr)
                p = tl.exp(qk * softmax_scale - lse_i[:, None])
                do = tl.load(do_ptrs)
                dv += tl.dot(tl.trans(p.to(do.dtype)), do)
                dp = tl.dot(do, tl.trans(v))
                di = tl.load(D_ptr + offs_m_curr)
                ds = (p * (dp - di[:, None]) * softmax_scale).to(q.dtype)
                dk += tl.dot(tl.trans(ds), q)
                dq = tl.load(dq_ptrs)
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq)
                dq_ptrs += BLOCK_M * stride_dqm
                q_ptrs += BLOCK_M * stride_qm
                do_ptrs += BLOCK_M * stride_dom
                if BIAS_TYPE == "matrix":
                    b_ptrs += BLOCK_M * stride_bm
            dv_ptrs = DV_ptr + (offs_n[:, None] * stride_dvn + offs_d[None, :])
            dk_ptrs = DK_ptr + (offs_n[:, None] * stride_dkn + offs_d[None, :])
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)

    def get_random_input(self, fixed=False):
        del fixed
        shape = (self.batch_size, self.num_heads, self.sequence_length, self.head_dim)
        q, k, v = (
            torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)
        )
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        output = torch.softmax(scores, dim=-1).to(v.dtype) @ v
        lse = torch.logsumexp(scores, dim=-1)
        return q, k, v, output, torch.randn_like(output), lse

    def get_shape_information(self):
        return "- q/k/v/output/output_gradient: [B, H, T, D]; lse/query_gradient: [B, H, T] and [B, H, T, D]"

    def forward_triton(self, inputs, ptx=False):
        q, k, v, output, output_gradient, lse = inputs
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        bias = torch.empty(1, device=q.device, dtype=q.dtype)
        delta = (output.float() * output_gradient.float()).sum(-1)
        seqlen_q, seqlen_k = q.shape[-2], k.shape[-2]
        headdim = q.shape[-1]
        seqlen_q_rounded = triton.cdiv(seqlen_q, self.block_m) * self.block_m
        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        compiled = launch[(1, self.batch_size * self.num_heads)](
            q,
            k,
            v,
            bias,
            output_gradient,
            dq,
            dk,
            dv,
            lse,
            delta,
            self.scale,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            0,
            0,
            0,
            *output_gradient.stride()[:3],
            *dq.stride()[:3],
            *dk.stride()[:3],
            *dv.stride()[:3],
            self.num_heads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            headdim,
            seqlen_q,
            seqlen_k,
            BIAS_TYPE="none",
            IS_CAUSAL=False,
            BLOCK_HEADDIM=triton.next_power_of_2(headdim),
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            **kwargs,
        )
        return dq, compiled

    def forward_torch(self, inputs):
        q, k, v, output, output_gradient, lse = inputs
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        p = torch.exp(scores - lse[..., :, None])
        delta = (output.float() * output_gradient.float()).sum(-1, keepdim=True)
        ds = (
            p
            * (
                torch.matmul(output_gradient.float(), v.float().transpose(-1, -2))
                - delta
            )
            * self.scale
        ).to(q.dtype)
        return torch.matmul(ds, k)
