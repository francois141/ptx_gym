"""
Causal Forgetting Attention forward and query-gradient kernels (ICLR 2025).
https://github.com/zhixuan-lin/forgetting-transformer/tree/main
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class ForgettingAttentionICLR2025Forward(TritonPTXKernel):
    """Compute causal softmax attention with cumulative forgetting gates."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_m": (16, 32, 64, 128),
        "block_n": (16, 32, 64, 128),
        "num_warps": (4, 8, 16),
        "num_stages": (2, 3, 4),
    }

    def __init__(self, *, ptx=None):
        self.batch_size = 16
        self.num_heads = 16
        self.sequence_length = 1024
        self.head_dim = 64
        self.block_m = 32
        self.block_n = 32
        self.num_warps = 4
        self.num_stages = 3
        self.sm_scale = self.head_dim**-0.5
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        q_ptr, k_ptr, v_ptr, log_lambda_ptr, seq_start_ptr, start_index_ptr,
        sm_scale: tl.constexpr, l_ptr, o_ptr,
        stride_qz: tl.constexpr, stride_qh: tl.constexpr,
        stride_qm: tl.constexpr, stride_qk: tl.constexpr,
        stride_kz: tl.constexpr, stride_kh: tl.constexpr,
        stride_kn: tl.constexpr, stride_kk: tl.constexpr,
        stride_vz: tl.constexpr, stride_vh: tl.constexpr,
        stride_vn: tl.constexpr, stride_vk: tl.constexpr,
        stride_log_lambda_z: tl.constexpr,
        stride_log_lambda_h: tl.constexpr,
        stride_log_lambda_n: tl.constexpr,
        stride_start_index_z: tl.constexpr,
        stride_start_index_h: tl.constexpr,
        stride_start_index_mb: tl.constexpr,
        stride_oz: tl.constexpr, stride_oh: tl.constexpr,
        stride_om: tl.constexpr, stride_ok: tl.constexpr,
        Z: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
        N: tl.constexpr, P_SEQ: tl.constexpr,
        num_groups: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
        IS_ADAPTIVE: tl.constexpr,
        DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    ):
        tl.static_assert(
            DIVISIBLE_M and DIVISIBLE_N,
            "DIVISIBLE_M and DIVISIBLE_N must both be true",
        )
        input_dtype = q_ptr.dtype.element_ty
        # -- grid id --
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        off_z = tl.program_id(2)

        # scale sm_scale by log_2(e) and use
        # 2^x instead of exp in the loop because CSE and LICM
        # don't work as expected with `exp` in the loop
        log2e: tl.constexpr = 1.4426950408889634
        loge2: tl.constexpr = 0.6931471805599453
        qk_scale = sm_scale * log2e

        # offset pointers for (batch, head)
        off_hk = off_h // num_groups
        q_ptr += off_z * stride_qz + off_h * stride_qh
        k_ptr += off_z * stride_kz + off_hk * stride_kh
        v_ptr += off_z * stride_vz + off_hk * stride_vh
        log_lambda_ptr += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
        o_ptr += off_z * stride_oz + off_h * stride_oh
        l_ptr += (off_z * H + off_h) * M # l's shape is (B, H, M)

        offs_m_base = tl.arange(0, BLOCK_M)
        offs_m = start_m * BLOCK_M + offs_m_base
        offs_n_base = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        # initialize pointers to value-like data
        q_tile_ptr = q_ptr + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
        log_lambda_out_ptr = log_lambda_ptr + (P_SEQ + offs_m) * stride_log_lambda_n
        o_tile_ptr = o_ptr + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
        l_tile_ptr = l_ptr + offs_m

        # initialize pointer to m and l, fp32 for accumulators
        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        # load q
        if DIVISIBLE_M:
            q = tl.load(q_tile_ptr, cache_modifier=".cg")
            log_lambda_out = tl.load(log_lambda_out_ptr, cache_modifier=".cg")
        else:
            mask_m = offs_m < M
            q = tl.load(q_tile_ptr, mask=mask_m[:, None], cache_modifier=".cg")
            log_lambda_out = tl.load(log_lambda_out_ptr, mask=mask_m, cache_modifier=".cg")

        #Dot I trick: to place q in registers, it saves shared memory
        # if BLOCK_DMODEL < 128:
        #     I = tl.where(offs_k[:, None] == offs_k,
        #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
        #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        #     q = tl.dot(q, I, input_precision="ieee").to(input_dtype)
        # else:
        #     I = tl.where(offs_m_base[:, None] == offs_m_base,
        #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
        #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
        #     q = tl.dot(I, q, input_precision="ieee").to(input_dtype)

        # NOTE: Loop-Bound-For-N
        # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
        # According to the rule of causal masking, then max index in n-dimension that this block may access
        # is `P_SEQ + (start_m + 1) * BLOCK_M`.
        # However, the upper bound of index in n-dimension should never exceed the sequence length of k/v(`P_SEQ + N_CTX`).
        # `P_SEQ + (start_m + 1) * BLOCK_M` may be larger than `N`.
        # At this case, there would be illegal memory access when loading k & v tiles
        # if mask_n is not applied for loading(only when `DIVISIBLE_N`` is true).
        # See also https://github.com/FlagOpen/FlagAttention/pull/8
        if IS_CAUSAL:
            hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
            if LARGER_M:
                hi = tl.maximum(0, hi)
        else:
            hi = N

        offs_n_init = offs_n_base

        if HAS_SEQ_START:
            seq_start_ptr += off_z
            seq_start = tl.load(seq_start_ptr)
            lo = tl.minimum(seq_start, hi)
        else:
            lo = 0
            seq_start = 0

        if IS_ADAPTIVE:
            # No need to multiple start_m by BLOCK_M here
            start_index_ptr += off_z * stride_start_index_z + off_h * stride_start_index_h + start_m * stride_start_index_mb
            start_index = tl.load(start_index_ptr)
            lo = tl.maximum(start_index, lo)
        lo = (lo // BLOCK_N) * BLOCK_N
        offs_n_init += lo

        # loop over k, v and update accumulators
        k_tile_ptr = k_ptr + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn) # (BLOCK_DMODEL, BLOCK_N)
        v_tile_ptr = v_ptr + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
        log_lambda_in_ptr = log_lambda_ptr + (offs_n_init * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + offs_n_base

            # -- load k, v --
            if DIVISIBLE_N:
                k = tl.load(k_tile_ptr, cache_modifier=".cg")
                v = tl.load(v_tile_ptr, cache_modifier=".cg")
                log_lambda_in = tl.load(log_lambda_in_ptr, cache_modifier=".cg")
            else:
                mask_n = offs_n < N
                k = tl.load(k_tile_ptr, mask=mask_n[None, :], cache_modifier=".cg")
                v = tl.load(v_tile_ptr, mask=mask_n[:, None], cache_modifier=".cg")
                log_lambda_in = tl.load(log_lambda_in_ptr, mask=mask_n, cache_modifier=".cg")

            # -- compute qk ---
            # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            if BLOCK_M > 1:
                s = tl.dot(q, k, input_precision="ieee") * qk_scale
            else:
                # (1, D), (D, T)
                s = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True) * qk_scale
            decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
            s += decay_bias * log2e

            if not DIVISIBLE_N:
                s = tl.where(mask_n[None, :], s, float("-inf"))
            if IS_CAUSAL:
                causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                s = tl.where(causal_mask, s, float("-inf"))
            if HAS_SEQ_START:
                s = tl.where(offs_n[None, :] >= seq_start, s, float("-inf"))

            # -- compute scaling constant ---
            m_i_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2((m_i - m_i_new))
            p = tl.math.exp2(s - m_i_new[:, None])

            # -- compute partial sumexpn before applying dropout
            p_sum = tl.sum(p, 1)

            # -- scale and update acc: acc *= alpha[:, None]--
            acc *= alpha[:, None]
            if BLOCK_M > 1:
                acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
            else:
                acc += tl.sum(p.T * v, axis=0, keep_dims=True)

            # -- update m_i and l_i --
            l_i = l_i * alpha + p_sum
            m_i = m_i_new
            # update pointers
            k_tile_ptr += BLOCK_N * stride_kn
            v_tile_ptr += BLOCK_N * stride_vn
            log_lambda_in_ptr += BLOCK_N * stride_log_lambda_n

        # write back l & o
        if IS_CAUSAL and (LARGER_M or HAS_SEQ_START):
            is_empty_line = (offs_m + P_SEQ) < seq_start
            acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
            l = tl.where(is_empty_line, float("-inf"), m_i * loge2 + tl.log(l_i))
        else:
            acc = acc * (1.0 / l_i[:, None])
            l = m_i * loge2 + tl.log(l_i) # log(normalizer)

        if DIVISIBLE_M:
            tl.store(l_tile_ptr, l, cache_modifier=".cg")
            tl.store(o_tile_ptr, acc.to(input_dtype), cache_modifier=".cg")
        else:
            tl.store(l_tile_ptr, l, mask=mask_m, cache_modifier=".cg")
            tl.store(o_tile_ptr, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


    def get_random_input(self, fixed=False):
        del fixed
        shape = (self.batch_size, self.num_heads, self.sequence_length, self.head_dim)
        return (
            torch.randn(shape, device="cuda", dtype=torch.float16),
            torch.randn(shape, device="cuda", dtype=torch.float16),
            torch.randn(shape, device="cuda", dtype=torch.float16),
            torch.randn(
                self.batch_size,
                self.num_heads,
                self.sequence_length,
                device="cuda",
                dtype=torch.float32,
            ).cumsum(-1),
        )

    def get_shape_information(self):
        return "- q/k/v: [B, H, T, D]; log_lambda: [B, H, T]; output: [B, H, T, D]"

    def forward_triton(self, inputs, ptx=False):
        q, k, v, log_lambda = inputs
        output = torch.empty_like(q)
        l = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)
        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        compiled = launch[
            (
                triton.cdiv(q.shape[2], self.block_m),
                q.shape[1],
                q.shape[0],
            )
        ](
            q,
            k,
            v,
            log_lambda,
            None,
            None,
            self.sm_scale,
            l,
            output,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            log_lambda.stride(0),
            log_lambda.stride(1),
            log_lambda.stride(2),
            0,
            0,
            0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
            0,
            q.shape[1] // k.shape[1],
            BLOCK_M=self.block_m,
            BLOCK_DMODEL=q.shape[3],
            BLOCK_N=self.block_n,
            IS_CAUSAL=False,
            LARGER_M=False,
            HAS_SEQ_START=False,
            IS_ADAPTIVE=False,
            DIVISIBLE_M=q.shape[2] % self.block_m == 0,
            DIVISIBLE_N=k.shape[2] % self.block_n == 0,
            **kwargs,
        )
        return (output, l), compiled

    def forward_torch(self, inputs):
        q, k, v, log_lambda = inputs
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.sm_scale
        scores += log_lambda[..., :, None] - log_lambda[..., None, :]
        mask = torch.tril(
            torch.ones(
                self.sequence_length,
                self.sequence_length,
                device=q.device,
                dtype=torch.bool,
            )
        )
        scores = scores.masked_fill(~mask, -torch.inf)
        l = torch.logsumexp(scores, dim=-1)
        return torch.softmax(scores, dim=-1).to(v.dtype) @ v, l


class ForgettingAttentionICLR2025Backward(TritonPTXKernel):
    """Compute the query gradient of causal Forgetting Attention."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_m": (16, 32, 64, 128),
        "block_n": (16, 32, 64, 128),
        "num_warps": (4, 8, 16),
        "num_stages": (2, 3, 4),
    }

    def __init__(self, *, ptx=None):
        self.batch_size = 16
        self.num_heads = 16
        self.sequence_length = 1024
        self.head_dim = 64
        self.block_m = 32
        self.block_n = 32
        self.num_warps = 4
        self.num_stages = 3
        self.sm_scale = self.head_dim**-0.5
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        q_ptr, k_ptr, v_ptr, log_lambda_ptr, seq_start_ptr, end_index_ptr,
        sm_scale: tl.constexpr, do_ptr, dk_ptr, dv_ptr, dlog_lambda_ptr, l_ptr,
        delta_ptr,
        stride_qz: tl.constexpr, stride_qh: tl.constexpr,
        stride_qm: tl.constexpr, stride_qk: tl.constexpr,
        stride_kz: tl.constexpr, stride_kh: tl.constexpr,
        stride_kn: tl.constexpr, stride_kk: tl.constexpr,
        stride_vz: tl.constexpr, stride_vh: tl.constexpr,
        stride_vn: tl.constexpr, stride_vk: tl.constexpr,
        stride_log_lambda_z: tl.constexpr,
        stride_log_lambda_h: tl.constexpr,
        stride_log_lambda_n: tl.constexpr,
        stride_start_index_z: tl.constexpr,
        stride_start_index_h: tl.constexpr,
        stride_start_index_nb: tl.constexpr,
        stride_doz: tl.constexpr, stride_doh: tl.constexpr,
        stride_dom: tl.constexpr, stride_dok: tl.constexpr,
        stride_dkz: tl.constexpr, stride_dkh: tl.constexpr,
        stride_dkn: tl.constexpr, stride_dkk: tl.constexpr,
        stride_dvz: tl.constexpr, stride_dvh: tl.constexpr,
        stride_dvn: tl.constexpr, stride_dvk: tl.constexpr,
        stride_dlog_lambda_z: tl.constexpr,
        stride_dlog_lambda_h: tl.constexpr,
        stride_dlog_lambda_n: tl.constexpr,
        Z: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
        N: tl.constexpr, P_SEQ: tl.constexpr,
        num_groups: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, HAS_SEQ_START: tl.constexpr,
        IS_ADAPTIVE: tl.constexpr
    ):
        tl.static_assert(
            DIVISIBLE_M and DIVISIBLE_N,
            "DIVISIBLE_M and DIVISIBLE_N must both be true",
        )
        input_dtype = q_ptr.dtype.element_ty
        # -- grid id --
        start_n = tl.program_id(0)
        off_h = tl.program_id(1)
        off_z = tl.program_id(2)
        log2e: tl.constexpr = 1.4426950408889634
        qk_scale = sm_scale * log2e

        # offset pointers for (batch, head)
        off_hk = off_h // num_groups
        q_ptr += off_z * stride_qz + off_h * stride_qh
        k_ptr += off_z * stride_kz + off_hk * stride_kh
        v_ptr += off_z * stride_vz + off_hk * stride_vh
        log_lambda_ptr += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
        do_ptr += off_z * stride_doz + off_h * stride_doh

        # offset pointers for batch/head
        dk_ptr += off_z * stride_dkz + off_h * stride_dkh
        dv_ptr += off_z * stride_dvz + off_h * stride_dvh
        dlog_lambda_ptr += off_z * stride_dlog_lambda_z + off_h * stride_dlog_lambda_h

        # offset pointers for batch/head
        delta_ptr += (off_z * H + off_h) * M
        l_ptr += (off_z * H + off_h) * M

        if CAUSAL:
            lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
            lo = (lo // BLOCK_M) * BLOCK_M
        else:
            lo = 0

        offs_m_init = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_base = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        # initialize pointers to value-like data
        q_tile_ptr = q_ptr + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
        log_lambda_out_ptr = log_lambda_ptr + (P_SEQ + offs_m_init) * stride_log_lambda_n # (BLOCK_N, BLOCK_DMODEL)
        k_tile_ptr = k_ptr + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
        v_tile_ptr = v_ptr + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
        log_lambda_in_ptr = log_lambda_ptr + (offs_n * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
        do_tile_ptr = do_ptr + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

        dv_tile_ptr = dv_ptr + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk) # (BLOCK_N, BLOCK_DMODEL)
        dk_tile_ptr = dk_ptr + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk) # (BLOCK_N, BLOCK_DMODEL)
        dlog_lambda_in_ptr = dlog_lambda_ptr + (offs_n * stride_dlog_lambda_n) # (BLOCK_N, BLOCK_DMODEL)

        # k and v stay in SRAM throughout
        if DIVISIBLE_N:
            v = tl.load(v_tile_ptr)
            k = tl.load(k_tile_ptr)
            log_lambda_in = tl.load(log_lambda_in_ptr)
        else:
            mask_n = offs_n < N
            v = tl.load(v_tile_ptr, mask=mask_n[:, None])
            k = tl.load(k_tile_ptr, mask=mask_n[:, None])
            log_lambda_in = tl.load(log_lambda_in_ptr, mask=mask_n)

        # If the N block doesn't contain seq_start, no need to loop
        hi = M
        if IS_ADAPTIVE:
            end_index_ptr += off_z * stride_start_index_z + off_h * stride_start_index_h + start_n * stride_start_index_nb
            hi = tl.minimum(tl.load(end_index_ptr), M)
        else:
            hi = M

        # Ignore this column if seq_start larger than the this column
        if HAS_SEQ_START:
            seq_start_ptr += off_z
            seq_start = tl.load(seq_start_ptr)
            hi = tl.where(start_n * BLOCK_N + BLOCK_N >= seq_start - 1, hi, lo)

        # initialize dk amd dv
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dlog_lambda_in = tl.zeros([BLOCK_N], dtype=tl.float32)

        # loop over a col
        for start_m in range(lo, hi, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + offs_m_base
            causal_mask = (P_SEQ + offs_m[None, :]) >= (offs_n[:, None]) # (BLOCK_M, BLOCK_N)

            # load q1, k1, q2, k2, v, do on-chip
            if DIVISIBLE_M:
                q = tl.load(q_tile_ptr)
                log_lambda_out = tl.load(log_lambda_out_ptr)
            else:
                mask_m = offs_m < M
                valid_mask = mask_m[None, :] # & mask_n
                q = tl.load(q_tile_ptr, mask=mask_m[:, None])
                log_lambda_out = tl.load(log_lambda_out_ptr, mask=mask_m)
            # recompute p = softmax(qk * sm_scale, dim=-1)
            # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            sT = tl.dot(k, tl.trans(q), input_precision="ieee") * qk_scale
            decay_bias = log_lambda_out[None, :] - log_lambda_in[:, None]
            sT += decay_bias * log2e
            # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
            # So masking on s is not needed.
            # s = tl.where(valid_mask, s , float("-inf"))
            # if CAUSAL:
            #     s = tl.where(causal_mask, s, float("-inf"))

            # -- recompute p ---
            if DIVISIBLE_M:
                l = tl.load(l_ptr + offs_m)
            else:
                l = tl.load(l_ptr + offs_m, mask=mask_m)
            pT = tl.math.exp2(sT - l[None, :] * log2e) # (BLOCK_M, BLOCK_N)

            if not DIVISIBLE_M:
                pT = tl.where(valid_mask, pT, 0.0)
            if CAUSAL:
                pT = tl.where(causal_mask, pT, 0.0)

            # compute dv = dot(p, do)
            if DIVISIBLE_M:
                do = tl.load(do_tile_ptr)
            else:
                do = tl.load(do_tile_ptr, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)


            dv += tl.dot(pT.to(input_dtype), do, input_precision="ieee") # (BLOCK_N, BLOCK_DMODEL)  # still correct

            # compute dp = dot(v, do)
            if DIVISIBLE_M:
                delta = tl.load(delta_ptr + offs_m)
            else:
                delta = tl.load(delta_ptr + offs_m, mask=mask_m)
            # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            dpT = tl.dot(v, tl.trans(do), input_precision="ieee")


            # compute ds = p * (dp - delta[:, None])
            dsT = pT * (dpT - delta[None, :]) # (BLOCK_M, BLOCK_N)

            if not DIVISIBLE_M:
                dsT = tl.where(valid_mask, dsT, 0.0)
            if CAUSAL:
                dsT = tl.where(causal_mask, dsT, 0.0)

            # compute dk = dot(ds.T, q) masking
            dk += tl.dot(dsT.to(input_dtype), q, input_precision="ieee")
            dlog_lambda_in += -tl.sum(dsT, axis=1)

            # increment pointers
            q_tile_ptr += BLOCK_M * stride_qm
            log_lambda_out_ptr += BLOCK_M * stride_log_lambda_n
            do_tile_ptr += BLOCK_M * stride_dom

        dk *= sm_scale
        if HAS_SEQ_START:
            # Mask out 
            seq_mask = (offs_n >= seq_start)
            dk = tl.where(seq_mask[:, None], dk, 0.0)
            dv = tl.where(seq_mask[:, None], dv, 0.0)
            dlog_lambda_in = tl.where(seq_mask, dlog_lambda_in, 0.0)
        if DIVISIBLE_N:
            tl.store(dk_tile_ptr, dk.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL)
            tl.store(dv_tile_ptr, dv.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL,)
            tl.store(dlog_lambda_in_ptr, dlog_lambda_in.to(tl.float32)) # (BLOCK_N, BLOCK_DMODEL,)
        else:
            tl.store(dk_tile_ptr, dk.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
            tl.store(dv_tile_ptr, dv.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
            tl.store(dlog_lambda_in_ptr, dlog_lambda_in.to(tl.float32), mask=mask_n) # (BLOCK_N, BLOCK_DMODEL,)


    def get_random_input(self, fixed=False):
        del fixed
        shape = (self.batch_size, self.num_heads, self.sequence_length, self.head_dim)
        q = torch.randn(shape, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        log_lambda = torch.randn(
            self.batch_size,
            self.num_heads,
            self.sequence_length,
            device="cuda",
            dtype=torch.float32,
        ).cumsum(-1)
        output, l = ForgettingAttentionICLR2025Forward().forward_torch(
            (q, k, v, log_lambda)
        )
        return q, k, v, log_lambda, output, torch.randn_like(output), l

    def get_shape_information(self):
        return "- q/k/v/output/output_gradient: [B, H, T, D]; log_lambda and log-normalizer: [B, H, T]"

    def forward_triton(self, inputs, ptx=False):
        q, k, v, log_lambda, output, output_gradient, l = inputs
        key_gradient = torch.empty_like(k)
        value_gradient = torch.empty_like(v)
        log_lambda_gradient = torch.empty_like(log_lambda)
        delta = (output.float() * output_gradient.float()).sum(dim=-1)
        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        compiled = launch[
            (
                triton.cdiv(k.shape[2], self.block_n),
                q.shape[1],
                q.shape[0],
            )
        ](
            q,
            k,
            v,
            log_lambda,
            None,
            None,
            self.sm_scale,
            output_gradient,
            key_gradient,
            value_gradient,
            log_lambda_gradient,
            l,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            log_lambda.stride(0),
            log_lambda.stride(1),
            log_lambda.stride(2),
            0,
            0,
            0,
            output_gradient.stride(0),
            output_gradient.stride(1),
            output_gradient.stride(2),
            output_gradient.stride(3),
            key_gradient.stride(0),
            key_gradient.stride(1),
            key_gradient.stride(2),
            key_gradient.stride(3),
            value_gradient.stride(0),
            value_gradient.stride(1),
            value_gradient.stride(2),
            value_gradient.stride(3),
            log_lambda_gradient.stride(0),
            log_lambda_gradient.stride(1),
            log_lambda_gradient.stride(2),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
            0,
            q.shape[1] // k.shape[1],
            BLOCK_M=self.block_m,
            BLOCK_DMODEL=q.shape[3],
            BLOCK_N=self.block_n,
            CAUSAL=False,
            DIVISIBLE_M=q.shape[2] % self.block_m == 0,
            DIVISIBLE_N=k.shape[2] % self.block_n == 0,
            HAS_SEQ_START=False,
            IS_ADAPTIVE=False,
            **kwargs,
        )
        return (key_gradient, value_gradient, log_lambda_gradient), compiled

    def forward_torch(self, inputs):
        q, k, v, log_lambda, output, output_gradient, l = inputs
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.sm_scale
        scores += log_lambda[..., :, None] - log_lambda[..., None, :]
        mask = torch.tril(
            torch.ones(
                self.sequence_length,
                self.sequence_length,
                device=q.device,
                dtype=torch.bool,
            )
        )
        probabilities = torch.exp(scores - l[..., :, None]).masked_fill(~mask, 0.0)
        delta = (output.float() * output_gradient.float()).sum(-1, keepdim=True)
        d_scores = probabilities * (
            torch.matmul(output_gradient.float(), v.float().transpose(-1, -2)) - delta
        )
        return torch.matmul(d_scores, k.float()).to(q.dtype) * self.sm_scale
