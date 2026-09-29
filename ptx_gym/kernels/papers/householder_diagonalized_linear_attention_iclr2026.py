"""
Core chunked forward and backward kernels from HDLA (ICLR 2026).
https://github.com/Zhangjiefu777/HDLA-Impl/
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class HouseholderDiagonalizedLinearAttentionICLR2026Forward(TritonPTXKernel):
    """Compute HDLA's chunk output from transformed queries and chunk state."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {}

    def __init__(self, *, ptx=None):
        self.batch_size = 2
        self.sequence_length = 128
        self.num_heads = 4
        self.key_dim = 64
        self.value_dim = 64
        self.chunk_size = 32
        self.rank = 2
        self.block_k = 32
        self.block_v = 32
        self.num_warps = 4
        self.num_stages = 3
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        q_ptr, # [B, T, H, K]
        k_ptr, # [B, T, H, K]
        a_ptr, # [B, T, H, K]
        b_ptr, # [B, T, H, K]
        gi_ptr, # [B, T, H, K]
        ge_ptr, # [B, T, H, K]
        # intermediate results
        a_qk_ptr, # [B, T, H, BT]
        a_qb_ptr, # [B, T, H, BT * RANK_AB]
        a_ab_ptr, # [B, T * RANK_AB, H, BT * RANK_AB]
        a_ak_ptr, # [B, T * RANK_AB, H, BT]
        RANK_AB: tl.constexpr,
        scale: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
        BC_AB: tl.constexpr,
        BK: tl.constexpr,
        NC: tl.constexpr,
    ):
        q = q_ptr
        k = k_ptr
        a = a_ptr
        b = b_ptr
        gi = gi_ptr
        ge = ge_ptr
        Aqk = a_qk_ptr
        Aqb = a_qb_ptr
        Aab = a_ab_ptr
        Aak = a_ak_ptr
        # grid shape: grid = (NT, NC * NC, B * H)
        i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H

        i_i, i_j = i_c // NC, i_c % NC
        
        if i_t * BT + i_i * BC >= T:
            return
        
        if i_i <= i_j: 
            return
        
        b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqb = tl.zeros([BC, BC_AB], dtype=tl.float32)
        b_Aab = tl.zeros([BC_AB, BC_AB], dtype=tl.float32)
        b_Aak = tl.zeros([BC_AB, BC], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK) 
            m_k = o_k < K

            # q: [B, T, H, K]
            p_q = tl.make_block_ptr(
                q + i_b * T * H * K + i_h * K, 
                (T, K), 
                (H * K, 1), 
                (i_t * BT + i_i * BC, i_k * BK), 
                (BC, BK), 
                (1, 0)
            ) # p_q: [BC, BK]

            # a: [B, T * RANK_AB, H, K]
            p_a = tl.make_block_ptr(
                a + i_b * (T * RANK_AB) * H * K + i_h * K, 
                (T * RANK_AB, K), 
                (H * K, 1), 
                (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), i_k * BK), 
                (BC_AB, BK), 
                (1, 0), 
            ) # [BC * RANK_AB, BK]
            
            # ge: [B, T, H, K]
            p_ga_e = tl.make_block_ptr(
                ge + i_b * T * H * K + i_h * K, 
                (T, K), 
                (H * K, 1), 
                (i_t * BT + i_i * BC, i_k * BK), 
                (BC, BK), 
                (1, 0)
            ) # [BC, BK]

            # gi: [B, T, H, K]
            p_gq_i = tl.make_block_ptr(
                gi + i_b * T * H * K + i_h * K, 
                (T, K), 
                (H * K, 1), 
                (i_t * BT + i_i * BC, i_k * BK), 
                (BC, BK), 
                (1, 0)
            ) # [BC, BK]
            
            # k: [B, T, H, K]
            p_k = tl.make_block_ptr(
                k + i_b * T * H * K + i_h * K, 
                (K, T), 
                (1, H * K), 
                (i_k * BK, i_t * BT + i_j * BC), 
                (BK, BC), 
                (0, 1)
            ) # [BK, BC]

            # b: [B, T * RANK_AB, H, K]
            p_b = tl.make_block_ptr(
                b + i_b * (T * RANK_AB) * H * K + i_h * K, 
                (K, T * RANK_AB), 
                (1, H * K), 
                (i_k * BK, i_t * (BT * RANK_AB) + i_j * (BC * RANK_AB)), 
                (BK, BC_AB), 
                (0, 1)
            ) # [BK, BC * RANK_AB]

            # gk: [B, T, H, K]
            p_gk = tl.make_block_ptr(
                gi + i_b * T * H * K + i_h * K, 
                (K, T), 
                (1, H * K), 
                (i_k * BK, i_t * BT + i_j * BC), 
                (BK, BC), 
                (0, 1)
            ) # [BK, BC] 

            p_gn = tl.max_contiguous( 
                tl.multiple_of(
                    gi + (
                        (i_b * T + i_t * BT + i_i * BC - 1) * H + i_h
                    ) * K + o_k,
                    BK
                ), 
                BK
            ) # [BK, ]

            # [BK, ]
            b_gn = tl.load(p_gn, mask=m_k, other=0) 
            
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1)) # [BC, BK]
            b_a = tl.load(p_a, boundary_check=(0, 1)) # [BC, BK]

            b_gq_i = tl.load(p_gq_i, boundary_check=(0, 1)) # [BC, BK], inclusive cumsum of query gate
            b_ga_e_rank1 = tl.load(p_ga_e, boundary_check=(0, 1))
            b_ag = tl.zeros([BC_AB, BK], dtype=tl.float32)

            # if RANK_AB == 2:
            b_ga_e = tl.interleave(b_ga_e_rank1.T, b_ga_e_rank1.T) # [BK, BC * 2]
            b_ga_e = b_ga_e.T # [BC * 2, BK]
            
            b_ag = b_a * tl.exp(b_ga_e - b_gn[None, :]) # [BC * 2, BK]
            # else:
            #     b_ag = b_a * tl.exp(b_ga_e_rank1 - b_gn[None, :]) # [BC, BK]
            
            b_qg = b_q * tl.exp(b_gq_i - b_gn[None, :]) * scale # [BC, BK] * ([BC, BK] - [1, BK]) -> [BC, BK]
            
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_b = tl.load(p_b, boundary_check=(0, 1))

            b_gk = tl.load(p_gk, boundary_check=(0, 1))

            tmp_k = tl.exp(b_gn[:, None] - b_gk)
            b_kg = b_k * tmp_k # [BK, BC] * [BK, BC]

            # if RANK_AB == 2:
            tmp_b = tl.interleave(tmp_k, tmp_k) # [BK, BC] -> [BK, BC * 2]
            b_bg = b_b * tmp_b # [BK, BC * 2] * [BK, BC * 2]
            # else:
            #     b_bg = b_b * tmp_k # [BK, BC] * [BK, BC]

            # [BC, BC] using tf32 to improve precision here.
            b_Aab += tl.dot(b_ag, b_bg) # [BC * RANK_AB, BK] @ [BK, BC * RANK_AB] -> [BC * RANK_AB, BC * RANK_AB]
            b_Aak += tl.dot(b_ag, b_kg) # [BC * RANK_AB, BK] @ [BK, BC] -> [BC * RANK_AB, BC]
            b_Aqk += tl.dot(b_qg, b_kg) # [BC, BK] @ [BK, BC] -> [BC, BC]
            b_Aqb += tl.dot(b_qg, b_bg) # [BC, BK] @ [BK, BC * RANK_AB] -> [BC, BC * RANK_AB]

            # Aqk: [B, T, H, BT]
            p_Aqk = tl.make_block_ptr(
                Aqk + i_b * T * H * BT + i_h * BT, 
                (T, BT), 
                (H * BT, 1), 
                (i_t * BT + i_i * BC, i_j * BC), 
                (BC, BC), 
                (1, 0)
            )

            # Aqb: [B, T, H, BT * RANK_AB]
            p_Aqb = tl.make_block_ptr(
                Aqb + i_b * T * H * (BT * RANK_AB) + i_h * (BT * RANK_AB), 
                (T, BT * RANK_AB), 
                (H * BT * RANK_AB, 1), 
                (i_t * BT + i_i * BC, i_j * (BC * RANK_AB)), 
                (BC, BC_AB), 
                (1, 0)
            )

            # Aab: [B, T * RANK_AB, H, BT * RANK_AB]
            p_Aab = tl.make_block_ptr(
                Aab + i_b * (T * RANK_AB) * H * (BT * RANK_AB) + i_h * BT * RANK_AB, 
                (T * RANK_AB, BT * RANK_AB), 
                (H * BT * RANK_AB, 1), 
                (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), i_j * (BC * RANK_AB)), 
                (BC_AB, BC_AB), 
                (1, 0)
            )

            # Aak: [B, T * RANK_AB, H, BT]
            p_Aak = tl.make_block_ptr(
                Aak + i_b * (T * RANK_AB) * H * BT + i_h * BT, 
                (T * RANK_AB, BT), 
                (H * BT, 1), 
                (i_t * (BT * RANK_AB) + i_i * (BC * RANK_AB), i_j * BC), 
                (BC_AB, BC), 
                (1, 0)
            )

            tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_Aqb, b_Aqb.to(Aqb.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_Aab, b_Aab.to(Aab.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_Aak, b_Aak.to(Aak.dtype.element_ty), boundary_check=(0, 1))

    def get_random_input(self, fixed=False):
        del fixed
        device = "cuda"
        dtype = torch.float16
        return (
            torch.randn(
                self.batch_size,
                self.sequence_length,
                self.num_heads,
                self.key_dim,
                device=device,
                dtype=dtype,
            ),
            torch.randn(
                self.batch_size,
                self.sequence_length,
                self.num_heads,
                self.key_dim,
                device=device,
                dtype=dtype,
            ),
            torch.randn(
                self.batch_size,
                self.sequence_length * self.rank,
                self.num_heads,
                self.key_dim,
                device=device,
                dtype=dtype,
            ),
            torch.randn(
                self.batch_size,
                self.sequence_length * self.rank,
                self.num_heads,
                self.key_dim,
                device=device,
                dtype=dtype,
            ),
            torch.randn(
                self.batch_size,
                self.sequence_length,
                self.num_heads,
                self.key_dim,
                device=device,
                dtype=torch.float32,
            ).cumsum(1),
            torch.randn(
                self.batch_size,
                self.sequence_length,
                self.num_heads,
                self.key_dim,
                device=device,
                dtype=torch.float32,
            ).cumsum(1),
        )

    def get_shape_information(self):
        return (
            "- q_ptr/k_ptr/gi_ptr/ge_ptr: [B, T, H, K]\n"
            "- a_ptr/b_ptr: [B, T * rank, H, K]\n"
            "- a_qk_ptr: [B, T, H, BT]; a_qb_ptr: [B, T, H, BT * rank]\n"
            "- a_ab_ptr: [B, T * rank, H, BT * rank]; "
            "a_ak_ptr: [B, T * rank, H, BT]"
        )

    def forward_triton(self, inputs, ptx=False):
        q, k, a, b, gi, ge = inputs
        batch_size, sequence_length, num_heads, key_dim = q.shape
        block_c = min(16, self.chunk_size)
        num_chunks = triton.cdiv(sequence_length, self.chunk_size)
        num_c_blocks = triton.cdiv(self.chunk_size, block_c)
        a_qk = torch.empty(
            (batch_size, sequence_length, num_heads, self.chunk_size),
            device=q.device,
            dtype=q.dtype,
        )
        a_qb = torch.empty(
            (batch_size, sequence_length, num_heads, self.chunk_size * self.rank),
            device=q.device,
            dtype=q.dtype,
        )
        a_ab = torch.empty(
            (
                batch_size,
                sequence_length * self.rank,
                num_heads,
                self.chunk_size * self.rank,
            ),
            device=q.device,
            dtype=torch.float32,
        )
        a_ak = torch.empty(
            (batch_size, sequence_length * self.rank, num_heads, self.chunk_size),
            device=q.device,
            dtype=torch.float32,
        )
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        kernel = launch_kernel[
            (
                num_chunks,
                num_c_blocks * num_c_blocks,
                batch_size * num_heads,
            )
        ](
            q,
            k,
            a,
            b,
            gi,
            ge,
            a_qk,
            a_qb,
            a_ab,
            a_ak,
            RANK_AB=self.rank,
            scale=key_dim**-0.5,
            T=sequence_length,
            H=num_heads,
            K=key_dim,
            BT=self.chunk_size,
            BC=block_c,
            BC_AB=block_c * self.rank,
            BK=triton.next_power_of_2(key_dim),
            NC=num_c_blocks,
            **launch_kwargs,
        )
        return (a_qk, a_qb, a_ab, a_ak), kernel

    def forward_torch(self, inputs):
        query_gated, value, value_new, a_qk, a_qb, state = inputs
        batch, time, heads, _ = query_gated.shape
        chunks = time // self.chunk_size
        query = query_gated.view(batch, chunks, self.chunk_size, heads, self.key_dim)
        value_chunked = value.view(
            batch, chunks, self.chunk_size, heads, self.value_dim
        )
        value_new = value_new.view(
            batch, chunks, self.chunk_size * self.rank, heads, self.value_dim
        )
        a_qk = a_qk.view(batch, chunks, self.chunk_size, heads, self.chunk_size)
        a_qb = a_qb.view(
            batch, chunks, self.chunk_size, heads, self.chunk_size * self.rank
        )
        causal_qk = torch.tril(
            torch.ones(self.chunk_size, self.chunk_size, device=value.device)
        )
        rank_columns = (
            torch.arange(self.chunk_size * self.rank, device=value.device) // self.rank
        )
        causal_qb = (
            torch.arange(self.chunk_size, device=value.device)[:, None] >= rank_columns
        )
        output = torch.einsum("bnthk,bnhkv->bnthv", query, state)
        output += torch.einsum(
            "bnhts,bnshv->bnthv", a_qk.permute(0, 1, 3, 2, 4) * causal_qk, value_chunked
        )
        output += torch.einsum(
            "bnhtr,bnrhv->bnthv", a_qb.permute(0, 1, 3, 2, 4) * causal_qb, value_new
        )
        return output.reshape(batch, time, heads, self.value_dim)


class HouseholderDiagonalizedLinearAttentionICLR2026Backward(TritonPTXKernel):
    """Compute HDLA's local A and low-rank-value backward gradients."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {}

    def __init__(self, *, ptx=None):
        self.batch_size = 2
        self.sequence_length = 128
        self.num_heads = 4
        self.value_dim = 64
        self.chunk_size = 32
        self.rank = 2
        self.block_v = 32
        self.scale = 0.125
        self.num_warps = 4
        self.num_stages = 3
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        q_ptr,
        k_ptr,
        a_ptr,
        b_ptr,
        gi_ptr,
        ge_ptr,
        d_a_qk_ptr,
        d_a_qb_ptr,
        d_a_ak_ptr,
        d_a_ab_ptr,
        d_q_ptr,
        d_k_ptr,
        d_a_ptr,
        d_b_ptr,
        d_qg_ptr,
        d_kg_ptr,
        d_ag_ptr,
        d_bg_ptr,
        d_gk_ptr,
        d_gk_offset_ptr,
        cu_seqlens_ptr,
        chunk_indices_ptr,
        scale: tl.constexpr,
        T: tl.constexpr,
        RANK_AB: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BT_AB: tl.constexpr,
        BC: tl.constexpr,
        BC_AB: tl.constexpr,
        BK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        GATHER_SUPPORTED: tl.constexpr,
    ):
        q = q_ptr
        k = k_ptr
        a = a_ptr
        b = b_ptr
        gi = gi_ptr
        ge = ge_ptr
        dAqk = d_a_qk_ptr
        dAqb = d_a_qb_ptr
        dAak = d_a_ak_ptr
        dAab = d_a_ab_ptr
        dq = d_q_ptr
        dk = d_k_ptr
        da = d_a_ptr
        db = d_b_ptr
        dqg = d_qg_ptr
        dkg = d_kg_ptr
        dag = d_ag_ptr
        dbg = d_bg_ptr
        dgk = d_gk_ptr
        dgk_offset = d_gk_offset_ptr
        # grid shape: [NK, NT, B * H]
        i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H

        bos = (i_b * T).to(tl.int32)
        if i_t * BT >= T:
            return

        # offset calculation
        ge += (bos * H + i_h) * K
        gi += (bos * H + i_h) * K
        q += (bos * H + i_h) * K
        a += (bos * RANK_AB * H + i_h) * K
        b += (bos * RANK_AB * H + i_h) * K
        k += (bos * H + i_h) * K

        dq += (bos * H + i_h) * K
        dk += (bos * H + i_h) * K
        da += (bos * RANK_AB * H + i_h) * K
        db += (bos * RANK_AB * H + i_h) * K

        dqg += (bos * H + i_h) * K
        dag += (bos * RANK_AB * H + i_h) * K
        dkg += (bos * H + i_h) * K
        dbg += (bos * RANK_AB * H + i_h) * K

        dgk += (bos * H + i_h) * K
        dgk_offset += (bos * H + i_h) * K

        dAqk += (bos * H + i_h) * BT
        dAqb += (bos * H + i_h) * BT_AB
        dAak += (bos * RANK_AB * H + i_h) * BT
        dAab += (bos * RANK_AB * H + i_h) * BT_AB

        stride_qk = H * K
        stride_A_k = H * BT
        stride_A_b = H * BT_AB

        p_ge = tl.make_block_ptr(ge, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_gi = tl.make_block_ptr(gi, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        # [BC, BK]
        b_ge_kv = tl.load(p_ge, boundary_check=(0, 1))
        b_ge_ab = tl.interleave(b_ge_kv.T, b_ge_kv.T).T
        b_gi_kv = tl.load(p_gi, boundary_check=(0, 1))
        b_gi_ab = tl.interleave(b_gi_kv.T, b_gi_kv.T).T
        b_dq = tl.zeros([BC, BK], dtype=tl.float32)
        b_da = tl.zeros([BC_AB, BK], dtype=tl.float32)
        b_dk = tl.zeros([BC, BK], dtype=tl.float32)
        b_db = tl.zeros([BC_AB, BK], dtype=tl.float32)
        # intra chunk gradient calculation
        p_dAqk = tl.make_block_ptr(
            dAqk, (T, BT), (stride_A_k, 1), (i_t * BT, 0), (BC, BC), (1, 0)
        ) # [BC, BC]
        # dAab: [B, T * RANK_AB, H, BT_AB]
        p_dAab = tl.make_block_ptr(
            dAab, (T * RANK_AB, BT_AB), (stride_A_b, 1), (i_t * BT_AB, 0), 
            (BC_AB, BC_AB), (1, 0)
        )
        # dAqb: [B, T, H, BT_AB]
        p_dAqb = tl.make_block_ptr(
            dAqb, (T, BT_AB), (stride_A_b, 1), (i_t * BT, 0), (BC, BC_AB), (1, 0)
        )
        # dAak: [B, T * RANK_AB, H, BT]
        p_dAak = tl.make_block_ptr(
            dAak, (T * RANK_AB, BT), (stride_A_k, 1), (i_t * BT_AB, 0), (BC_AB, BC), 
            (1, 0)
        )
        
        o_i_kv = tl.arange(0, BC)
        o_i_ab = tl.interleave(o_i_kv, o_i_kv)

        range_BT_AB = tl.arange(0, BC_AB)

        p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
        p_b = tl.make_block_ptr(
            b, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK), 
            (1, 0)
        )
        p_a = tl.make_block_ptr(
            a, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK), 
            (1, 0)
        )
        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
        
        b_k = tl.load(p_k, boundary_check=(0, 1)) # [BC, BK]
        b_b = tl.load(p_b, boundary_check=(0, 1)) # [BC_AB, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)) # [BC, BK]
        b_a = tl.load(p_a, boundary_check=(0, 1)) # [BC_AB, BK]
        b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1)) # [BC, BC]
        b_dAab = tl.load(p_dAab, boundary_check=(0, 1)) # [BC_AB, BC_AB]
        b_dAqb = tl.load(p_dAqb, boundary_check=(0, 1)) # [BC, BC_AB]
        b_dAak = tl.load(p_dAak, boundary_check=(0, 1)) # [BC_AB, BC]

        # inter chunk gradient calculation
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        
        # intra chunk gradient calculation
        for j in range(min(BC, T - i_t * BT)):
            mask_idx_qk = (o_i_kv == j)
            b_kj = tl.sum(tl.where(mask_idx_qk[:, None], b_k, 0), 0)[None, :] # [1, BK]
            b_gij = tl.sum(tl.where(mask_idx_qk[:, None], b_gi_kv, 0), 0)[None, :] # [1, BK]
            b_gej = tl.sum(tl.where(mask_idx_qk[:, None], b_ge_kv, 0), 0)[None, :] # [1, BK]
            
            # the j-th column of dAqk (the attention of all qs on k_j)
            b_dAqk_j = tl.sum(tl.where(mask_idx_qk[None, :], b_dAqk, 0), 1)[:, None] # [BC, 1]
            # the j-th column of dAak (the attention of all as on k_j)
            b_dAak_j = tl.sum(tl.where(mask_idx_qk[None, :], b_dAak, 0), 1)[:, None] # [BC_AB, 1]
            
            # the j-th row of dAqk (the attention of q_j on all ks)
            b_dA_qk_j = tl.sum(tl.where(mask_idx_qk[:, None], b_dAqk, 0), 0)[:, None] # [BC, 1]
            # the j-th row of dAqb (the attention of q_j on all bs)
            b_dA_qb_j = tl.sum(tl.where(mask_idx_qk[:, None], b_dAqb, 0), 0)[:, None] # [BC_AB, 1]
                
            b_qj = tl.sum(tl.where(mask_idx_qk[:, None], b_q, 0), 0)[None, :] # [1, BK]

            # row masks
            # part 1
            m_e_a = o_i_ab[:, None] > j
            m_i_q = o_i_kv[:, None] >= j

            tmp_i_q = tl.exp(b_gi_kv - b_gij) # [BC, BK]
            tmp_e_q = tl.exp(b_ge_kv - b_gij) # [BC, BK]
            tmp_e_a = tl.interleave(tmp_e_q.T, tmp_e_q.T).T # [BC_AB, BK]

            b_dq += tl.where(m_i_q, b_dAqk_j * b_kj * tmp_i_q, 0.) # [BC, 1] * [1, BK] * [BC, BK]
            b_da += tl.where(m_e_a, b_dAak_j * b_kj * tmp_e_a, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]

            # part 2
            m_i_k = o_i_kv[:, None] <= j
            m_i_b = o_i_ab[:, None] <= j
            m_e_k = o_i_kv[:, None] < j
            m_e_b = o_i_ab[:, None] < j

            tmp_i_k = tl.exp(b_gij - b_gi_kv) # [BC, BK]
            tmp_i_b = tl.interleave(tmp_i_k.T, tmp_i_k.T).T # [BC_AB, BK]
            tmp_e_k = tl.exp(b_gej - b_gi_kv) # [BC, BK]
            tmp_e_b = tl.interleave(tmp_e_k.T, tmp_e_k.T).T # [BC_AB, BK]

            b_dk += tl.where(m_i_k, b_dA_qk_j * b_qj * tmp_i_k, 0.) # [BC, 1] * [1, BK] * [BC, BK]
            b_db += tl.where(m_i_b, b_dA_qb_j * b_qj * tmp_i_b, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]

            for i_r in range(RANK_AB):
                
                mask_idx_ab = (range_BT_AB == j * RANK_AB + i_r)
                b_aj = tl.sum(tl.where(mask_idx_ab[:, None], b_a, 0), 0)[None, :] # [1, BK]
                b_bj = tl.sum(tl.where(mask_idx_ab[:, None], b_b, 0), 0)[None, :] # [1, BK]
                
                # the (j * RANK_AB + i_r)-column of dAab (the attention of all as on b_{j, i_r})
                b_dAab_j = tl.sum(tl.where(mask_idx_ab[None, :], b_dAab, 0), 1)[:, None] # [BC_AB, 1]
                # the (j * RANK_AB + i_r)-column of dAqb (the attention of all qs on b_{j, i_r})
                b_dAqb_j = tl.sum(tl.where(mask_idx_ab[None, :], b_dAqb, 0), 1)[:, None] # [BC, 1]
                
                # the (j * RANK_AB + i_r)-th row of dAab (the attention of a_j on all bs)
                b_dA_ab_j = tl.sum(tl.where(mask_idx_ab[:, None], b_dAab, 0), 0)[:, None] # [BC_AB, ]
                # the (j * RANK_AB + i_r)-th row of dAak (the attention of a_j on all ks)
                b_dA_ak_j = tl.sum(tl.where(mask_idx_ab[:, None], b_dAak, 0), 0)[:, None] # [BC, ]

                # part 1
                b_dq += tl.where(m_i_q, b_dAqb_j * b_bj * tmp_i_q, 0.) # [BC, 1] * [1, BK] * [BC, BK]
                b_da += tl.where(m_e_a, b_dAab_j * b_bj * tmp_e_a, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]

                # part 2
                b_dk += tl.where(m_e_k, b_dA_ak_j * b_aj * tmp_e_k, 0.) # [BC, 1] * [1, BK] * [BC, BK]
                b_db += tl.where(m_e_b, b_dA_ab_j * b_aj * tmp_e_b, 0.) # [BC_AB, 1] * [1, BK] * [BC_AB, BK]
        
        p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_da = tl.make_block_ptr(
            da, (T * RANK_AB, K), (stride_qk, 1), 
            (i_t * BT_AB, i_k * BK), (BC_AB, BK), (1, 0)
        )
        p_db = tl.make_block_ptr(
            db, (T * RANK_AB, K), (stride_qk, 1), 
            (i_t * BT_AB, i_k * BK), (BC_AB, BK), (1, 0)
        )
        p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_dqg = tl.make_block_ptr(dqg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_dkg = tl.make_block_ptr(dkg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_dag = tl.make_block_ptr(
            dag, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK),
            (1, 0)
        )
        p_dbg = tl.make_block_ptr(
            dbg, (T * RANK_AB, K), (stride_qk, 1), (i_t * BT_AB, i_k * BK), (BC_AB, BK), 
            (1, 0)
        )
        p_gn = gi + (min(i_t * BT + BT, T) - 1) * stride_qk + o_k
        p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_da += tl.load(p_dag, boundary_check=(0, 1)) * tl.exp(b_ge_ab)
        b_dq += tl.load(p_dqg, boundary_check=(0, 1)) * tl.exp(b_gi_kv) * scale
        
        tmp_k = tl.exp(b_gn[None, :] - b_gi_kv)
        tmp_b = tl.exp(b_gn[None, :] - b_gi_ab)

        b_dk += tl.load(p_dkg, boundary_check=(0, 1)).to(tl.float32) * tmp_k
        b_db += tl.load(p_dbg, boundary_check=(0, 1)).to(tl.float32) * tmp_b

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_da, b_da.to(p_da.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0, 1))

        b_dgk = (
            b_dq * b_q 
            + tl.sum(
                tl.reshape(b_da * b_a, (BC, RANK_AB, BK)), axis=1
            ) 
            - b_dk * b_k 
            - tl.sum(
                tl.reshape(b_db * b_b, (BC, RANK_AB, BK)), axis=1
            )
        ).to(tl.float32)
        b_dgk_offset = tl.sum(
            tl.reshape(b_da * b_a, (BC, RANK_AB, BK)), axis=1
        )
        
        tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dgk_offset, b_dgk_offset.to(p_dgk_offset.dtype.element_ty), boundary_check=(0, 1))

    def get_random_input(self, fixed=False):
        del fixed
        device = "cuda"
        dtype = torch.float16
        q = torch.randn(
            self.batch_size, self.sequence_length, self.num_heads,
            self.value_dim, device=device, dtype=dtype,
        )
        k = torch.randn_like(q)
        a = torch.randn(
            self.batch_size, self.sequence_length * self.rank, self.num_heads,
            self.value_dim, device=device, dtype=dtype,
        )
        b = torch.randn_like(a)
        gi = torch.randn_like(q, dtype=torch.float32).cumsum(1)
        ge = torch.randn_like(q, dtype=torch.float32).cumsum(1)
        d_a_qk = torch.randn(
            self.batch_size, self.sequence_length, self.num_heads,
            self.chunk_size, device=device, dtype=dtype,
        )
        d_a_qb = torch.randn(
            self.batch_size, self.sequence_length, self.num_heads,
            self.chunk_size * self.rank, device=device, dtype=dtype,
        )
        d_a_ak = torch.randn(
            self.batch_size, self.sequence_length * self.rank, self.num_heads,
            self.chunk_size, device=device, dtype=torch.float32,
        )
        d_a_ab = torch.randn(
            self.batch_size, self.sequence_length * self.rank, self.num_heads,
            self.chunk_size * self.rank, device=device, dtype=torch.float32,
        )
        return (
            q, k, a, b, gi, ge, d_a_qk, d_a_qb, d_a_ak, d_a_ab,
            torch.randn_like(q), torch.randn_like(q), torch.randn_like(a),
            torch.randn_like(b),
        )

    def get_shape_information(self):
        return (
            "- value_ptr and output_gradient_ptr: [B, T, H, V]\n"
            "- value_new_ptr: [B, T * rank, H, V]; a_qb_ptr: [B, T, H, BT * rank]\n"
            "- outputs: dA_qk [B, T, H, BT], dA_qb [B, T, H, BT * rank], "
            "and d_value_new [B, T * rank, H, V]"
        )

    def forward_triton(self, inputs, ptx=False):
        (
            q,
            k,
            a,
            b,
            gi,
            ge,
            d_a_qk,
            d_a_qb,
            d_a_ak,
            d_a_ab,
            d_qg,
            d_kg,
            d_ag,
            d_bg,
        ) = inputs
        batch_size, sequence_length, num_heads, key_dim = q.shape
        block_c = min(16, self.chunk_size)
        block_k = triton.next_power_of_2(key_dim)
        d_q = torch.empty_like(q)
        d_k = torch.empty_like(k)
        d_a = torch.empty_like(a)
        d_b = torch.empty_like(b)
        d_gk = torch.empty_like(gi, dtype=torch.float32)
        d_gk_offset = torch.empty_like(gi, dtype=torch.float32)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        kernel = launch_kernel[
            (
                triton.cdiv(key_dim, block_k),
                triton.cdiv(sequence_length, self.chunk_size),
                batch_size * num_heads,
            )
        ](
            q,
            k,
            a,
            b,
            gi,
            ge,
            d_a_qk,
            d_a_qb,
            d_a_ak,
            d_a_ab,
            d_q,
            d_k,
            d_a,
            d_b,
            d_qg,
            d_kg,
            d_ag,
            d_bg,
            d_gk,
            d_gk_offset,
            None,
            None,
            scale=self.scale,
            T=sequence_length,
            RANK_AB=self.rank,
            H=num_heads,
            K=key_dim,
            BT=self.chunk_size,
            BT_AB=self.chunk_size * self.rank,
            BC=block_c,
            BC_AB=block_c * self.rank,
            BK=block_k,
            IS_VARLEN=False,
            GATHER_SUPPORTED=False,
            **launch_kwargs,
        )
        return (d_q, d_k, d_a, d_b, d_gk, d_gk_offset), kernel

    def forward_torch(self, inputs):
        value, output_gradient, value_new, a_qb = inputs
        batch, time, heads, _ = value.shape
        chunks = time // self.chunk_size
        value = value.view(batch, chunks, self.chunk_size, heads, self.value_dim)
        output_gradient = output_gradient.view(
            batch, chunks, self.chunk_size, heads, self.value_dim
        )
        value_new = value_new.view(
            batch, chunks, self.chunk_size * self.rank, heads, self.value_dim
        )
        a_qb = a_qb.view(
            batch, chunks, self.chunk_size, heads, self.chunk_size * self.rank
        )
        do = output_gradient.permute(0, 1, 3, 2, 4)
        d_a_qk = torch.matmul(do, value.permute(0, 1, 3, 4, 2))
        d_a_qb = torch.matmul(do, value_new.permute(0, 1, 3, 4, 2))
        causal_qk = torch.tril(
            torch.ones(self.chunk_size, self.chunk_size, device=value.device)
        )
        rank_columns = (
            torch.arange(self.chunk_size * self.rank, device=value.device) // self.rank
        )
        causal_qb = (
            torch.arange(self.chunk_size, device=value.device)[:, None] >= rank_columns
        )
        d_a_qk = d_a_qk * causal_qk * self.scale
        d_a_qb = d_a_qb * causal_qb * self.scale
        d_value_new = torch.matmul(a_qb.permute(0, 1, 3, 4, 2), do)
        return (
            d_a_qk.permute(0, 1, 3, 2, 4).reshape(batch, time, heads, self.chunk_size),
            d_a_qb.permute(0, 1, 3, 2, 4).reshape(
                batch, time, heads, self.chunk_size * self.rank
            ),
            d_value_new.permute(0, 1, 3, 2, 4).reshape(
                batch, time * self.rank, heads, self.value_dim
            ),
        )
