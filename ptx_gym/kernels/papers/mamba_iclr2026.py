"""Faithful Mamba-3 SISO inference recurrences."""

from __future__ import annotations

from typing import ClassVar

import torch
import triton.language as tl
from torch.nn import functional
from ptx_gym.kernels.base import TritonPTXKernel


def _rotate_torch(values, angles):
    pairs = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    cosine, sine = angles.cos(), angles.sin()
    return torch.stack(
        (
            pairs[..., 0] * cosine - pairs[..., 1] * sine,
            pairs[..., 0] * sine + pairs[..., 1] * cosine,
        ),
        dim=-1,
    ).flatten(-2)


class MambaICLR2026Forward(TritonPTXKernel):
    """Mamba-3 SISO prefill with full trapezoidal state passing."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "num_warps": (2, 4, 8),
        "num_stages": (1, 2, 3),
        "maxnreg": (None, 128, 256),
    }
    verification_tolerance = 1e-1
    autotune_tolerance = 5e-2

    def __init__(self, *, ptx=None):
        self.batch_size, self.sequence_length = 8, 2048
        self.num_heads, self.key_dim, self.value_dim = 16, 64, 64
        self.chunk_size = 128
        self.num_warps, self.num_stages = 4, 1
        self.maxnreg = None
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        # Inputs
        Q_ptr, K_ptr, V_ptr, ADT_ptr, DT_ptr, Trap_ptr, Q_bias_ptr, K_bias_ptr,
        Angles_ptr, D_input_ptr, Z_ptr, Initial_SSM_State_ptr,
        Initial_K_State_ptr, Initial_V_State_ptr, Cu_Seqlens_ptr,
        # Outputs
        Out_ptr, Out_v_ptr, SSM_States_ptr, DA_CS_Store_ptr,
        DA_CS_SUM_Store_ptr, Q_store_ptr, K_store_ptr, QK_store_ptr,
        Scale_store_ptr, Gamma_store_ptr, Final_SSM_State_ptr, Final_K_State_ptr,
        # Input Strides
        stride_q_batch: tl.constexpr, stride_q_seqlen: tl.constexpr,
        stride_q_head: tl.constexpr, stride_q_qkdim: tl.constexpr,
        stride_k_batch: tl.constexpr, stride_k_seqlen: tl.constexpr,
        stride_k_head: tl.constexpr, stride_k_qkdim: tl.constexpr,
        stride_v_batch: tl.constexpr, stride_v_seqlen: tl.constexpr,
        stride_v_head: tl.constexpr, stride_v_vdim: tl.constexpr,
        stride_adt_batch: tl.constexpr, stride_adt_head: tl.constexpr,
        stride_adt_seqlen: tl.constexpr, stride_dt_batch: tl.constexpr,
        stride_dt_head: tl.constexpr, stride_dt_seqlen: tl.constexpr,
        stride_trap_batch: tl.constexpr, stride_trap_head: tl.constexpr,
        stride_trap_seqlen: tl.constexpr, stride_q_bias_head: tl.constexpr,
        stride_q_bias_qkdim: tl.constexpr, stride_k_bias_head: tl.constexpr,
        stride_k_bias_qkdim: tl.constexpr, stride_angles_batch: tl.constexpr,
        stride_angles_seqlen: tl.constexpr, stride_angles_head: tl.constexpr,
        stride_angles_qkdim: tl.constexpr, stride_d_head: tl.constexpr,
        stride_z_batch: tl.constexpr, stride_z_seqlen: tl.constexpr,
        stride_z_head: tl.constexpr, stride_z_vdim: tl.constexpr,
        stride_init_ssm_state_seq: tl.constexpr,
        stride_init_ssm_state_head: tl.constexpr,
        stride_init_ssm_state_vdim: tl.constexpr,
        stride_init_ssm_state_qkdim: tl.constexpr,
        stride_init_k_state_seq: tl.constexpr,
        stride_init_k_state_head: tl.constexpr,
        stride_init_k_state_qkdim: tl.constexpr,
        stride_init_v_state_seq: tl.constexpr,
        stride_init_v_state_head: tl.constexpr,
        stride_init_v_state_vdim: tl.constexpr,
        stride_cu_seqlen: tl.constexpr,
        # Output Strides
        stride_o_batch: tl.constexpr, stride_o_seqlen: tl.constexpr,
        stride_o_head: tl.constexpr, stride_o_vdim: tl.constexpr,
        stride_o_v_batch: tl.constexpr, stride_o_v_seqlen: tl.constexpr,
        stride_o_v_head: tl.constexpr, stride_o_v_vdim: tl.constexpr,
        stride_ssm_states_batch: tl.constexpr,
        stride_ssm_states_head: tl.constexpr,
        stride_ssm_states_vdim: tl.constexpr,
        stride_ssm_states_qkdim: tl.constexpr,
        stride_da_cs_store_batch: tl.constexpr,
        stride_da_cs_store_head: tl.constexpr,
        stride_da_cs_store_seqlen: tl.constexpr,
        stride_da_cs_sum_store_batch: tl.constexpr,
        stride_da_cs_sum_store_head: tl.constexpr,
        stride_da_cs_sum_store_seqlen: tl.constexpr,
        stride_q_store_batch: tl.constexpr,
        stride_q_store_seqlen: tl.constexpr,
        stride_q_store_head: tl.constexpr,
        stride_q_store_qkdim: tl.constexpr,
        stride_k_store_batch: tl.constexpr,
        stride_k_store_seqlen: tl.constexpr,
        stride_k_store_head: tl.constexpr,
        stride_k_store_qkdim: tl.constexpr,
        stride_qk_store_batch: tl.constexpr,
        stride_qk_store_head: tl.constexpr,
        stride_qk_store_seqlen: tl.constexpr,
        stride_scale_store_batch: tl.constexpr,
        stride_scale_store_head: tl.constexpr,
        stride_scale_store_seqlen: tl.constexpr,
        stride_gamma_store_batch: tl.constexpr,
        stride_gamma_store_head: tl.constexpr,
        stride_gamma_store_seqlen: tl.constexpr,
        stride_final_ssm_state_seq: tl.constexpr,
        stride_final_ssm_state_head: tl.constexpr,
        stride_final_ssm_state_vdim: tl.constexpr,
        stride_final_ssm_state_qkdim: tl.constexpr,
        stride_final_k_state_seq: tl.constexpr,
        stride_final_k_state_head: tl.constexpr,
        stride_final_k_state_chunk: tl.constexpr,
        stride_final_k_state_qkdim: tl.constexpr,
        # Dimensions
        seqlen: tl.constexpr, nheads_qk: tl.constexpr,
        headdim_qk: tl.constexpr, headdim_v: tl.constexpr,
        headdim_angles: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        HEADDIM_QK: tl.constexpr,
        HEADDIM_V: tl.constexpr,
        STORE_SSM_STATES_ADT_OUTV: tl.constexpr,
        HAS_INITIAL_STATES: tl.constexpr,
        RETURN_FINAL_STATES: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_Z: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        """
        Mamba-3 forward kernel.

        Grid: (nheads, batch) for batched, (nheads, 1, num_sequences) for varlen

        Inputs:
            Q, K:                       (batch, seqlen, nheads_qk, headdim_qk)
            V:                          (batch, seqlen, nheads, headdim_v)
            ADT, DT, Trap:              (batch, nheads, seqlen)
            Q_bias, K_bias:             (nheads, headdim_qk)
            Angles:                     (batch, seqlen, nheads, headdim_angles)
            D:                          (nheads,)
            Z:                          (batch, seqlen, nheads, headdim_v)
            Initial SSM State:          (num_sequences, nheads, headdim_v, headdim_qk)
            Initial K State:            (num_sequences, nheads, headdim_qk)
            Initial V State:            (num_sequences, nheads, headdim_v)
            Cu_Seqlens:                 (num_sequences + 1,)

        NOTE: num_sequences = batch for batched mode, or len(cu_seqlens)-1 for varlen mode.

        Compile-time constants:
            CHUNK_SIZE:                 Chunk size for processing sequences
            HEADDIM_QK:                 Head dimension for Q/K
            HEADDIM_V:                  Head dimension for V

            STORE_SSM_STATES_ADT_OUTV:  Whether to store SSM states, ADT, and Out_v for backward pass
                                        Set to FALSE for inference-only runs for efficiency
            HAS_INITIAL_STATES:         Whether input SSM states are provided for state passing
            RETURN_FINAL_STATES:        Whether to return final SSM states for state passing
            HAS_D:                      Whether D-skip connection is used
            HAS_Z:                      Whether Z-gating is used
            IS_VARLEN:                  Whether the input is a variable-length sequence

        NOTE:
            1. nheads % nheads_qk == 0
            2. Kernel is optimized for headdim_qk = 128 and headdim_v = 64

        Outputs:
            Out:                    (batch, seqlen, nheads, headdim_v)
            Out_v:                  (batch, seqlen, nheads, headdim_v) (if STORE_SSM_STATES_ADT_OUTV)
            SSM_States:             (batch, nheads, headdim_v, nchunks * headdim_qk) (if STORE_SSM_STATES_ADT_OUTV)
            DA_CS_Store:            (batch, nheads, seqlen) (if STORE_SSM_STATES_ADT_OUTV)
            DA_CS_SUM_Store:        (batch, nheads, nchunks) (if STORE_SSM_STATES_ADT_OUTV)
            Q_store:                (batch, seqlen, nheads, headdim_qk)
            K_store:                (batch, seqlen, nheads, headdim_qk)
            QK_store:               (batch, seqlen, nheads)
            Scale_store:            (batch, seqlen, nheads)
            Gamma_store:            (batch, seqlen, nheads)
            Final SSM State:        (num_sequences, nheads, headdim_v, headdim_qk) (if RETURN_FINAL_STATES)
            Final K State:          (num_sequences, nheads, chunk_size, headdim_qk) (if RETURN_FINAL_STATES)

        NOTE:
        1. For batched inputs, nchunks = ceil(seqlen / CHUNK_SIZE) and for varlen inputs, nchunks = num_sequences +
        total_seqlen//CHUNK_SIZE.
        2. Final K state has an additional chunk_size dimension since triton does not allow indexing within a chunk. We
        pick the correct index in the wrapper.
        """
        pid_head = tl.program_id(0)
        pid_batch = tl.program_id(1)

        if IS_VARLEN:
            pid_seq = tl.program_id(2)
            seq_idx = pid_seq

            cu_seqlen_start = tl.load(
                Cu_Seqlens_ptr + pid_seq * stride_cu_seqlen
            ).to(tl.int32)
            cu_seqlen_end = tl.load(
                Cu_Seqlens_ptr + (pid_seq + 1) * stride_cu_seqlen
            ).to(tl.int32)
            seqlen = cu_seqlen_end - cu_seqlen_start
            seq_offset = cu_seqlen_start
            chunk_offset = pid_seq + cu_seqlen_start // CHUNK_SIZE
        else:
            seq_idx = pid_batch
            seq_offset = 0
            chunk_offset = 0

        num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

        # Compute head index for Q/K (supports Grouped Query Attention)
        nheads = tl.num_programs(0)
        head_idx_qk = pid_head // (nheads // nheads_qk)

        # Setup input pointers
        q_ptr = Q_ptr + pid_batch * stride_q_batch + head_idx_qk * stride_q_head + seq_offset * stride_q_seqlen
        k_ptr = K_ptr + pid_batch * stride_k_batch + head_idx_qk * stride_k_head + seq_offset * stride_k_seqlen
        v_ptr = V_ptr + pid_batch * stride_v_batch + pid_head * stride_v_head + seq_offset * stride_v_seqlen
        adt_ptr = ADT_ptr + pid_batch * stride_adt_batch + pid_head * stride_adt_head + seq_offset * stride_adt_seqlen
        dt_ptr = DT_ptr + pid_batch * stride_dt_batch + pid_head * stride_dt_head + seq_offset * stride_dt_seqlen
        trap_ptr = Trap_ptr + pid_batch * stride_trap_batch + pid_head * stride_trap_head + seq_offset * stride_trap_seqlen
        q_bias_ptr = Q_bias_ptr + pid_head * stride_q_bias_head
        k_bias_ptr = K_bias_ptr + pid_head * stride_k_bias_head
        angle_ptr = Angles_ptr + pid_batch * stride_angles_batch + pid_head * stride_angles_head + seq_offset * stride_angles_seqlen

        if HAS_D:
            D_ptr = D_input_ptr + pid_head * stride_d_head
            D_val = tl.load(D_ptr).to(tl.float32)
        if HAS_Z:
            z_ptr = Z_ptr + pid_batch * stride_z_batch + pid_head * stride_z_head + seq_offset * stride_z_seqlen

        # State pointers use seq_idx (unified for batched and varlen)
        if HAS_INITIAL_STATES:
            init_ssm_state_ptr = Initial_SSM_State_ptr + seq_idx * stride_init_ssm_state_seq + pid_head * stride_init_ssm_state_head
            init_k_state_ptr = Initial_K_State_ptr + seq_idx * stride_init_k_state_seq + pid_head * stride_init_k_state_head
            init_v_state_ptr = Initial_V_State_ptr + seq_idx * stride_init_v_state_seq + pid_head * stride_init_v_state_head

        # Setup output pointers
        o_ptr = Out_ptr + pid_batch * stride_o_batch + pid_head * stride_o_head + seq_offset * stride_o_seqlen
        if STORE_SSM_STATES_ADT_OUTV:
            out_v_ptr = Out_v_ptr + pid_batch * stride_o_v_batch + pid_head * stride_o_v_head + seq_offset * stride_o_v_seqlen
            ssm_states_ptr = SSM_States_ptr + pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head + chunk_offset * HEADDIM_QK * stride_ssm_states_qkdim
            da_cs_store_ptr = DA_CS_Store_ptr + pid_batch * stride_da_cs_store_batch + pid_head * stride_da_cs_store_head + seq_offset * stride_da_cs_store_seqlen
            da_cs_sum_store_ptr = DA_CS_SUM_Store_ptr + pid_batch * stride_da_cs_sum_store_batch + pid_head * stride_da_cs_sum_store_head + chunk_offset * stride_da_cs_sum_store_seqlen

        q_store_ptr = Q_store_ptr + pid_batch * stride_q_store_batch + pid_head * stride_q_store_head + seq_offset * stride_q_store_seqlen
        k_store_ptr = K_store_ptr + pid_batch * stride_k_store_batch + pid_head * stride_k_store_head + seq_offset * stride_k_store_seqlen
        qk_store_ptr = QK_store_ptr + pid_batch * stride_qk_store_batch + pid_head * stride_qk_store_head + seq_offset * stride_qk_store_seqlen
        scale_store_ptr = Scale_store_ptr + pid_batch * stride_scale_store_batch + pid_head * stride_scale_store_head + seq_offset * stride_scale_store_seqlen
        gamma_store_ptr = Gamma_store_ptr + pid_batch * stride_gamma_store_batch + pid_head * stride_gamma_store_head + seq_offset * stride_gamma_store_seqlen

        if RETURN_FINAL_STATES:
            final_ssm_state_ptr = Final_SSM_State_ptr + seq_idx * stride_final_ssm_state_seq + pid_head * stride_final_ssm_state_head
            final_k_state_ptr = Final_K_State_ptr + seq_idx * stride_final_k_state_seq + pid_head * stride_final_k_state_head

        # Create TMA tensor descriptors
        q_desc = tl.make_tensor_descriptor(
            q_ptr,
            shape=[seqlen, headdim_qk],
            strides=[stride_q_seqlen, stride_q_qkdim],
            block_shape=[CHUNK_SIZE, HEADDIM_QK],
        )
        k_desc = tl.make_tensor_descriptor(
            k_ptr,
            shape=[seqlen, headdim_qk],
            strides=[stride_k_seqlen, stride_k_qkdim],
            block_shape=[CHUNK_SIZE, HEADDIM_QK],
        )
        v_desc = tl.make_tensor_descriptor(
            v_ptr,
            shape=[seqlen, headdim_v],
            strides=[stride_v_seqlen, stride_v_vdim],
            block_shape=[CHUNK_SIZE, HEADDIM_V],
        )
        if HAS_Z:
            z_desc = tl.make_tensor_descriptor(
                z_ptr,
                shape=[seqlen, headdim_v],
                strides=[stride_z_seqlen, stride_z_vdim],
                block_shape=[CHUNK_SIZE, HEADDIM_V],
            )

        q_store_desc = tl.make_tensor_descriptor(
            q_store_ptr,
            shape=[seqlen, headdim_qk],
            strides=[stride_q_store_seqlen, stride_q_store_qkdim],
            block_shape=[CHUNK_SIZE, HEADDIM_QK],
        )
        k_store_desc = tl.make_tensor_descriptor(
            k_store_ptr,
            shape=[seqlen, headdim_qk],
            strides=[stride_k_store_seqlen, stride_k_store_qkdim],
            block_shape=[CHUNK_SIZE, HEADDIM_QK],
        )
        o_desc = tl.make_tensor_descriptor(
            o_ptr,
            shape=[seqlen, headdim_v],
            strides=[stride_o_seqlen, stride_o_vdim],
            block_shape=[CHUNK_SIZE, HEADDIM_V],
        )
        if STORE_SSM_STATES_ADT_OUTV:
            ssm_states_desc = tl.make_tensor_descriptor(
                ssm_states_ptr,
                shape=[headdim_v, num_chunks * headdim_qk],
                strides=[stride_ssm_states_vdim, stride_ssm_states_qkdim],
                block_shape=[HEADDIM_V, HEADDIM_QK],
            )

        # Phase 1: Preprocessing - Apply bias, rotary embeddings, compute QK dots.
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)
            offs_hd = tl.arange(0, HEADDIM_QK)
            offs_hdr = tl.arange(0, HEADDIM_QK // 2)

            # Load Q and K blocks via TMA
            q_pre_block = q_desc.load([chunk_start, 0])
            k_pre_block = k_desc.load([chunk_start, 0])

            # Load rotary angles
            angle_block = tl.load(
                angle_ptr
                + offs_seqlen[:, None] * stride_angles_seqlen
                + offs_hdr[None, :] * stride_angles_qkdim
            )

            # Compute shifted gamma and scale
            dt = tl.load(dt_ptr + offs_seqlen * stride_dt_seqlen).to(tl.float32)
            dt_shifted = tl.load(dt_ptr + (offs_seqlen + 1) * stride_dt_seqlen).to(
                tl.float32
            )
            trap = tl.load(trap_ptr + offs_seqlen * stride_trap_seqlen).to(tl.float32)
            trap = tl.sigmoid(trap)
            trap_shifted = tl.load(
                trap_ptr + (offs_seqlen + 1) * stride_trap_seqlen
            ).to(tl.float32)
            trap_shifted = tl.sigmoid(trap_shifted)

            shifted_gamma = dt_shifted * (1 - trap_shifted)
            gamma = dt * trap
            scale = shifted_gamma + gamma

            # Store scale and shifted gamma for backward pass
            tl.store(gamma_store_ptr + offs_seqlen * stride_gamma_store_seqlen, gamma)
            tl.store(scale_store_ptr + offs_seqlen * stride_scale_store_seqlen, scale)

            # Add biases to Q and K
            q_bias_block = tl.load(q_bias_ptr + offs_hd * stride_q_bias_qkdim)
            q_pre_block += q_bias_block[None, :]
            k_bias_block = tl.load(k_bias_ptr + offs_hd * stride_k_bias_qkdim)
            k_pre_block += k_bias_block[None, :]

            # Compute QK dot products for skip connection
            store_qk_dot = tl.dot(
                q_pre_block * k_pre_block,
                tl.full([HEADDIM_QK, 1], 1, dtype=q_pre_block.dtype)
            ).to(q_pre_block.dtype)
            store_qk_dot = store_qk_dot.reshape(CHUNK_SIZE)
            store_qk_dot *= gamma
            tl.store(qk_store_ptr + offs_seqlen * stride_qk_store_seqlen, store_qk_dot)

            # Compute rotary embedding cos/sin
            cos_block = tl.inline_asm_elementwise(
                "cos.approx.f32 $0, $1;",
                constraints="=f,f",
                args=[angle_block.to(tl.float32)],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            sin_block = tl.inline_asm_elementwise(
                "sin.approx.f32 $0, $1;",
                constraints="=f,f",
                args=[angle_block.to(tl.float32)],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

            # Apply rotary embeddings to K and scale
            k0, k1 = tl.split(tl.reshape(k_pre_block, [CHUNK_SIZE, HEADDIM_QK // 2, 2]))
            ko0 = k0 * cos_block - k1 * sin_block
            ko1 = k0 * sin_block + k1 * cos_block
            k_pre_block = tl.reshape(tl.join(ko0, ko1), [CHUNK_SIZE, HEADDIM_QK]).to(k_pre_block.dtype)

            if chunk_idx == num_chunks - 1 and RETURN_FINAL_STATES:
                tl.store(final_k_state_ptr + tl.arange(0, CHUNK_SIZE)[:, None] * stride_final_k_state_chunk
                    + offs_hd[None, :] * stride_final_k_state_qkdim,
                    k_pre_block)

            k_pre_block *= scale[:, None]
            k_store_desc.store([chunk_start, 0], k_pre_block)

            # Apply rotary embeddings to Q
            q0, q1 = tl.split(tl.reshape(q_pre_block, [CHUNK_SIZE, HEADDIM_QK // 2, 2]))
            qo0 = q0 * cos_block - q1 * sin_block
            qo1 = q0 * sin_block + q1 * cos_block
            q_pre_block = tl.reshape(tl.join(qo0, qo1), [CHUNK_SIZE, HEADDIM_QK]).to(q_pre_block.dtype)
            q_store_desc.store([chunk_start, 0], q_pre_block)

        # Phase 2: Main computation and output generation.
        if HAS_INITIAL_STATES:
            acc_ssm_states = tl.load(
                init_ssm_state_ptr
                + tl.arange(0, HEADDIM_V)[:, None] * stride_init_ssm_state_vdim
                + tl.arange(0, HEADDIM_QK)[None, :] * stride_init_ssm_state_qkdim
            ).to(tl.float32)
            input_k_state = tl.load(
                init_k_state_ptr + tl.arange(0, HEADDIM_QK) * stride_init_k_state_qkdim
            ).to(tl.float32)
            input_v_state = tl.load(
                init_v_state_ptr + tl.arange(0, HEADDIM_V) * stride_init_v_state_vdim
            ).to(tl.float32)

            dt_scalar = tl.load(dt_ptr).to(tl.float32)
            trap_scalar = tl.load(trap_ptr).to(tl.float32)
            trap_scalar = tl.sigmoid(trap_scalar)
            # Step on the SSM states with input K/V states to account for trapezoidal discretization
            acc_ssm_states += input_v_state[:, None] * input_k_state[None, :] * dt_scalar * (1 - trap_scalar)
        else:
            acc_ssm_states = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)

        if HAS_D:
            D_val = tl.load(D_ptr).to(tl.float32)
        else:
            D_val = 0.0

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)

            # Load decay factors (log2 scale for exp2 computation)
            adt_ptrs = adt_ptr + offs_seqlen * stride_adt_seqlen
            da = tl.load(adt_ptrs) * 1.44269504089  # log2(e)

            # Load preprocessed Q, K, V blocks
            q_block = q_store_desc.load([chunk_start, 0])
            k_block = k_store_desc.load([chunk_start, 0])
            v_block = v_desc.load([chunk_start, 0])
            if HAS_Z:
                z_block = z_desc.load([chunk_start, 0])

            # Compute cumulative decay for this chunk
            da_cs = tl.cumsum(da)
            da_cs_last = tl.sum(da)
            da_cs_rev = da_cs_last - da_cs

            # Store decay info for backward pass
            if STORE_SSM_STATES_ADT_OUTV:
                tl.store(da_cs_store_ptr + offs_seqlen * stride_da_cs_store_seqlen, da_cs)
                tl.store(da_cs_sum_store_ptr + chunk_idx * stride_da_cs_sum_store_seqlen, da_cs_last)

            # Output contribution from previous state: Q @ SSM_States^T * exp(da_cs)
            acc_o = tl.dot(q_block, tl.trans(acc_ssm_states).to(q_block.dtype))
            acc_o *= tl.math.exp2(da_cs)[:, None]

            # Output contribution from current chunk: causal(Q @ K^T * exp(decay)) @ V
            # NOTE: We compute the (i,i) component using QK dot to prevent non-causal numerical leakage
            s_block = tl.dot(q_block, tl.trans(k_block))
            s_block *= tl.math.exp2(tl.minimum((da_cs[:, None] - da_cs[None, :]), 0.0))
            s_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[:, None] > tl.arange(0, CHUNK_SIZE)[None, :],
                s_block,
                0.0
            )
            acc_o += tl.dot(s_block.to(v_block.dtype), v_block)

            # Add D-skip connection and subtract QK dot contribution
            qk_dot = tl.load(qk_store_ptr + offs_seqlen * stride_qk_store_seqlen)
            acc_o += (D_val + qk_dot)[:, None] * v_block

            if STORE_SSM_STATES_ADT_OUTV:
                tl.store(out_v_ptr + offs_seqlen[:, None] * stride_o_v_seqlen
                    + tl.arange(0, HEADDIM_V)[None, :] * stride_o_v_vdim, acc_o)

            # Apply Z-gating if present
            if HAS_Z:
                z_block = z_block.to(tl.float32)
                acc_o = acc_o * z_block * tl.sigmoid(z_block)

            # Store output
            o_desc.store([chunk_start, 0], acc_o)

            if STORE_SSM_STATES_ADT_OUTV:
                ssm_states_desc.store([0, chunk_idx * headdim_qk], acc_ssm_states.to(ssm_states_desc.dtype))

            # Update recurrent states
            scale = tl.math.exp2(da_cs_rev)
            v_block *= scale[:, None]
            acc_ssm_states = acc_ssm_states * tl.math.exp2(da_cs_last) + tl.dot(
                tl.trans(v_block).to(k_block.dtype), k_block
            )

        # Store final states if requested
        if RETURN_FINAL_STATES:
            tl.store(final_ssm_state_ptr + tl.arange(0, HEADDIM_V)[:, None] * stride_final_ssm_state_vdim
                + tl.arange(0, HEADDIM_QK)[None, :] * stride_final_ssm_state_qkdim,
                acc_ssm_states)


    def get_random_input(self, fixed=False):
        del fixed
        device = "cuda"
        b, h, t, k, v = (
            self.batch_size,
            self.num_heads,
            self.sequence_length,
            self.key_dim,
            self.value_dim,
        )
        return (
            torch.randn(b, h, t, k, device=device, dtype=torch.float16),
            torch.randn(b, h, t, k, device=device, dtype=torch.float16),
            torch.randn(b, h, t, v, device=device, dtype=torch.float16),
            torch.randn(b, h, t, device=device).clamp(max=0),
            torch.rand(b, h, t, device=device),
            torch.randn(b, h, t, device=device),
            torch.randn(h, k, device=device, dtype=torch.float16),
            torch.randn(h, k, device=device, dtype=torch.float16),
            torch.randn(b, h, t, k // 2, device=device),
            torch.ones(h, device=device),
            torch.randn(b, h, t, v, device=device, dtype=torch.float16),
            (
                torch.zeros(b, h, k // 2, device=device),
                torch.zeros(b, h, v, k, device=device),
                torch.zeros(b, h, k, device=device),
                torch.zeros(b, h, v, device=device),
            ),
        )

    def get_shape_information(self):
        return (
            "- q/k: [B, H, T, K]; v/z: [B, H, T, V]; "
            "states: (angle [B,H,K/2], SSM [B,H,V,K], K [B,H,K], V [B,H,V])"
        )

    def forward_triton(self, inputs, ptx=False):
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, states = inputs
        _, ssm_state, k_state, v_state = states
        batch_size, num_heads, sequence_length, key_dim = q.shape
        value_dim = v.shape[-1]
        if sequence_length % 32 or key_dim % 32 or value_dim % 32:
            raise ValueError(
                "MambaICLR2026Forward requires sequence, key, and value dimensions "
                "that are multiples of 32."
            )
        chunk_size = self.chunk_size
        if sequence_length % chunk_size:
            raise ValueError(
                "MambaICLR2026Forward requires sequence_length to be divisible "
                f"by the configured chunk_size ({chunk_size})."
            )
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        dt_padded = functional.pad(dt, (0, 1))
        trap_padded = functional.pad(trap, (0, 1))

        output = torch.empty_like(v)
        out_v = torch.empty_like(v)
        ssm_states = torch.empty(
            batch_size,
            num_heads,
            value_dim,
            num_chunks * key_dim,
            device=v.device,
            dtype=v.dtype,
        )
        da_cs_store = torch.empty_like(adt)
        da_cs_sum_store = torch.empty(
            batch_size,
            num_heads,
            num_chunks,
            device=adt.device,
            dtype=adt.dtype,
        )
        q_store = torch.empty_like(q)
        k_store = torch.empty_like(k)
        qk_store = torch.empty_like(adt)
        scale_store = torch.empty_like(adt)
        gamma_store = torch.empty_like(adt)
        final_ssm_state = torch.empty_like(ssm_state)
        final_k_state = torch.empty(
            batch_size,
            num_heads,
            chunk_size,
            key_dim,
            device=k.device,
            dtype=k.dtype,
        )
        cu_seqlens = torch.arange(
            batch_size + 1,
            device=q.device,
            dtype=torch.int32,
        ) * sequence_length

        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {
                key: value
                for key, value in {
                    "num_warps": self.num_warps,
                    "num_stages": self.num_stages,
                    "maxnreg": self.maxnreg,
                }.items()
                if value is not None
            }
        )
        compiled = launch[(num_heads, batch_size)](
            q,
            k,
            v,
            adt,
            dt_padded,
            trap_padded,
            q_bias,
            k_bias,
            angles,
            d,
            z,
            ssm_state,
            k_state,
            v_state,
            cu_seqlens,
            output,
            out_v,
            ssm_states,
            da_cs_store,
            da_cs_sum_store,
            q_store,
            k_store,
            qk_store,
            scale_store,
            gamma_store,
            final_ssm_state,
            final_k_state,
            stride_q_batch=q.stride(0),
            stride_q_seqlen=q.stride(2),
            stride_q_head=q.stride(1),
            stride_q_qkdim=q.stride(3),
            stride_k_batch=k.stride(0),
            stride_k_seqlen=k.stride(2),
            stride_k_head=k.stride(1),
            stride_k_qkdim=k.stride(3),
            stride_v_batch=v.stride(0),
            stride_v_seqlen=v.stride(2),
            stride_v_head=v.stride(1),
            stride_v_vdim=v.stride(3),
            stride_adt_batch=adt.stride(0),
            stride_adt_head=adt.stride(1),
            stride_adt_seqlen=adt.stride(2),
            stride_dt_batch=dt_padded.stride(0),
            stride_dt_head=dt_padded.stride(1),
            stride_dt_seqlen=dt_padded.stride(2),
            stride_trap_batch=trap_padded.stride(0),
            stride_trap_head=trap_padded.stride(1),
            stride_trap_seqlen=trap_padded.stride(2),
            stride_q_bias_head=q_bias.stride(0),
            stride_q_bias_qkdim=q_bias.stride(1),
            stride_k_bias_head=k_bias.stride(0),
            stride_k_bias_qkdim=k_bias.stride(1),
            stride_angles_batch=angles.stride(0),
            stride_angles_seqlen=angles.stride(2),
            stride_angles_head=angles.stride(1),
            stride_angles_qkdim=angles.stride(3),
            stride_d_head=d.stride(0),
            stride_z_batch=z.stride(0),
            stride_z_seqlen=z.stride(2),
            stride_z_head=z.stride(1),
            stride_z_vdim=z.stride(3),
            stride_init_ssm_state_seq=ssm_state.stride(0),
            stride_init_ssm_state_head=ssm_state.stride(1),
            stride_init_ssm_state_vdim=ssm_state.stride(2),
            stride_init_ssm_state_qkdim=ssm_state.stride(3),
            stride_init_k_state_seq=k_state.stride(0),
            stride_init_k_state_head=k_state.stride(1),
            stride_init_k_state_qkdim=k_state.stride(2),
            stride_init_v_state_seq=v_state.stride(0),
            stride_init_v_state_head=v_state.stride(1),
            stride_init_v_state_vdim=v_state.stride(2),
            stride_cu_seqlen=cu_seqlens.stride(0),
            stride_o_batch=output.stride(0),
            stride_o_seqlen=output.stride(2),
            stride_o_head=output.stride(1),
            stride_o_vdim=output.stride(3),
            stride_o_v_batch=out_v.stride(0),
            stride_o_v_seqlen=out_v.stride(2),
            stride_o_v_head=out_v.stride(1),
            stride_o_v_vdim=out_v.stride(3),
            stride_ssm_states_batch=ssm_states.stride(0),
            stride_ssm_states_head=ssm_states.stride(1),
            stride_ssm_states_vdim=ssm_states.stride(2),
            stride_ssm_states_qkdim=ssm_states.stride(3),
            stride_da_cs_store_batch=da_cs_store.stride(0),
            stride_da_cs_store_head=da_cs_store.stride(1),
            stride_da_cs_store_seqlen=da_cs_store.stride(2),
            stride_da_cs_sum_store_batch=da_cs_sum_store.stride(0),
            stride_da_cs_sum_store_head=da_cs_sum_store.stride(1),
            stride_da_cs_sum_store_seqlen=da_cs_sum_store.stride(2),
            stride_q_store_batch=q_store.stride(0),
            stride_q_store_seqlen=q_store.stride(2),
            stride_q_store_head=q_store.stride(1),
            stride_q_store_qkdim=q_store.stride(3),
            stride_k_store_batch=k_store.stride(0),
            stride_k_store_seqlen=k_store.stride(2),
            stride_k_store_head=k_store.stride(1),
            stride_k_store_qkdim=k_store.stride(3),
            stride_qk_store_batch=qk_store.stride(0),
            stride_qk_store_head=qk_store.stride(1),
            stride_qk_store_seqlen=qk_store.stride(2),
            stride_scale_store_batch=scale_store.stride(0),
            stride_scale_store_head=scale_store.stride(1),
            stride_scale_store_seqlen=scale_store.stride(2),
            stride_gamma_store_batch=gamma_store.stride(0),
            stride_gamma_store_head=gamma_store.stride(1),
            stride_gamma_store_seqlen=gamma_store.stride(2),
            stride_final_ssm_state_seq=final_ssm_state.stride(0),
            stride_final_ssm_state_head=final_ssm_state.stride(1),
            stride_final_ssm_state_vdim=final_ssm_state.stride(2),
            stride_final_ssm_state_qkdim=final_ssm_state.stride(3),
            stride_final_k_state_seq=final_k_state.stride(0),
            stride_final_k_state_head=final_k_state.stride(1),
            stride_final_k_state_chunk=final_k_state.stride(2),
            stride_final_k_state_qkdim=final_k_state.stride(3),
            seqlen=sequence_length,
            nheads_qk=k.shape[1],
            headdim_qk=key_dim,
            headdim_v=value_dim,
            headdim_angles=angles.shape[-1],
            CHUNK_SIZE=chunk_size,
            HEADDIM_QK=key_dim,
            HEADDIM_V=value_dim,
            STORE_SSM_STATES_ADT_OUTV=True,
            HAS_INITIAL_STATES=True,
            RETURN_FINAL_STATES=True,
            HAS_D=True,
            HAS_Z=True,
            IS_VARLEN=False,
            **kwargs,
        )
        final_k = final_k_state[:, :, (sequence_length - 1) % chunk_size]
        return (output, (final_ssm_state, final_k)), compiled

    def forward_torch(self, inputs):
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, states = inputs
        _, state, previous_k, previous_v = (item.float().clone() for item in states)
        q_pre = q + q_bias[None, :, None]
        k_pre = k + k_bias[None, :, None]
        trap_values = trap.float().sigmoid()
        gamma = dt.float() * trap_values
        shifted_gamma = functional.pad(
            dt[:, :, 1:].float() * (1.0 - trap_values[:, :, 1:]),
            (0, 1),
        )
        scale = gamma + shifted_gamma
        qk_reduction = torch.ones(q.shape[-1], device=q.device, dtype=q.dtype)
        qk_dot = torch.matmul(q_pre * k_pre, qk_reduction).float() * gamma
        q_value = _rotate_torch(q_pre, angles.float()).to(q.dtype)
        k_value = _rotate_torch(k_pre, angles.float()).to(k.dtype)
        k_scaled = (k_value.float() * scale[..., None]).to(k.dtype)
        state += (
            previous_v[..., :, None]
            * previous_k[..., None, :]
            * dt[:, :, 0, None, None].float()
            * (1.0 - trap_values[:, :, 0, None, None])
        )
        output = torch.empty_like(v)
        for start in range(0, q.shape[2], self.chunk_size):
            stop = start + self.chunk_size
            q_block = q_value[:, :, start:stop]
            k_block = k_scaled[:, :, start:stop]
            v_block = v[:, :, start:stop]
            decay = torch.cumsum(
                adt[:, :, start:stop].float() * 1.44269504089,
                dim=-1,
            )
            output_block = torch.einsum(
                "bhtk,bhvk->bhtv",
                q_block.float(),
                state.to(q.dtype).float(),
            ) * torch.exp2(decay)[..., None]
            scores = torch.einsum("bhtk,bhsk->bhts", q_block.float(), k_block.float())
            scores *= torch.exp2(
                (decay[..., :, None] - decay[..., None, :]).clamp(max=0.0)
            )
            scores = torch.tril(scores, diagonal=-1).to(v.dtype)
            output_block += torch.einsum(
                "bhts,bhsv->bhtv",
                scores.float(),
                v_block.float(),
            )
            output_block += (
                d[None, :, None, None] + qk_dot[:, :, start:stop, None]
            ) * v_block.float()
            output[:, :, start:stop] = (
                output_block * functional.silu(z[:, :, start:stop].float())
            ).to(v.dtype)
            reverse_decay = decay[..., -1:] - decay
            state = state * torch.exp2(decay[..., -1, None, None])
            state += torch.einsum(
                "bhtv,bhtk->bhvk",
                (v_block.float() * torch.exp2(reverse_decay)[..., None]).to(v.dtype).float(),
                k_block.float(),
            )
        return output, (state, k_value[:, :, -1])


class MambaICLR2026Step(TritonPTXKernel):
    """One-token Mamba-3 SISO decode recurrence."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "num_warps": (2, 4, 8),
        "num_stages": (1, 2, 3),
    }
    verification_tolerance = 5e-2
    autotune_tolerance = 5e-2

    def __init__(self, *, ptx=None):
        self.batch_size, self.sequence_length = 64, 2048
        self.num_heads, self.key_dim, self.value_dim = 16, 64, 64
        self.num_warps, self.num_stages = 4, 1
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        # Inputs
        Q_ptr, K_ptr, V_ptr, ADT_ptr, DT_ptr, Trap_ptr, Q_bias_ptr, K_bias_ptr,
        Angles_ptr, D_input_ptr, Z_ptr, Input_Angle_State_ptr,
        Input_SSM_State_ptr, Input_K_State_ptr, Input_V_State_ptr,
        # Outputs
        Out_ptr, Output_Angle_State_ptr, Output_SSM_State_ptr,
        Output_K_State_ptr,
        # Input Strides
        stride_q_batch: tl.constexpr, stride_q_head: tl.constexpr,
        stride_q_qkdim: tl.constexpr, stride_k_batch: tl.constexpr,
        stride_k_head: tl.constexpr, stride_k_qkdim: tl.constexpr,
        stride_v_batch: tl.constexpr, stride_v_head: tl.constexpr,
        stride_v_vdim: tl.constexpr, stride_adt_batch: tl.constexpr,
        stride_adt_head: tl.constexpr, stride_dt_batch: tl.constexpr,
        stride_dt_head: tl.constexpr, stride_trap_batch: tl.constexpr,
        stride_trap_head: tl.constexpr, stride_q_bias_head: tl.constexpr,
        stride_q_bias_qkdim: tl.constexpr, stride_k_bias_head: tl.constexpr,
        stride_k_bias_qkdim: tl.constexpr, stride_angles_batch: tl.constexpr,
        stride_angles_head: tl.constexpr, stride_angles_qkdim: tl.constexpr,
        stride_d_head: tl.constexpr, stride_z_batch: tl.constexpr,
        stride_z_head: tl.constexpr, stride_z_vdim: tl.constexpr,
        stride_angle_state_batch: tl.constexpr,
        stride_angle_state_head: tl.constexpr,
        stride_angle_state_anglesdim: tl.constexpr,
        stride_input_ssm_state_batch: tl.constexpr,
        stride_input_ssm_state_head: tl.constexpr,
        stride_input_ssm_state_vdim: tl.constexpr,
        stride_input_ssm_state_qkdim: tl.constexpr,
        stride_input_k_state_batch: tl.constexpr,
        stride_input_k_state_head: tl.constexpr,
        stride_input_k_state_qkdim: tl.constexpr,
        stride_input_v_state_batch: tl.constexpr,
        stride_input_v_state_head: tl.constexpr,
        stride_input_v_state_vdim: tl.constexpr,
        # Output Strides
        stride_o_batch: tl.constexpr, stride_o_head: tl.constexpr,
        stride_o_vdim: tl.constexpr,
        stride_output_angle_state_batch: tl.constexpr,
        stride_output_angle_state_head: tl.constexpr,
        stride_output_angle_state_anglesdim: tl.constexpr,
        stride_output_ssm_state_batch: tl.constexpr,
        stride_output_ssm_state_head: tl.constexpr,
        stride_output_ssm_state_vdim: tl.constexpr,
        stride_output_ssm_state_qkdim: tl.constexpr,
        stride_output_k_state_batch: tl.constexpr,
        stride_output_k_state_head: tl.constexpr,
        stride_output_k_state_qkdim: tl.constexpr,
        # Dimensions
        nheads_qk: tl.constexpr,
        HEADDIM_QK: tl.constexpr,
        HEADDIM_V: tl.constexpr,
        HEADDIM_ANGLES: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_Z: tl.constexpr,
    ):
        """
        Mamba-3 Step kernel.

        Inputs:
            Q, K:                       (batch, nheads_qk, headdim_qk)
            V:                          (batch, nheads, headdim_v)  
            ADT, DT, Trap:              (batch, nheads)
            Q_bias, K_bias:             (nheads, headdim_qk)
            Angles:                     (batch, nheads, headdim_angles)
            D:                          (nheads,)
            Z:                          (batch, nheads, headdim_v)
            Out:                        (batch, nheads, headdim_v)
            SSM_States:                 (batch, nheads, headdim_v, headdim_qk)
            Input/Output Angle State:   (batch, nheads, headdim_angles)
            Input/Output SSM State:     (batch, nheads, headdim_v, headdim_qk)
            Input/Output K State:       (batch, nheads, headdim_qk)
            Input/Output V State:       (batch, nheads, headdim_v)

        Compile-time constants:
            HEADDIM_QK:                 Head dimension for Q/K
            HEADDIM_V:                  Head dimension for V
            HEADDIM_ANGLES:             Head dimension for Angles
            HAS_D:                      Whether D-skip connection is used
            HAS_Z:                      Whether Z-gating is used

        Outputs:
            Out:                    (batch, nheads, headdim_v)
            Output_Angle_State:     (batch, nheads, headdim_angles)
            Output_SSM_State:       (batch, nheads, headdim_v, headdim_qk)
            Output_K_State:         (batch, nheads, headdim_qk)
        """
        # Program ID determines which (head, batch) pair this instance processes
        pid_head = tl.program_id(0)
        pid_batch = tl.program_id(1)

        # Compute head index for Q/K (supports Grouped Query Attention)
        nheads = tl.num_programs(0)
        head_idx_qk = pid_head // (nheads // nheads_qk)

        # Setup input pointers
        q_ptr = Q_ptr + pid_batch * stride_q_batch + head_idx_qk * stride_q_head
        k_ptr = K_ptr + pid_batch * stride_k_batch + head_idx_qk * stride_k_head
        v_ptr = V_ptr + pid_batch * stride_v_batch + pid_head * stride_v_head
        adt_ptr = ADT_ptr + pid_batch * stride_adt_batch + pid_head * stride_adt_head
        dt_ptr = DT_ptr + pid_batch * stride_dt_batch + pid_head * stride_dt_head
        trap_ptr = Trap_ptr + pid_batch * stride_trap_batch + pid_head * stride_trap_head
        q_bias_ptr = Q_bias_ptr + pid_head * stride_q_bias_head
        k_bias_ptr = K_bias_ptr + pid_head * stride_k_bias_head
        angle_ptr = Angles_ptr + pid_batch * stride_angles_batch + pid_head * stride_angles_head
        if HAS_D:
            D_ptr = D_input_ptr + pid_head * stride_d_head
            D_val = tl.load(D_ptr).to(tl.float32)
        if HAS_Z:
            z_ptr = Z_ptr + pid_batch * stride_z_batch + pid_head * stride_z_head
        input_angle_state_ptr = Input_Angle_State_ptr + pid_batch * stride_angle_state_batch + pid_head * stride_angle_state_head
        input_ssm_state_ptr = Input_SSM_State_ptr + pid_batch * stride_input_ssm_state_batch + pid_head * stride_input_ssm_state_head
        input_k_state_ptr = Input_K_State_ptr + pid_batch * stride_input_k_state_batch + pid_head * stride_input_k_state_head
        input_v_state_ptr = Input_V_State_ptr + pid_batch * stride_input_v_state_batch + pid_head * stride_input_v_state_head

        # Setup output pointers
        o_ptr = Out_ptr + pid_batch * stride_o_batch + pid_head * stride_o_head
        output_angle_state_ptr = Output_Angle_State_ptr + pid_batch * stride_output_angle_state_batch + pid_head * stride_output_angle_state_head
        output_ssm_state_ptr = Output_SSM_State_ptr + pid_batch * stride_output_ssm_state_batch + pid_head * stride_output_ssm_state_head
        output_k_state_ptr = Output_K_State_ptr + pid_batch * stride_output_k_state_batch + pid_head * stride_output_k_state_head

        PI = 3.141592653589793
        TWO_PI = 2 * PI
        offs_qk = tl.arange(0, HEADDIM_QK)
        offs_v = tl.arange(0, HEADDIM_V)
        offs_qkr = tl.arange(0, HEADDIM_QK // 2)

        # Load Q and K blocks
        q_pre_block = tl.load(q_ptr + offs_qk * stride_q_qkdim) # (HEADDIM_QK)
        k_pre_block = tl.load(k_ptr + offs_qk * stride_k_qkdim) # (HEADDIM_QK)

        # Load Q and K biases
        q_bias_block = tl.load(q_bias_ptr + offs_qk * stride_q_bias_qkdim) # (HEADDIM_QK)
        k_bias_block = tl.load(k_bias_ptr + offs_qk * stride_k_bias_qkdim) # (HEADDIM_QK)

        q_pre_block += q_bias_block
        k_pre_block += k_bias_block

        # Load rotary angles (smaller block, direct load is faster than TMA)
        dt = tl.load(dt_ptr)
        angle_block = tl.load(
            angle_ptr + offs_qkr * stride_angles_qkdim, mask=offs_qkr < HEADDIM_ANGLES, other=0.0
        ) # (HEADDIM_QK)
        angle_block = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[angle_block.to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        ) * PI * dt
        angle_state = tl.load(
            input_angle_state_ptr + offs_qkr * stride_angle_state_anglesdim, mask=offs_qkr < HEADDIM_ANGLES, other=0.0
        ) # (HEADDIM_QK)

        angle_block += angle_state
        angle_block -= TWO_PI * tl.floor(angle_block / TWO_PI)
        # angles mod 2pi

        tl.store(output_angle_state_ptr + offs_qkr * stride_output_angle_state_anglesdim, angle_block, mask=offs_qkr < HEADDIM_ANGLES)

        # Rotate Q and K with angles
        cos_block = tl.inline_asm_elementwise(
            "cos.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[angle_block.to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        sin_block = tl.inline_asm_elementwise(
            "sin.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[angle_block.to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

        # Apply rotary embeddings to K and scale
        q0, q1 = tl.split(tl.reshape(q_pre_block, [HEADDIM_QK // 2, 2]))
        qo0 = q0 * cos_block - q1 * sin_block
        qo1 = q0 * sin_block + q1 * cos_block
        q_block = tl.reshape(tl.join(qo0, qo1), [HEADDIM_QK]).to(q_pre_block.dtype)

        k0, k1 = tl.split(tl.reshape(k_pre_block, [HEADDIM_QK // 2, 2]))
        ko0 = k0 * cos_block - k1 * sin_block
        ko1 = k0 * sin_block + k1 * cos_block
        k_block = tl.reshape(tl.join(ko0, ko1), [HEADDIM_QK]).to(k_pre_block.dtype)

        # Store K state
        tl.store(output_k_state_ptr + offs_qk * stride_output_k_state_qkdim, k_block)

        # Load previous K, V and current V
        k_prev_state = tl.load(input_k_state_ptr + offs_qk * stride_input_k_state_qkdim) # (HEADDIM_QK)
        v_prev_state = tl.load(input_v_state_ptr + offs_v * stride_input_v_state_vdim) # (HEADDIM_V)
        v_block = tl.load(v_ptr + offs_v * stride_v_vdim) # (HEADDIM_V)
            
        # Load ADT, DT and Trap
        adt = tl.load(adt_ptr) * 1.44269504089
        trap = tl.load(trap_ptr)
        trap = tl.sigmoid(trap.to(tl.float32))

        alpha = tl.math.exp2(adt)
        beta = alpha * dt * (1 - trap)
        gamma = trap * dt

        ssm_state_diff = (beta * v_prev_state)[:, None] * k_prev_state[None, :] + (gamma * v_block)[:, None] * k_block[None, :]

        # Load previous SSM state
        ssm_state = tl.load(
            input_ssm_state_ptr + offs_v[:, None] * stride_input_ssm_state_vdim 
            + offs_qk[None, :] * stride_input_ssm_state_qkdim).to(tl.float32) # (HEADDIM_V, HEADDIM_QK)
        
        ssm_state = ssm_state * alpha + ssm_state_diff

        # Store updated SSM state
        tl.store(output_ssm_state_ptr + offs_v[:, None] * stride_output_ssm_state_vdim 
            + offs_qk[None, :] * stride_output_ssm_state_qkdim, ssm_state)

        # Compute output
        out = tl.dot(ssm_state.to(tl.bfloat16), q_block.reshape([HEADDIM_QK, 1]).to(tl.bfloat16)) # (HEADDIM_V, 1)
        out = out.reshape([HEADDIM_V]).to(tl.float32)

        # out = tl.sum(ssm_state * q_block[None, :], axis=1)  # (HEADDIM_V,)

        # Add D-skip connection
        if HAS_D:
            out += D_val * v_block

        # Apply Z-gating
        if HAS_Z:
            z_block = tl.load(z_ptr + offs_v * stride_z_vdim) # (HEADDIM_V)
            z_block = z_block.to(tl.float32)
            out = out * z_block * tl.sigmoid(z_block)
        
        # Store output
        tl.store(o_ptr + offs_v * stride_o_vdim, out)


    def get_random_input(self, fixed=False):
        del fixed
        device = "cuda"
        b, h, k, v = self.batch_size, self.num_heads, self.key_dim, self.value_dim
        return (
            torch.randn(b, h, k, device=device, dtype=torch.float16),
            torch.randn(b, h, k, device=device, dtype=torch.float16),
            torch.randn(b, h, v, device=device, dtype=torch.float16),
            torch.randn(b, h, device=device).clamp(max=0),
            torch.rand(b, h, device=device),
            torch.randn(b, h, device=device),
            torch.randn(h, k, device=device, dtype=torch.float16),
            torch.randn(h, k, device=device, dtype=torch.float16),
            torch.randn(b, h, k // 2, device=device),
            torch.ones(h, device=device),
            torch.randn(b, h, v, device=device, dtype=torch.float16),
            (
                torch.zeros(b, h, k // 2, device=device),
                torch.zeros(b, h, v, k, device=device),
                torch.zeros(b, h, k, device=device),
                torch.zeros(b, h, v, device=device),
            ),
        )

    def get_shape_information(self):
        return "- q/k: [B, H, K]; v/z: [B, H, V]; states: (angle, SSM, K, V)"

    def forward_triton(self, inputs, ptx=False):
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, states = inputs
        angle_state, ssm_state, k_state, v_state = states
        batch_size, num_heads, key_dim = q.shape
        value_dim = v.shape[-1]

        output = torch.empty_like(v)
        output_angle_state = torch.empty_like(angle_state)
        output_ssm_state = torch.empty_like(ssm_state)
        output_k_state = torch.empty_like(k_state)
        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {
                "num_warps": self.num_warps,
                "num_stages": self.num_stages,
            }
        )
        compiled = launch[(num_heads, batch_size)](
            q,
            k,
            v,
            adt,
            dt,
            trap,
            q_bias,
            k_bias,
            angles,
            d,
            z,
            angle_state,
            ssm_state,
            k_state,
            v_state,
            output,
            output_angle_state,
            output_ssm_state,
            output_k_state,
            stride_q_batch=q.stride(0),
            stride_q_head=q.stride(1),
            stride_q_qkdim=q.stride(2),
            stride_k_batch=k.stride(0),
            stride_k_head=k.stride(1),
            stride_k_qkdim=k.stride(2),
            stride_v_batch=v.stride(0),
            stride_v_head=v.stride(1),
            stride_v_vdim=v.stride(2),
            stride_adt_batch=adt.stride(0),
            stride_adt_head=adt.stride(1),
            stride_dt_batch=dt.stride(0),
            stride_dt_head=dt.stride(1),
            stride_trap_batch=trap.stride(0),
            stride_trap_head=trap.stride(1),
            stride_q_bias_head=q_bias.stride(0),
            stride_q_bias_qkdim=q_bias.stride(1),
            stride_k_bias_head=k_bias.stride(0),
            stride_k_bias_qkdim=k_bias.stride(1),
            stride_angles_batch=angles.stride(0),
            stride_angles_head=angles.stride(1),
            stride_angles_qkdim=angles.stride(2),
            stride_d_head=d.stride(0),
            stride_z_batch=z.stride(0),
            stride_z_head=z.stride(1),
            stride_z_vdim=z.stride(2),
            stride_angle_state_batch=angle_state.stride(0),
            stride_angle_state_head=angle_state.stride(1),
            stride_angle_state_anglesdim=angle_state.stride(2),
            stride_input_ssm_state_batch=ssm_state.stride(0),
            stride_input_ssm_state_head=ssm_state.stride(1),
            stride_input_ssm_state_vdim=ssm_state.stride(2),
            stride_input_ssm_state_qkdim=ssm_state.stride(3),
            stride_input_k_state_batch=k_state.stride(0),
            stride_input_k_state_head=k_state.stride(1),
            stride_input_k_state_qkdim=k_state.stride(2),
            stride_input_v_state_batch=v_state.stride(0),
            stride_input_v_state_head=v_state.stride(1),
            stride_input_v_state_vdim=v_state.stride(2),
            stride_o_batch=output.stride(0),
            stride_o_head=output.stride(1),
            stride_o_vdim=output.stride(2),
            stride_output_angle_state_batch=output_angle_state.stride(0),
            stride_output_angle_state_head=output_angle_state.stride(1),
            stride_output_angle_state_anglesdim=output_angle_state.stride(2),
            stride_output_ssm_state_batch=output_ssm_state.stride(0),
            stride_output_ssm_state_head=output_ssm_state.stride(1),
            stride_output_ssm_state_vdim=output_ssm_state.stride(2),
            stride_output_ssm_state_qkdim=output_ssm_state.stride(3),
            stride_output_k_state_batch=output_k_state.stride(0),
            stride_output_k_state_head=output_k_state.stride(1),
            stride_output_k_state_qkdim=output_k_state.stride(2),
            nheads_qk=k.shape[1],
            HEADDIM_QK=key_dim,
            HEADDIM_V=value_dim,
            HEADDIM_ANGLES=angles.shape[-1],
            HAS_D=True,
            HAS_Z=True,
            **kwargs,
        )
        return (output, (output_angle_state, output_ssm_state, output_k_state)), compiled

    def forward_torch(self, inputs):
        q, k, v, adt, dt, trap, q_bias, k_bias, delta, d, z, states = inputs
        angle, state, previous_k, previous_v = (item.float().clone() for item in states)
        angle = (angle + delta.float().tanh() * torch.pi * dt[..., None]) % (
            2 * torch.pi
        )
        q_value = _rotate_torch(q + q_bias, angle).to(q.dtype)
        k_value = _rotate_torch(k + k_bias, angle).to(k.dtype)
        value = v.float()
        alpha = torch.exp2(adt.float() * 1.44269504089)
        trap_value = trap.float().sigmoid()
        state = state * alpha[..., None, None]
        state += (
            (alpha * dt * (1 - trap_value))[..., None, None]
            * previous_v[..., :, None]
            * previous_k[..., None, :]
        )
        state += (
            (dt * trap_value)[..., None, None]
            * value[..., :, None]
            * k_value[..., None, :]
        )
        output = (
            state.to(torch.bfloat16).float()
            * q_value.to(torch.bfloat16).float()[..., None, :]
        ).sum(-1) + d[None, :, None] * value
        return (output * functional.silu(z.float())).to(v.dtype), (
            angle,
            state,
            k_value.float(),
        )
