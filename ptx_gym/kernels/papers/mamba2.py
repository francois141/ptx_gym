"""Mamba-2 chunk-state forward kernel.

Adapted from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_state.py
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from torch.nn import functional
from ptx_gym.kernels.base import TritonPTXKernel


class Mamba2ChunkStateForward(TritonPTXKernel):
    """Compute Mamba-2 SSM states independently for every sequence chunk."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_size_m": (32, 64),
        "block_size_n": (32, 64, 128),
        "block_size_k": (32, 64),
        "num_warps": (2, 4, 8),
        "num_stages": (3, 4, 5),
    }

    verification_tolerance = 2e-2

    def __init__(self, *, ptx=None, has_seq_idx: bool = False):
        self.batch_size = 64
        self.sequence_length = 2048
        self.num_heads = 8
        self.num_groups = 4
        self.head_dim = 64
        self.state_dim = 128
        self.chunk_size = 256
        self.has_seq_idx = has_seq_idx
        self.block_size_m = 32
        self.block_size_n = 32
        self.block_size_k = 32
        self.num_warps = 4
        self.num_stages = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        # Pointers to matrices
        x_ptr,
        b_ptr,
        states_ptr,
        dt_ptr,
        dA_cumsum_ptr,
        seq_idx_ptr,
        # Matrix dimensions
        hdim: tl.constexpr,
        dstate: tl.constexpr,
        chunk_size: tl.constexpr,
        batch: tl.constexpr,
        seqlen: tl.constexpr,
        nheads_ngroups_ratio: tl.constexpr,
        # Strides
        stride_x_batch: tl.constexpr,
        stride_x_seqlen: tl.constexpr,
        stride_x_head: tl.constexpr,
        stride_x_hdim: tl.constexpr,
        stride_b_batch: tl.constexpr,
        stride_b_seqlen: tl.constexpr,
        stride_b_head: tl.constexpr,
        stride_b_dstate: tl.constexpr,
        stride_states_batch: tl.constexpr,
        stride_states_chunk: tl.constexpr,
        stride_states_head: tl.constexpr,
        stride_states_hdim: tl.constexpr,
        stride_states_dstate: tl.constexpr,
        stride_dt_batch: tl.constexpr,
        stride_dt_chunk: tl.constexpr,
        stride_dt_head: tl.constexpr,
        stride_dt_csize: tl.constexpr,
        stride_dA_cs_batch: tl.constexpr,
        stride_dA_cs_chunk: tl.constexpr,
        stride_dA_cs_head: tl.constexpr,
        stride_dA_cs_csize: tl.constexpr,
        stride_seq_idx_batch: tl.constexpr,
        stride_seq_idx_seqlen: tl.constexpr,
        # Meta-parameters
        HAS_SEQ_IDX: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        tl.static_assert(hdim % BLOCK_SIZE_M == 0)
        tl.static_assert(dstate % BLOCK_SIZE_N == 0)
        tl.static_assert(chunk_size % BLOCK_SIZE_K == 0)
        tl.static_assert(seqlen % chunk_size == 0)

        pid_bc = tl.program_id(axis=1)
        pid_c = pid_bc // batch
        pid_b = pid_bc - pid_c * batch
        pid_h = tl.program_id(axis=2)
        num_pid_n = dstate // BLOCK_SIZE_N
        pid_m = tl.program_id(axis=0) // num_pid_n
        pid_n = tl.program_id(axis=0) % num_pid_n
        b_ptr += (
            pid_b * stride_b_batch
            + pid_c * chunk_size * stride_b_seqlen
            + (pid_h // nheads_ngroups_ratio) * stride_b_head
        )
        x_ptr += (
            pid_b * stride_x_batch
            + pid_c * chunk_size * stride_x_seqlen
            + pid_h * stride_x_head
        )
        dt_ptr += (
            pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
        )
        dA_cumsum_ptr += (
            pid_b * stride_dA_cs_batch
            + pid_c * stride_dA_cs_chunk
            + pid_h * stride_dA_cs_head
        )
        if HAS_SEQ_IDX:
            seq_idx_ptr += (
                pid_b * stride_seq_idx_batch
                + pid_c * chunk_size * stride_seq_idx_seqlen
            )

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (
            offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
        )
        b_ptrs = b_ptr + (
            offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
        )
        dt_ptrs = dt_ptr + offs_k * stride_dt_csize
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
            tl.float32
        )
        dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

        if HAS_SEQ_IDX:
            seq_idx_last = tl.load(
                seq_idx_ptr + (chunk_size - 1) * stride_seq_idx_seqlen
            )

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, chunk_size, BLOCK_SIZE_K):
            x = tl.load(x_ptrs)
            b = tl.load(b_ptrs).to(tl.float32)
            dA_cs_k = tl.load(dA_cumsum_ptrs).to(tl.float32)
            if HAS_SEQ_IDX:
                seq_idx_k = tl.load(seq_idx_ptrs)
            dt_k = tl.load(dt_ptrs).to(tl.float32)
            if not HAS_SEQ_IDX:
                # scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
                scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k
            else:
                # scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
                scale = tl.where(
                    (seq_idx_last >= 0) & (seq_idx_k == seq_idx_last),
                    tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k,
                    0.0,
                )
            b *= scale[:, None]
            b = b.to(x_ptr.dtype.element_ty)
            acc += tl.dot(x, b)
            x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
            b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
            dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
            dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
            if HAS_SEQ_IDX:
                seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
        states = acc.to(states_ptr.dtype.element_ty)

        states_ptr += (
            pid_b * stride_states_batch
            + pid_c * stride_states_chunk
            + pid_h * stride_states_head
        )
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        states_ptrs = states_ptr + (
            offs_m[:, None] * stride_states_hdim
            + offs_n[None, :] * stride_states_dstate
        )
        tl.store(states_ptrs, states)

    def get_random_input(self, fixed: bool = False):
        del fixed
        device = "cuda"
        chunks = triton.cdiv(self.sequence_length, self.chunk_size)
        b = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.num_groups,
            self.state_dim,
            device=device,
            dtype=torch.float16,
        )
        x = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
            device=device,
            dtype=torch.float16,
        )
        dt = torch.rand(
            self.batch_size,
            self.num_heads,
            chunks,
            self.chunk_size,
            device=device,
        )
        dA_cumsum = -torch.rand_like(dt).cumsum(dim=-1)
        seq_idx = (
            torch.arange(self.sequence_length, device=device, dtype=torch.int32)
            .div(32, rounding_mode="floor")
            .expand(self.batch_size, -1)
        )
        return b, x, dt, dA_cumsum, seq_idx

    def get_shape_information(self) -> str:
        chunks = triton.cdiv(self.sequence_length, self.chunk_size)
        return (
            f"- b_ptr: float16 tensor with shape ({self.batch_size}, "
            f"{self.sequence_length}, {self.num_groups}, {self.state_dim})\n"
            f"- x_ptr: float16 tensor with shape ({self.batch_size}, "
            f"{self.sequence_length}, {self.num_heads}, {self.head_dim})\n"
            f"- dt_ptr, dA_cumsum_ptr: float32 tensors with shape "
            f"({self.batch_size}, {self.num_heads}, {chunks}, {self.chunk_size})\n"
            f"- seq_idx_ptr: int32 tensor with shape ({self.batch_size}, "
            f"{self.sequence_length})"
        )

    def forward_triton(self, inputs, ptx: bool = False):
        b, x, dt, dA_cumsum, seq_idx = inputs
        batch_size, sequence_length, num_heads, head_dim = x.shape
        num_chunks = dt.shape[2]
        states = torch.empty(
            batch_size,
            num_chunks,
            num_heads,
            head_dim,
            b.shape[-1],
            device=x.device,
            dtype=torch.float32,
        )
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        grid = lambda meta: (
            triton.cdiv(head_dim, meta["BLOCK_SIZE_M"])
            * triton.cdiv(b.shape[-1], meta["BLOCK_SIZE_N"]),
            batch_size * num_chunks,
            num_heads,
        )
        kernel = launch_kernel[grid](
            x,
            b,
            states,
            dt,
            dA_cumsum,
            seq_idx,
            head_dim,
            b.shape[-1],
            self.chunk_size,
            batch_size,
            sequence_length,
            num_heads // b.shape[2],
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            seq_idx.stride(0),
            seq_idx.stride(1),
            HAS_SEQ_IDX=False,
            BLOCK_SIZE_M=self.block_size_m,
            BLOCK_SIZE_N=self.block_size_n,
            BLOCK_SIZE_K=self.block_size_k,
            **launch_kwargs,
        )
        return states, kernel

    def forward_torch(self, inputs):
        b, x, dt, dA_cumsum, seq_idx = inputs
        batch_size, sequence_length, num_heads, head_dim = x.shape
        num_chunks, chunk_size = dt.shape[2:]
        padded_length = num_chunks * chunk_size
        padding = padded_length - sequence_length
        x = functional.pad(x, (0, 0, 0, 0, 0, padding))
        b = functional.pad(b, (0, 0, 0, 0, 0, padding))
        seq_idx = functional.pad(seq_idx, (0, padding), value=-1)
        x = x.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)
        b = b.repeat_interleave(num_heads // b.shape[2], dim=2)
        b = b.reshape(batch_size, num_chunks, chunk_size, num_heads, -1)
        scale = torch.exp((dA_cumsum[..., -1:] - dA_cumsum).clamp(max=0.0)) * dt
        if self.has_seq_idx:
            seq_idx = seq_idx.reshape(batch_size, num_chunks, chunk_size)
            scale *= (seq_idx == seq_idx[..., -1:]).unsqueeze(1)
        return torch.einsum("bclhm,bclhn,bhcl->bchmn", x.float(), b.float(), scale)
