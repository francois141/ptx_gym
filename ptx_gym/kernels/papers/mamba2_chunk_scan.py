"""Runnable Mamba-2 chunk-scan forward operator."""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class Mamba2ChunkScanForward(TritonPTXKernel):
    """Run Mamba-2's chunk scan with state, residual, and gate paths."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_size_m": (32, 64, 128),
        "block_size_n": (32, 64, 128),
        "block_size_k": (32, 64),
        "num_warps": (2, 4, 8),
        "num_stages": (3, 4, 5),
    }
    
    verification_tolerance = 3e-2

    def __init__(self, *, ptx=None):
        self.batch_size, self.sequence_length = 4, 2048
        self.num_heads, self.num_groups = 8, 2
        self.head_dim, self.state_dim, self.chunk_size = 128, 32, 128
        self.block_size_m, self.block_size_n, self.block_size_k = 64, 64, 32
        self.num_warps, self.num_stages = 4, 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        # Pointers to matrices
        cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr,
        # Matrix dimensions
        chunk_size: tl.constexpr, hdim: tl.constexpr, dstate: tl.constexpr,
        batch: tl.constexpr, seqlen: tl.constexpr,
        nheads_ngroups_ratio: tl.constexpr,
        # Strides
        stride_cb_batch: tl.constexpr, stride_cb_chunk: tl.constexpr,
        stride_cb_head: tl.constexpr, stride_cb_csize_m: tl.constexpr,
        stride_cb_csize_k: tl.constexpr,
        stride_x_batch: tl.constexpr, stride_x_seqlen: tl.constexpr,
        stride_x_head: tl.constexpr, stride_x_hdim: tl.constexpr,
        stride_z_batch: tl.constexpr, stride_z_seqlen: tl.constexpr,
        stride_z_head: tl.constexpr, stride_z_hdim: tl.constexpr,
        stride_out_batch: tl.constexpr, stride_out_seqlen: tl.constexpr,
        stride_out_head: tl.constexpr, stride_out_hdim: tl.constexpr,
        stride_dt_batch: tl.constexpr, stride_dt_chunk: tl.constexpr,
        stride_dt_head: tl.constexpr, stride_dt_csize: tl.constexpr,
        stride_dA_cs_batch: tl.constexpr, stride_dA_cs_chunk: tl.constexpr,
        stride_dA_cs_head: tl.constexpr, stride_dA_cs_csize: tl.constexpr,
        stride_seq_idx_batch: tl.constexpr, stride_seq_idx_seqlen: tl.constexpr,
        stride_C_batch: tl.constexpr, stride_C_seqlen: tl.constexpr,
        stride_C_head: tl.constexpr, stride_C_dstate: tl.constexpr,
        stride_states_batch: tl.constexpr, stride_states_chunk: tl.constexpr,
        stride_states_head: tl.constexpr, stride_states_hdim: tl.constexpr,
        stride_states_dstate: tl.constexpr,
        stride_D_head: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_DSTATE: tl.constexpr,
        IS_TRITON_22: tl.constexpr,
    ):
        tl.static_assert(seqlen % chunk_size == 0)
        tl.static_assert(chunk_size % BLOCK_SIZE_M == 0)
        tl.static_assert(chunk_size % BLOCK_SIZE_K == 0)
        tl.static_assert(hdim % BLOCK_SIZE_N == 0)
        tl.static_assert(dstate == BLOCK_SIZE_DSTATE)

        pid_bc = tl.program_id(axis=1)
        pid_c = pid_bc // batch
        pid_b = pid_bc - pid_c * batch
        pid_h = tl.program_id(axis=2)
        num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
        pid_m = tl.program_id(axis=0) // num_pid_n
        pid_n = tl.program_id(axis=0) % num_pid_n
        cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
        x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
        dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
        dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
        C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
        prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize).to(tl.float32)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Without the if (pid_c > -1), with Triton 2.1.0, I get
        # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
        # With Triton 2.2.0, this works
        if IS_TRITON_22 or pid_c > -1:
            offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE)
            C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
            prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
            scale_m = tl.exp(dA_cs_m)
            C = tl.load(C_ptrs)
            prev_states = tl.load(prev_states_ptrs).to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
        x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
        dt_ptrs = dt_ptr + offs_k * stride_dt_csize
        dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
        for k in range(0, chunk_size, BLOCK_SIZE_K):
            cb = tl.load(cb_ptrs).to(tl.float32)
            dA_cs_k = tl.load(dA_cumsum_ptrs).to(tl.float32)
            cb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_k[None, :]), 0.0))
            dt_k = tl.load(dt_ptrs).to(tl.float32)
            cb *= dt_k
            cb = cb.to(x_ptr.dtype.element_ty)
            x = tl.load(x_ptrs)
            acc += tl.dot(cb, x)
            cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
            x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
            dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
            dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

        offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
        tl.store(out_ptrs, acc)
        out_x_ptr += (
            pid_b * stride_out_batch
            + pid_c * chunk_size * stride_out_seqlen
            + pid_h * stride_out_head
        )
        out_x_ptrs = out_x_ptr + (
            stride_out_seqlen * offs_out_m[:, None]
            + offs_out_n[None, :] * stride_out_hdim
        )
        tl.store(out_x_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32))



    def get_random_input(self, fixed: bool = False):
        del fixed
        device = "cuda"
        chunks = triton.cdiv(self.sequence_length, self.chunk_size)
        cb = torch.tril(
            torch.randn(
                self.batch_size,
                chunks,
                self.num_groups,
                self.chunk_size,
                self.chunk_size,
                device=device,
            )
        )
        x = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
            device=device,
            dtype=torch.float16,
        )
        z = torch.randn_like(x)
        dt = torch.rand(
            self.batch_size, self.num_heads, chunks, self.chunk_size, device=device
        )
        dA = -torch.rand_like(dt).cumsum(dim=-1)
        c = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.num_groups,
            self.state_dim,
            device=device,
            dtype=torch.float16,
        )
        states = torch.randn(
            self.batch_size,
            chunks,
            self.num_heads,
            self.head_dim,
            self.state_dim,
            device=device,
        )
        d = torch.randn(self.num_heads, self.head_dim, device=device)
        seq_idx = torch.arange(self.sequence_length, device=device, dtype=torch.int32)
        seq_idx = seq_idx.div(32, rounding_mode="floor").expand(self.batch_size, -1)
        return cb, x, z, dt, dA, c, states, d, seq_idx

    def get_shape_information(self) -> str:
        return "- Mamba-2 chunk scan tensors: cb, x, z, dt, dA, C, states, D, seq_idx"

    def forward_triton(self, inputs, ptx: bool = False):
        cb, x, z, dt, dA, c, states, d, seq_idx = inputs
        batch, seqlen, heads, hdim = x.shape
        chunks, dstate = dt.shape[2], c.shape[-1]
        out, out_x = torch.empty_like(x), torch.empty_like(x)
        launch = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        grid = lambda meta: (
            triton.cdiv(self.chunk_size, meta["BLOCK_SIZE_M"])
            * triton.cdiv(hdim, meta["BLOCK_SIZE_N"]),
            batch * chunks,
            heads,
        )
        kernel = launch[grid](
            cb,
            x,
            z,
            out,
            out_x,
            dt,
            dA,
            seq_idx,
            c,
            states,
            d,
            self.chunk_size,
            hdim,
            dstate,
            batch,
            seqlen,
            heads // c.shape[2],
            *cb.stride(),
            *x.stride(),
            *z.stride(),
            *out.stride(),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA.stride(0),
            dA.stride(2),
            dA.stride(1),
            dA.stride(3),
            *seq_idx.stride(),
            *c.stride(),
            *states.stride(),
            d.stride(0),
            BLOCK_SIZE_M=self.block_size_m,
            BLOCK_SIZE_N=self.block_size_n,
            BLOCK_SIZE_K=self.block_size_k,
            BLOCK_SIZE_DSTATE=32,
            IS_TRITON_22=True,
            **kwargs,
        )
        return (out, out_x), kernel

    def forward_torch(self, inputs):
        cb, x, z, dt, dA, c, states, d, seq_idx = inputs
        del z, d, seq_idx
        batch, _, heads, hdim = x.shape
        chunks = dt.shape[2]
        ratio = heads // c.shape[2]
        c = c.reshape(batch, chunks, self.chunk_size, c.shape[2], -1)
        c = c.repeat_interleave(ratio, dim=3).permute(0, 1, 3, 2, 4)
        x = x.reshape(batch, chunks, self.chunk_size, heads, hdim)
        x = x.permute(0, 1, 3, 2, 4)
        cb = cb.repeat_interleave(ratio, dim=2)
        dA = dA.permute(0, 2, 1, 3)
        dt = dt.permute(0, 2, 1, 3)

        carry = torch.einsum("bchms,bchds->bchmd", c.float(), states.float())
        carry *= dA.exp().unsqueeze(-1)
        decay = (dA.unsqueeze(-1) - dA.unsqueeze(-2)).clamp(max=0).exp()
        weights = cb * decay * dt.unsqueeze(-2)
        out = carry + torch.einsum("bchmk,bchkd->bchmd", weights, x.float())
        out = out.permute(0, 1, 3, 2, 4).reshape_as(x.permute(0, 1, 3, 2, 4))
        out = out.reshape(batch, chunks * self.chunk_size, heads, hdim).to(x.dtype)
        return out, torch.zeros_like(out)
