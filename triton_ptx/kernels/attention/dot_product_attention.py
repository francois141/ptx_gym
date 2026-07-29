from __future__ import annotations

import torch
import torch.nn.functional as F
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class DotProductAttentionKernel(TritonPTXKernel):
    """Scaled dot-product attention without masking or dropout."""

    def __init__(
        self,
        *,
        batch_size: int = 2,
        num_heads: int = 4,
        seq_len: int = 128,
        head_dim: int = 64,
        block_seq: int = 128,
        block_dim: int = 64,
        num_warps: int = 4,
        ptx: object | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.block_seq = block_seq
        self.block_dim = block_dim
        self.constexpr_values = {
            "SEQ_LEN": seq_len,
            "HEAD_DIM": head_dim,
            "BLOCK_SEQ": block_seq,
            "BLOCK_DIM": block_dim,
        }
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        n_rows,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_ob,
        stride_oh,
        stride_os,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """Compute one attention output row per Triton program."""
        row = tl.program_id(axis=0)
        head = tl.program_id(axis=1)
        batch = tl.program_id(axis=2)
        seq_offsets = tl.arange(0, BLOCK_SEQ)
        dim_offsets = tl.arange(0, BLOCK_DIM)
        seq_mask = seq_offsets < SEQ_LEN
        dim_mask = dim_offsets < HEAD_DIM

        q_base = q_ptr + batch * stride_qb + head * stride_qh + row * stride_qs
        k_base = k_ptr + batch * stride_kb + head * stride_kh
        v_base = v_ptr + batch * stride_vb + head * stride_vh

        q = tl.load(q_base + dim_offsets, mask=dim_mask, other=0.0)
        k = tl.load(
            k_base + seq_offsets[:, None] * stride_ks + dim_offsets[None, :],
            mask=seq_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        scores = tl.sum(k * q[None, :], axis=1) * (HEAD_DIM**-0.5)
        scores = tl.where(seq_mask, scores, -float("inf"))
        scores -= tl.max(scores, axis=0)
        weights = tl.exp(scores)
        weights /= tl.sum(weights, axis=0)

        values = tl.load(
            v_base + seq_offsets[:, None] * stride_vs + dim_offsets[None, :],
            mask=seq_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        output = tl.sum(weights[:, None] * values, axis=0)
        output_base = (
            output_ptr + batch * stride_ob + head * stride_oh + row * stride_os
        )
        tl.store(output_base + dim_offsets, output, mask=(row < n_rows) & dim_mask)

    def get_random_input(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create random query, key, and value tensors."""
        seq_len = min(self.seq_len, self.block_seq)
        head_dim = min(self.head_dim, self.block_dim)
        shape = (self.batch_size, self.num_heads, seq_len, head_dim)
        return (
            torch.randn(shape, device="cuda", dtype=torch.float32),
            torch.randn(shape, device="cuda", dtype=torch.float32),
            torch.randn(shape, device="cuda", dtype=torch.float32),
        )

    def forward_triton(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ptx: bool = False,
    ) -> tuple[torch.Tensor, object]:
        """Run the Triton attention kernel."""
        q, k, v = inputs
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have the same shape")
        if q.ndim != 4:
            raise ValueError("q, k, and v must be 4D tensors")
        output = torch.empty_like(q)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        batch_size, num_heads, seq_len, head_dim = q.shape
        grid = (seq_len, num_heads, batch_size)
        launched_kernel = launch_kernel[grid](
            q,
            k,
            v,
            output,
            seq_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            BLOCK_SEQ=max(seq_len, 1),
            BLOCK_DIM=max(head_dim, 1),
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate attention with PyTorch's fused reference implementation."""
        q, k, v = inputs
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
