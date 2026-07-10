from __future__ import annotations

import torch
import torch.nn.functional as F
import triton.language as tl

from triton_ptx.kernels.attention.causal_dot_product_attention import (
    CausalDotProductAttentionKernel,
)


class GroupedQueryAttentionKernel(CausalDotProductAttentionKernel):
    """Scaled dot-product grouped-query attention."""

    def __init__(
        self,
        *,
        batch_size: int = 2,
        num_heads: int = 4,
        kv_heads: int = 2,
        seq_len: int = 128,
        head_dim: int = 64,
        num_warps: int = 4,
        ptx: object | None = None,
    ) -> None:
        self.kv_heads = kv_heads
        super().__init__(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_warps=num_warps,
            ptx=ptx,
        )

    @staticmethod
    def kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
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
        KV_GROUP_SIZE: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Compute one grouped-query attention output row per Triton program."""
        row = tl.program_id(axis=0)
        head = tl.program_id(axis=1)
        batch = tl.program_id(axis=2)
        kv_head = head // KV_GROUP_SIZE
        seq_offsets = tl.arange(0, SEQ_LEN)
        dim_offsets = tl.arange(0, HEAD_DIM)
        q_base = q_ptr + batch * stride_qb + head * stride_qh + row * stride_qs
        k_base = k_ptr + batch * stride_kb + kv_head * stride_kh
        v_base = v_ptr + batch * stride_vb + kv_head * stride_vh

        q = tl.load(q_base + dim_offsets)
        k = tl.load(k_base + seq_offsets[:, None] * stride_ks + dim_offsets[None, :])
        scores = tl.sum(k * q[None, :], axis=1) * (HEAD_DIM**-0.5)
        scores -= tl.max(scores, axis=0)
        weights = tl.exp(scores)
        weights /= tl.sum(weights, axis=0)
        values = tl.load(v_base + seq_offsets[:, None] * stride_vs + dim_offsets[None, :])
        output = tl.sum(weights[:, None] * values, axis=0)
        output_base = output_ptr + batch * stride_ob + head * stride_oh + row * stride_os
        tl.store(output_base + dim_offsets, output)

    def get_random_input(
        self, seq_len: int | None = None, head_dim: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create query, key, and value tensors for grouped-query attention."""
        seq_len = self.seq_len if seq_len is None else min(seq_len, self.seq_len)
        head_dim = self.head_dim if head_dim is None else min(head_dim, self.head_dim)
        query_shape = (self.batch_size, self.num_heads, seq_len, head_dim)
        key_shape = (self.batch_size, self.kv_heads, seq_len, head_dim)
        return (
            torch.randn(query_shape, device="cuda", dtype=torch.float32),
            torch.randn(key_shape, device="cuda", dtype=torch.float32),
            torch.randn(key_shape, device="cuda", dtype=torch.float32),
        )

    def forward_triton(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], ptx: bool = False
    ) -> tuple[torch.Tensor, object]:
        """Run the Triton grouped-query attention implementation."""
        q, k, v = inputs
        output = torch.empty_like(q)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        batch_size, num_heads, seq_len, head_dim = q.shape
        kv_group_size = num_heads // k.shape[1]
        launched_kernel = launch_kernel[(seq_len, num_heads, batch_size)](
            q,
            k,
            v,
            output,
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
            KV_GROUP_SIZE=kv_group_size,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Evaluate grouped-query attention with PyTorch."""
        q, k, v = inputs
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, enable_gqa=True)
