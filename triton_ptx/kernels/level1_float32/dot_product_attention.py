from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class DotProductAttentionKernel(TritonPTXKernel):
    """Scaled dot-product attention without masking or dropout."""

    def __init__(
        self,
        *,
        ptx: object | None = None,
    ) -> None:
        self.batch_size = 8
        self.num_heads = 16
        self.seq_len = 256
        self.head_dim = 64
        self.block_rows = 16
        self.block_seq = 256
        self.block_dim = 64
        self.stride_batch = self.num_heads * self.seq_len * self.head_dim
        self.stride_head = self.seq_len * self.head_dim
        self.constexpr_values = {
            "stride_qb": self.stride_batch,
            "stride_qh": self.stride_head,
            "stride_qs": self.head_dim,
            "stride_kb": self.stride_batch,
            "stride_kh": self.stride_head,
            "stride_ks": self.head_dim,
            "stride_vb": self.stride_batch,
            "stride_vh": self.stride_head,
            "stride_vs": self.head_dim,
            "stride_ob": self.stride_batch,
            "stride_oh": self.stride_head,
            "stride_os": self.head_dim,
            "SEQ_LEN": self.seq_len,
            "HEAD_DIM": self.head_dim,
            "BLOCK_ROWS": self.block_rows,
            "BLOCK_SEQ": self.block_seq,
            "BLOCK_DIM": self.block_dim,
        }
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qs: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_os: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """Compute a tile of attention output rows per Triton program."""
        row = tl.program_id(axis=0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        head = tl.program_id(axis=1)
        batch = tl.program_id(axis=2)
        seq_offsets = tl.arange(0, BLOCK_SEQ)
        dim_offsets = tl.arange(0, BLOCK_DIM)

        q_base = q_ptr + batch * stride_qb + head * stride_qh + row[:, None] * stride_qs
        k_base = k_ptr + batch * stride_kb + head * stride_kh
        v_base = v_ptr + batch * stride_vb + head * stride_vh

        q = tl.load(q_base + dim_offsets[None, :])
        k = tl.load(
            k_base + seq_offsets[:, None] * stride_ks + dim_offsets[None, :],
        )
        scores = tl.dot(
            q,
            tl.trans(k),
            out_dtype=tl.float32,
            input_precision="ieee",
        )
        scores *= HEAD_DIM**-0.5
        scores -= tl.max(scores, axis=1)[:, None]
        weights = tl.exp(scores)
        weights /= tl.sum(weights, axis=1)[:, None]

        values = tl.load(
            v_base + seq_offsets[:, None] * stride_vs + dim_offsets[None, :],
        )
        output = tl.dot(
            weights,
            values,
            out_dtype=tl.float32,
            input_precision="ieee",
        )
        output_base = (
            output_ptr + batch * stride_ob + head * stride_oh + row[:, None] * stride_os
        )
        tl.store(output_base + dim_offsets[None, :], output)

    def get_random_input(
        self, fixed: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create random query, key, and value tensors."""
        shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        return (
            torch.randn(shape, device="cuda", dtype=torch.float32),
            torch.randn(shape, device="cuda", dtype=torch.float32),
            torch.randn(shape, device="cuda", dtype=torch.float32),
        )

    def _default_tuning_options(self) -> dict[str, tuple[int, ...]]:
        return {
            "block_rows": (16, 32, 64, 128, 256),
            "block_seq": (self.seq_len,),
            "block_dim": (self.head_dim,),
            "num_warps": (4, 8, 16),
        }

    def get_shape_information(self) -> str:
        shape = (
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
        )
        return "\n".join(
            f"- {name}_ptr: float32 tensor with shape {shape}"
            for name in ("q", "k", "v", "output")
        )

    def forward_triton(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ptx: bool = False,
    ) -> tuple[torch.Tensor, object]:
        """Run the Triton attention kernel."""
        q, k, v = inputs
        expected_shape = (
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
        )
        if (
            q.shape != expected_shape
            or k.shape != expected_shape
            or v.shape != expected_shape
        ):
            raise ValueError(f"q, k, and v must each have shape {expected_shape}")
        output = torch.empty_like(q)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = (
            triton.cdiv(self.seq_len, self.block_rows),
            self.num_heads,
            self.batch_size,
        )
        launched_kernel = launch_kernel[grid](
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
            SEQ_LEN=self.seq_len,
            HEAD_DIM=self.head_dim,
            BLOCK_ROWS=self.block_rows,
            BLOCK_SEQ=self.block_seq,
            BLOCK_DIM=self.block_dim,
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
