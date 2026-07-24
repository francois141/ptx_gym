from __future__ import annotations

import math

import torch
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class FlashAttentionKernel(TritonPTXKernel):
    """Forward-only FlashAttention-style scaled dot-product attention."""

    def __init__(
        self,
        *,
        batch_size: int = 2,
        num_heads: int = 4,
        seq_len: int = 128,
        head_dim: int = 64,
        block_seq: int = 32,
        num_warps: int = 4,
        ptx: object | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.block_seq = block_seq
        self.constexpr_values = {
            "BLOCK_M": block_seq,
            "BLOCK_N": block_seq,
            "BLOCK_D": 1 << (head_dim - 1).bit_length(),
            "HEAD_DIM": head_dim,
            "NUM_HEADS": num_heads,
        }
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        softmax_scale,
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
        num_heads: tl.constexpr,
        seqlen_q,
        seqlen_k,
        head_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute a query block using online softmax over key/value blocks."""
        start_m = tl.program_id(axis=0)
        batch_head = tl.program_id(axis=1)
        batch = batch_head // num_heads
        head = batch_head % num_heads
        offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tl.arange(0, BLOCK_N)
        offsets_d = tl.arange(0, BLOCK_D)
        query_mask = (offsets_m[:, None] < seqlen_q) & (offsets_d[None, :] < head_dim)
        q_ptrs = (
            q_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :]
        )
        q = tl.load(q_ptrs, mask=query_mask, other=0.0)
        k_ptrs = (
            k_ptr
            + batch * stride_kb
            + head * stride_kh
            + offsets_n[:, None] * stride_ks
            + offsets_d[None, :]
        )
        v_ptrs = (
            v_ptr
            + batch * stride_vb
            + head * stride_vh
            + offsets_n[:, None] * stride_vs
            + offsets_d[None, :]
        )
        maximum = tl.full([BLOCK_M], -float("inf"), tl.float32)
        normalizer = tl.zeros([BLOCK_M], tl.float32)
        accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        for start_n in tl.range(0, seqlen_k, BLOCK_N):
            key_mask = (start_n + offsets_n[:, None] < seqlen_k) & (
                offsets_d[None, :] < head_dim
            )
            k = tl.load(k_ptrs + start_n * stride_ks, mask=key_mask, other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * softmax_scale
            scores = tl.where(
                (start_n + offsets_n)[None, :] < seqlen_k,
                scores,
                -float("inf"),
            )
            block_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            probabilities = tl.exp(scores - block_maximum[:, None])
            correction = tl.exp(maximum - block_maximum)
            normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
            v = tl.load(v_ptrs + start_n * stride_vs, mask=key_mask, other=0.0)
            accumulator = accumulator * correction[:, None]
            accumulator = tl.dot(
                probabilities.to(v.dtype), v, accumulator, input_precision="ieee"
            )
            maximum = block_maximum

        output = accumulator / normalizer[:, None]
        output_ptrs = (
            output_ptr
            + batch * stride_ob
            + head * stride_oh
            + offsets_m[:, None] * stride_os
            + offsets_d[None, :]
        )
        tl.store(output_ptrs, output, mask=query_mask)

    def get_random_input(
        self,
        seq_len: int | None = None,
        head_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create random self-attention inputs."""
        seq_len = self.seq_len
        head_dim = self.head_dim
        shape = (self.batch_size, self.num_heads, seq_len, head_dim)
        return tuple(
            torch.randn(shape, device="cuda", dtype=torch.float32) for _ in range(3)
        )

    def forward_triton(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ptx: bool = False,
    ) -> tuple[torch.Tensor, object]:
        """Run the FlashAttention-style forward kernel."""
        q, k, v = inputs
        self._validate_inputs(q, k, v)
        output = torch.empty_like(q)
        block_dim = 1 << (q.shape[-1] - 1).bit_length()
        block_seq = min(self.block_seq, 128)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = ((q.shape[-2] + block_seq - 1) // block_seq, q.shape[0] * q.shape[1])
        launched_kernel = launch_kernel[grid](
            q,
            k,
            v,
            output,
            1.0 / math.sqrt(q.shape[-1]),
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
            num_heads=q.shape[1],
            seqlen_q=q.shape[-2],
            seqlen_k=k.shape[-2],
            head_dim=q.shape[-1],
            BLOCK_M=block_seq,
            BLOCK_N=block_seq,
            BLOCK_D=block_dim,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """PyTorch execution is intentionally unavailable for this kernel."""
        raise NotImplementedError(
            "FlashAttention has no PyTorch forward implementation."
        )

    @staticmethod
    def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "q, k, and v must be 4D tensors shaped (batch, heads, seq, dim)."
            )
        if k.shape != v.shape:
            raise ValueError("k and v must have identical shapes.")
        if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
            raise ValueError(
                "q, k, and v must have matching batch, head, and head dimensions."
            )
        if q.shape[-2] < 1 or k.shape[-2] < 1:
            raise ValueError("query and key sequence lengths must be positive.")
        if q.shape[-1] > 128:
            raise ValueError("FlashAttention supports head dimensions up to 128.")
        if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
            raise ValueError("q, k, and v must be on the same CUDA device.")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("q, k, and v must have the same dtype.")
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            raise ValueError("q, k, and v must be contiguous in their head dimension.")
