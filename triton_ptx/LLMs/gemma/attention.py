"""Triton attention kernel for Gemma."""

import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel


class GemmaAttentionKernel(TritonPTXKernel):
    """Gemma's causal and sliding-window scaled dot-product attention kernel."""

    def __init__(self, *, ptx=None):
        self.block_m = 16
        self.block_n = 32
        self.block_d = 512
        self.num_warps = 4
        self.constexpr_values = {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_D": self.block_d,
        }
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        query_length,
        key_value_length,
        num_heads,
        window_size,
        query_batch_stride,
        query_head_stride,
        query_row_stride,
        key_batch_stride,
        key_head_stride,
        key_row_stride,
        value_batch_stride,
        value_head_stride,
        value_row_stride,
        head_dim,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(axis=0)
        batch_head = tl.program_id(axis=1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        feature_offsets = tl.arange(0, BLOCK_D)
        feature_mask = feature_offsets < head_dim
        batch_index = batch_head // num_heads
        head_index = batch_head % num_heads
        query_batch_head_offset = (
            batch_index * query_batch_stride + head_index * query_head_stride
        )
        key_batch_head_offset = (
            batch_index * key_batch_stride + head_index * key_head_stride
        )
        value_batch_head_offset = (
            batch_index * value_batch_stride + head_index * value_head_stride
        )
        output_batch_head_offset = batch_head * query_length * head_dim
        query_mask = query_offsets < query_length
        max_scores = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        score_sums = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_block_start in tl.range(0, key_value_length, BLOCK_N):
            current_key_offsets = key_block_start + key_offsets
            key_mask = current_key_offsets < key_value_length
            queries = tl.load(
                query_ptr
                + query_batch_head_offset
                + query_offsets[:, None] * query_row_stride
                + feature_offsets[None, :],
                mask=query_mask[:, None] & feature_mask[None, :],
                other=0.0,
            )
            keys = tl.load(
                key_ptr
                + key_batch_head_offset
                + current_key_offsets[:, None] * key_row_stride
                + feature_offsets[None, :],
                mask=key_mask[:, None] & feature_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(queries, tl.trans(keys))
            query_positions = query_offsets + key_value_length - query_length
            causal_mask = query_positions[:, None] >= current_key_offsets[None, :]
            window_mask = (window_size == 0) | (
                current_key_offsets[None, :]
                >= query_positions[:, None] - window_size + 1
            )
            scores = tl.where(
                causal_mask & window_mask & key_mask[None, :], scores, -float("inf")
            )
            block_max_scores = tl.max(scores, axis=1)
            next_max_scores = tl.maximum(max_scores, block_max_scores)
            probabilities = tl.exp(scores - next_max_scores[:, None])
            rescale = tl.exp(max_scores - next_max_scores)
            score_sums = score_sums * rescale + tl.sum(probabilities, axis=1)
            values = tl.load(
                value_ptr
                + value_batch_head_offset
                + current_key_offsets[:, None] * value_row_stride
                + feature_offsets[None, :],
                mask=key_mask[:, None] & feature_mask[None, :],
                other=0.0,
            )
            accumulator = accumulator * rescale[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            max_scores = next_max_scores

        tl.store(
            output_ptr
            + output_batch_head_offset
            + query_offsets[:, None] * head_dim
            + feature_offsets[None, :],
            accumulator / score_sums[:, None],
            mask=query_mask[:, None] & feature_mask[None, :],
        )

    def get_random_input(self, fixed: bool = False):
        query = torch.rand((1, 8, 1, 256), device="cuda", dtype=torch.bfloat16)
        key_value = tuple(
            torch.rand((1, 8, 32, 256), device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
        return query, *key_value, 16

    def get_shape_information(self) -> str:
        return (
            "- query_ptr: bfloat16 tensor with shape (1, 8, 1, 256)\n"
            "- key_ptr, value_ptr: bfloat16 tensors with shape "
            "(1, 8, key_value_length, 256)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 8, 1, 256)\n"
            "- window_size: 0 for causal attention, otherwise the local window size"
        )

    def forward_triton(self, inputs, ptx=False):
        query_states, key_states, value_states, window_size = inputs
        batch_size, num_heads, query_length, head_dim = query_states.shape
        key_value_length = key_states.shape[-2]
        assert query_states.is_cuda, "Attention requires CUDA queries."
        assert key_states.is_cuda and value_states.is_cuda, (
            "Attention requires CUDA keys and values."
        )
        assert query_states.device == key_states.device == value_states.device, (
            "Attention inputs must be on the same device."
        )
        assert query_states.dtype in (torch.float16, torch.bfloat16), (
            "Attention queries must use float16 or bfloat16."
        )
        assert key_states.dtype == query_states.dtype == value_states.dtype, (
            "Attention inputs must use the same dtype."
        )
        assert all(tensor.is_contiguous() for tensor in inputs[:3]), (
            "Attention inputs must be contiguous."
        )
        assert key_states.shape == value_states.shape, (
            "Attention keys and values must have the same shape."
        )
        assert key_states.shape[:2] == (batch_size, num_heads), (
            "Attention keys must match the query batch and head dimensions."
        )
        assert key_states.shape[-1] == head_dim, (
            "Attention keys must match the query head dimension."
        )
        assert 0 < query_length <= key_value_length, (
            "Attention query length must be positive and no greater than the "
            "key/value length."
        )
        assert head_dim in (256, 512), (
            "Gemma attention head dimension must be 256 or 512."
        )
        assert isinstance(window_size, int) and window_size >= 0, (
            "Attention window size must be a non-negative integer."
        )
        output = torch.empty_like(query_states)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[
            (
                triton.cdiv(query_length, self.block_m),
                batch_size * num_heads,
                1,
            )
        ](
            query_states,
            key_states,
            value_states,
            output,
            query_length=query_length,
            key_value_length=key_value_length,
            num_heads=num_heads,
            window_size=window_size,
            query_batch_stride=query_states.stride(0),
            query_head_stride=query_states.stride(1),
            query_row_stride=query_states.stride(2),
            key_batch_stride=key_states.stride(0),
            key_head_stride=key_states.stride(1),
            key_row_stride=key_states.stride(2),
            value_batch_stride=value_states.stride(0),
            value_head_stride=value_states.stride(1),
            value_row_stride=value_states.stride(2),
            head_dim=head_dim,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_D=self.block_d,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        query_states, key_states, value_states, window_size = inputs
        query_length = query_states.shape[-2]
        key_value_length = key_states.shape[-2]
        query_positions = torch.arange(
            key_value_length - query_length,
            key_value_length,
            device=query_states.device,
        )
        key_positions = torch.arange(key_value_length, device=query_states.device)
        attention_mask = key_positions[None, :] <= query_positions[:, None]
        if window_size:
            attention_mask &= key_positions[None, :] >= (
                query_positions[:, None] - window_size + 1
            )
        return functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            scale=1.0,
        )
