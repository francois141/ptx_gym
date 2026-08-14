"""Triton causal-attention kernel for Apertus."""

import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel

INFERENCE_BATCH_SIZE = 1
INFERENCE_NUM_ATTENTION_HEADS = 32
HEAD_DIM = 128
KERNEL_HEAD_DIM = tl.constexpr(HEAD_DIM)


class CausalAttentionKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_m = 32
        self.block_n = 64
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_M": self.block_m, "BLOCK_N": self.block_n}
        self.init_compiled_kernels(ptx=ptx, autotune=True)

    @staticmethod
    def kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        query_length,
        key_value_length,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(axis=0)
        batch_head = tl.program_id(axis=1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        feature_offsets = tl.arange(0, KERNEL_HEAD_DIM)
        query_head_offset = batch_head * query_length * KERNEL_HEAD_DIM
        key_value_head_offset = batch_head * key_value_length * KERNEL_HEAD_DIM
        query_mask = query_offsets < query_length
        query_ptrs = (
            query_ptr
            + query_head_offset
            + query_offsets[:, None] * KERNEL_HEAD_DIM
            + feature_offsets[None, :]
        )
        queries = tl.load(query_ptrs, mask=query_mask[:, None], other=0.0)
        max_scores = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        score_sums = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, KERNEL_HEAD_DIM), tl.float32)

        for key_block_start in tl.range(0, key_value_length, BLOCK_N):
            current_key_offsets = key_block_start + key_offsets
            key_mask = current_key_offsets < key_value_length
            key_ptrs = (
                key_ptr
                + key_value_head_offset
                + current_key_offsets[:, None] * KERNEL_HEAD_DIM
                + feature_offsets[None, :]
            )
            keys = tl.load(key_ptrs, mask=key_mask[:, None], other=0.0)
            scores = tl.dot(queries, tl.trans(keys)) * scale
            query_positions = query_offsets + key_value_length - query_length
            causal_mask = query_positions[:, None] >= current_key_offsets[None, :]
            scores = tl.where(causal_mask, scores, -float("inf"))
            block_max_scores = tl.max(scores, axis=1)
            next_max_scores = tl.maximum(max_scores, block_max_scores)
            probabilities = tl.exp(scores - next_max_scores[:, None])
            rescale = tl.exp(max_scores - next_max_scores)
            score_sums = score_sums * rescale + tl.sum(probabilities, axis=1)
            value_ptrs = (
                value_ptr
                + key_value_head_offset
                + current_key_offsets[:, None] * KERNEL_HEAD_DIM
                + feature_offsets[None, :]
            )
            values = tl.load(value_ptrs, mask=key_mask[:, None], other=0.0)
            accumulator = accumulator * rescale[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            max_scores = next_max_scores

        output_ptrs = (
            output_ptr
            + query_head_offset
            + query_offsets[:, None] * KERNEL_HEAD_DIM
            + feature_offsets[None, :]
        )
        tl.store(
            output_ptrs,
            accumulator / score_sums[:, None],
            mask=query_mask[:, None],
        )

    def get_random_input(self, fixed: bool = False):
        query = torch.rand(
            (INFERENCE_BATCH_SIZE, INFERENCE_NUM_ATTENTION_HEADS, 1, HEAD_DIM),
            device="cuda",
            dtype=torch.float16,
        )
        key_value = tuple(
            torch.rand(
                (INFERENCE_BATCH_SIZE, INFERENCE_NUM_ATTENTION_HEADS, 38, HEAD_DIM),
                device="cuda",
                dtype=torch.float16,
            )
            for _ in range(2)
        )
        return query, *key_value

    def get_shape_information(self) -> str:
        return (
            "- query_ptr: float16 tensor with shape (1, 32, 1, 128)\n"
            "- key_ptr, value_ptr: float16 tensors with shape "
            "(1, 32, key_value_length, 128)\n"
            "- output_ptr: float16 tensor with shape (1, 32, 1, 128)"
        )

    def forward_triton(self, inputs, ptx=False):
        query_states, key_states, value_states = inputs
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
        assert query_states.is_contiguous(), "Attention queries must be contiguous."
        assert key_states.is_contiguous(), "Attention keys must be contiguous."
        assert value_states.is_contiguous(), "Attention values must be contiguous."
        expected_key_value_shape = (
            INFERENCE_BATCH_SIZE,
            INFERENCE_NUM_ATTENTION_HEADS,
            key_value_length,
            HEAD_DIM,
        )
        assert key_states.shape == expected_key_value_shape, (
            "Attention keys have an invalid shape."
        )
        assert value_states.shape == expected_key_value_shape, (
            "Attention values have an invalid shape."
        )
        assert 0 < query_length <= key_value_length, (
            "Attention query length must be positive and no greater than the "
            "key/value length."
        )
        assert (batch_size, num_heads, head_dim) == (
            INFERENCE_BATCH_SIZE,
            INFERENCE_NUM_ATTENTION_HEADS,
            HEAD_DIM,
        ), (
            "Apertus inference attention requires query, key, and value shape "
            f"(1, 32, query_length, 128); received {query_states.shape}."
        )
        output = torch.empty_like(query_states)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[
            (
                triton.cdiv(query_length, self.block_m),
                INFERENCE_NUM_ATTENTION_HEADS,
            )
        ](
            query_states,
            key_states,
            value_states,
            output,
            query_length=query_length,
            key_value_length=key_value_length,
            scale=HEAD_DIM**-0.5,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        query_states, key_states, value_states = inputs
        query_length = query_states.shape[-2]
        key_value_length = key_states.shape[-2]
        query_positions = torch.arange(
            key_value_length - query_length,
            key_value_length,
            device=query_states.device,
        )
        key_positions = torch.arange(key_value_length, device=query_states.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        return functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
        )
