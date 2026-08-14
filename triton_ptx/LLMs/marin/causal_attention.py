import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel

INFERENCE_BATCH_SIZE = 1
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
KERNEL_NUM_ATTENTION_HEADS = tl.constexpr(NUM_ATTENTION_HEADS)
KERNEL_NUM_KEY_VALUE_HEADS = tl.constexpr(NUM_KEY_VALUE_HEADS)
KERNEL_HEAD_DIM = tl.constexpr(HEAD_DIM)


class MarinCausalAttentionKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_m = 16
        self.block_n = 32
        self.num_warps = 4
        self.constexpr_values = {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
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
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        query_head = tl.program_id(1)
        key_value_head = query_head // (
            KERNEL_NUM_ATTENTION_HEADS // KERNEL_NUM_KEY_VALUE_HEADS
        )
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        feature_offsets = tl.arange(0, KERNEL_HEAD_DIM)
        query_head_offset = query_head * query_length * KERNEL_HEAD_DIM
        key_value_head_offset = key_value_head * key_value_length * KERNEL_HEAD_DIM
        query_mask = query_offsets < query_length
        query_ptrs = (
            query_ptr
            + query_head_offset
            + query_offsets[:, None] * KERNEL_HEAD_DIM
            + feature_offsets[None, :]
        )
        queries = tl.load(query_ptrs, mask=query_mask[:, None], other=0.0)
        maximum_scores = tl.full((BLOCK_M,), -float("inf"), tl.float32)
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
            causal_mask = (query_positions[:, None] >= current_key_offsets[None, :]) & (
                key_mask[None, :]
            )
            scores = tl.where(causal_mask, scores, -float("inf"))
            block_maximum_scores = tl.max(scores, axis=1)
            next_maximum_scores = tl.maximum(maximum_scores, block_maximum_scores)
            probabilities = tl.exp(scores - next_maximum_scores[:, None])
            rescale = tl.exp(maximum_scores - next_maximum_scores)
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
            maximum_scores = next_maximum_scores
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

    def get_random_input(self, fixed=False):
        query = torch.rand(
            (1, NUM_ATTENTION_HEADS, 1, HEAD_DIM),
            device="cuda",
            dtype=torch.bfloat16,
        )
        key_value = tuple(
            torch.rand(
                (1, NUM_KEY_VALUE_HEADS, 38, HEAD_DIM),
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        )
        return query, *key_value

    def get_shape_information(self):
        return (
            "- query_ptr: bfloat16 tensor with shape (1, 32, 1, 128)\n"
            "- key_ptr, value_ptr: bfloat16 tensors with shape "
            "(1, 8, key_value_length, 128)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 32, 1, 128)"
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
            "Attention inputs must have the same dtype."
        )
        assert query_states.is_contiguous(), "Queries must be contiguous."
        assert key_states.is_contiguous(), "Keys must be contiguous."
        assert value_states.is_contiguous(), "Values must be contiguous."
        assert 0 < query_length <= key_value_length, (
            "Attention query length must be positive and no greater than the "
            "key/value length."
        )
        assert (batch_size, num_heads, head_dim) == (
            INFERENCE_BATCH_SIZE,
            NUM_ATTENTION_HEADS,
            HEAD_DIM,
        ), "Marin attention requires shape (1, 32, query_length, 128)."
        expected_key_value_shape = (
            INFERENCE_BATCH_SIZE,
            NUM_KEY_VALUE_HEADS,
            key_value_length,
            HEAD_DIM,
        )
        assert key_states.shape == expected_key_value_shape, (
            "Marin keys must have shape (1, 8, key_value_length, 128)."
        )
        assert value_states.shape == expected_key_value_shape, (
            "Marin values must have shape (1, 8, key_value_length, 128)."
        )
        output = torch.empty_like(query_states)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[
            (triton.cdiv(query_length, self.block_m), NUM_ATTENTION_HEADS)
        ](
            query_states,
            key_states,
            value_states,
            output,
            query_length,
            key_value_length,
            HEAD_DIM**-0.5,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        query_states, key_states, value_states = inputs
        repeat_factor = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
        value_states = value_states.repeat_interleave(repeat_factor, dim=1)
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
