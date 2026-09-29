from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


@triton.jit
def _flash_attention_fwd_inner(
    accumulator,
    row_sum,
    row_max,
    query,
    key_ptr,
    value_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    if STAGE == 1:
        lower_bound = 0
        upper_bound = start_m * BLOCK_M
    else:
        lower_bound = start_m * BLOCK_M
        upper_bound = lower_bound + BLOCK_M
        lower_bound = tl.multiple_of(lower_bound, BLOCK_M)

    column_offsets = tl.arange(0, BLOCK_N)
    dimension_offsets = tl.arange(0, HEAD_DIM)
    row_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    key_ptrs = (
        key_ptr
        + (lower_bound + column_offsets[None, :]) * HEAD_DIM
        + dimension_offsets[:, None]
    )
    value_ptrs = (
        value_ptr
        + (lower_bound + column_offsets[:, None]) * HEAD_DIM
        + dimension_offsets[None, :]
    )

    for start_n in tl.range(lower_bound, upper_bound, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key = tl.load(key_ptrs)
        scores = tl.dot(query, key, out_dtype=tl.float32) * qk_scale
        if STAGE == 2:
            causal_mask = row_offsets[:, None] >= (start_n + column_offsets[None, :])
            scores = tl.where(causal_mask, scores, -1.0e6)

        block_max = tl.maximum(row_max, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - block_max[:, None])
        correction = tl.math.exp2(row_max - block_max)
        block_sum = tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        value = tl.load(value_ptrs).to(tl.float16)
        accumulator = tl.dot(
            probabilities.to(tl.float16),
            value,
            acc=accumulator,
            out_dtype=tl.float32,
        )
        row_sum = row_sum * correction + block_sum
        row_max = block_max
        key_ptrs += BLOCK_N * HEAD_DIM
        value_ptrs += BLOCK_N * HEAD_DIM

    return accumulator, row_sum, row_max


class AdversialFlashAttentionKernel(TritonPTXKernel):
    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_m": (32, 64, 128, 256),
        "block_n": (32, 64, 128, 256),
        "num_warps": (4, 8, 16),
        "num_stages": (2, 3),
    }

    autotune_tolerance = 1e-2

    def __init__(self, *, ptx=None):
        self.batch_size = 8
        self.num_heads = 16
        self.seq_len = 256
        self.head_dim = 64
        self.block_m = 128
        self.block_n = 32
        self.stage = 3
        self.num_warps = 8
        self.num_stages = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
    ):
        tl.static_assert(STAGE == 3)
        tl.static_assert(SEQ_LEN % BLOCK_M == 0)
        tl.static_assert(BLOCK_M % BLOCK_N == 0)
        tl.static_assert(BLOCK_N <= HEAD_DIM)

        start_m = tl.program_id(axis=0)
        batch_head = tl.program_id(axis=1)
        tensor_offset = batch_head * SEQ_LEN * HEAD_DIM
        query_ptr += tensor_offset
        key_ptr += tensor_offset
        value_ptr += tensor_offset
        output_ptr += tensor_offset

        row_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        query = tl.load(
            query_ptr + row_offsets[:, None] * HEAD_DIM + dimension_offsets[None, :]
        )

        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        # Intentionally incorrect: the base-2 conversion factor is added rather
        # than multiplied, so this does not match scaled dot-product attention.
        qk_scale = HEAD_DIM**-0.5 + 1.4426950408889634

        accumulator, row_sum, row_max = _flash_attention_fwd_inner(
            accumulator,
            row_sum,
            row_max,
            query,
            key_ptr,
            value_ptr,
            start_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            1,
        )
        accumulator, row_sum, _ = _flash_attention_fwd_inner(
            accumulator,
            row_sum,
            row_max,
            query,
            key_ptr,
            value_ptr,
            start_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            2,
        )
        output = accumulator / row_sum[:, None]
        output_ptrs = (
            output_ptr + row_offsets[:, None] * HEAD_DIM + dimension_offsets[None, :]
        )
        tl.store(output_ptrs, output)

    def get_random_input(self, fixed=False):
        shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        return tuple(
            torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)
        )

    def get_shape_information(self):
        shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        return "\n".join(
            f"- {name}_ptr: float16 tensor with shape {shape}"
            for name in ("query", "key", "value", "output")
        )

    def forward_triton(self, inputs, ptx=False):
        query, key, value = inputs
        output = torch.empty_like(query)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = (
            self.seq_len // self.block_m,
            self.batch_size * self.num_heads,
        )
        launched_kernel = launch_kernel[grid](
            query,
            key,
            value,
            output,
            SEQ_LEN=self.seq_len,
            HEAD_DIM=self.head_dim,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            STAGE=self.stage,
            num_stages=self.num_stages,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, inputs):
        query, key, value = inputs
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
