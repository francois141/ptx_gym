import triton.language as tl
from triton_ptx.kernels.level2_float16.flash_attention import (
    FlashAttentionFloat16Kernel,
    _flash_attention_fwd_inner,
)


class AdversialFlashAttentionKernel(FlashAttentionFloat16Kernel):

    # There is a mistake in qk.scale on purpose instead of a * et have a +  for 1.44.....
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
