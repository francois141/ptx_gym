import triton.language as tl

VECTOR_SIZE = 4096 * 4096
BATCH_SIZE = 128
VECTOR_SIZE_CONSTEXPR = tl.constexpr(VECTOR_SIZE)
