import torch
import triton.language as tl

VECTOR_SIZE = 4096 * 4096
BATCH_SIZE = 8
VECTOR_SIZE_CONSTEXPR = tl.constexpr(VECTOR_SIZE)


def random_softmax_input(batch_size, size, dtype):
    values = torch.rand((batch_size, size), device="cuda", dtype=dtype)
    row_indices = torch.arange(batch_size, device="cuda")
    peak_indices = torch.randint(size, (batch_size,), device="cuda")
    values[row_indices, peak_indices] = 20.0
    return values
