import torch

from triton_ptx.helpers.serialization import tensor_bytes


def test_tensor_bytes_counts_nested_tensors():
    """Count tensor bytes recursively while ignoring non-tensor values."""
    data = {
        "a": torch.empty(4, dtype=torch.float32),
        "b": [
            torch.empty((2, 3), dtype=torch.float16),
            "ignored",
        ],
    }

    assert tensor_bytes(data) == 4 * 4 + 2 * 3 * 2

    assert tensor_bytes(torch.tensor([1, 2, 3], dtype=torch.int32)) == 3 * 4
