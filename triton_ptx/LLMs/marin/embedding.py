import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel


class MarinEmbeddingKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 256
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        input_ids_ptr,
        weight_ptr,
        output_ptr,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_index = tl.program_id(0)
        feature_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = feature_offsets < hidden_size
        token_id = tl.load(input_ids_ptr + token_index)
        values = tl.load(
            weight_ptr + token_id * hidden_size + feature_offsets,
            mask=mask,
        )
        tl.store(
            output_ptr + token_index * hidden_size + feature_offsets,
            values,
            mask=mask,
        )

    def get_random_input(self, fixed=False):
        return (
            torch.randint(0, 128, (1, 1), device="cuda"),
            torch.rand((128, 4096), device="cuda", dtype=torch.bfloat16),
        )

    def get_shape_information(self):
        return (
            "- input_ids_ptr: int64 tensor with shape (1, 1)\n"
            "- weight_ptr: bfloat16 tensor with shape (vocab_size, 4096)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 1, 4096)"
        )

    def forward_triton(self, inputs, ptx=False):
        input_ids, weight = inputs
        hidden_size = weight.shape[1]
        assert input_ids.is_cuda, "Embedding requires CUDA token IDs."
        assert input_ids.is_contiguous(), "Token IDs must be contiguous."
        assert input_ids.dtype == torch.int64, "Token IDs must use int64."
        assert weight.is_cuda, "Embedding weights must be on CUDA."
        assert weight.is_contiguous(), "Embedding weights must be contiguous."
        assert weight.dtype in (torch.float16, torch.bfloat16), (
            "Embedding weights must use float16 or bfloat16."
        )
        assert input_ids.device == weight.device, (
            "Token IDs and embedding weights must be on one device."
        )
        output = torch.empty(
            (*input_ids.shape, hidden_size),
            device=weight.device,
            dtype=weight.dtype,
        )
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[
            (input_ids.numel(), triton.cdiv(hidden_size, self.block_size))
        ](
            input_ids,
            weight,
            output,
            hidden_size,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        input_ids, weight = inputs
        return functional.embedding(input_ids, weight)
