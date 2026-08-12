import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class QwenRMSNormKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 128
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_offset = tl.program_id(0) * hidden_size
        feature_offsets = tl.arange(0, BLOCK_SIZE)
        squared_sum = 0.0
        for block_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offsets = block_offset + feature_offsets
            values = tl.load(
                input_ptr + row_offset + offsets,
                mask=offsets < hidden_size,
                other=0.0,
            ).to(tl.float32)
            squared_sum += tl.sum(values * values, axis=0)

        inverse_rms = tl.rsqrt(squared_sum / hidden_size + eps)
        for block_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offsets = block_offset + feature_offsets
            mask = offsets < hidden_size
            values = tl.load(
                input_ptr + row_offset + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weights = tl.load(weight_ptr + offsets, mask=mask)
            tl.store(
                output_ptr + row_offset + offsets,
                values * inverse_rms * weights,
                mask=mask,
            )

    def get_random_input(self, fixed=False):
        return (
            torch.rand((1, 2560), device="cuda", dtype=torch.bfloat16),
            torch.rand(2560, device="cuda", dtype=torch.bfloat16),
            1e-6,
        )

    def get_shape_information(self):
        return (
            "- input_ptr: bfloat16 tensor with shape (1, 2560)\n"
            "- weight_ptr: bfloat16 tensor with shape (hidden_size,)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 2560)"
        )

    def forward_triton(self, inputs, ptx=False):
        hidden_states, weight, eps = inputs
        hidden_size = hidden_states.shape[-1]
        assert hidden_states.is_cuda, "RMSNorm requires CUDA input."
        assert hidden_states.is_contiguous(), "RMSNorm input must be contiguous."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
            "RMSNorm input must use float16 or bfloat16."
        )
        assert weight.is_cuda, "RMSNorm weights must be on CUDA."
        assert weight.is_contiguous(), "RMSNorm weights must be contiguous."
        assert weight.device == hidden_states.device, (
            "RMSNorm weights must be on the input device."
        )
        assert weight.dtype == hidden_states.dtype, (
            "RMSNorm weights must match the input dtype."
        )
        assert weight.shape == (hidden_size,), (
            "RMSNorm weights must match the hidden dimension."
        )
        assert hidden_size % self.block_size == 0, (
            f"RMSNorm hidden size must be divisible by {self.block_size}."
        )
        flattened_input = hidden_states.reshape(-1, hidden_size)
        output = torch.empty_like(flattened_input)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(flattened_input.shape[0],)](
            flattened_input,
            weight,
            output,
            hidden_size,
            eps,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output.reshape_as(hidden_states), kernel

    def forward_torch(self, inputs):
        hidden_states, weight, eps = inputs
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized *= torch.rsqrt(variance + eps)
        return normalized.to(hidden_states.dtype) * weight
