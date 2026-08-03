"""Triton rotary-position-embedding kernel for Gemma."""

import torch
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class GemmaRoPEKernel(TritonPTXKernel):
    """Gemma local-attention rotary-position-embedding kernel."""

    def __init__(self, *, ptx=None):
        self.block_size = 256
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        input_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        sequence_length,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        feature_offsets = tl.arange(0, BLOCK_SIZE)
        half_head_dim = head_dim // 2
        is_first_half = feature_offsets < half_head_dim
        paired_offsets = tl.where(
            is_first_half,
            feature_offsets + half_head_dim,
            feature_offsets - half_head_dim,
        )
        input_offsets = row * head_dim + feature_offsets
        paired_values = tl.load(input_ptr + row * head_dim + paired_offsets)
        values = tl.load(input_ptr + input_offsets)
        rotated_values = tl.where(is_first_half, -paired_values, paired_values)
        position = row % sequence_length
        rope_offsets = position * head_dim + feature_offsets
        cos_values = tl.load(cos_ptr + rope_offsets)
        sin_values = tl.load(sin_ptr + rope_offsets)
        tl.store(
            output_ptr + input_offsets,
            values * cos_values + rotated_values * sin_values,
        )

    def get_random_input(self, fixed: bool = False):
        hidden_states = torch.rand(
            (1, 1, 16, 256), device="cuda", dtype=torch.bfloat16
        )
        frequencies = torch.rand((16, 256), device="cuda", dtype=torch.bfloat16)
        return hidden_states, frequencies, frequencies.clone()

    def get_shape_information(self) -> str:
        return (
            "- input_ptr: bfloat16 tensor with shape (1, 1, 16, 256)\n"
            "- cos_ptr: bfloat16 tensor with shape (16, 256)\n"
            "- sin_ptr: bfloat16 tensor with shape (16, 256)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 1, 16, 256)"
        )

    def forward_triton(self, inputs, ptx=False):
        hidden_states, cos, sin = inputs
        batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
        assert hidden_states.is_cuda, "RoPE requires CUDA input."
        assert hidden_states.is_contiguous(), "RoPE input must be contiguous."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
            "RoPE input must use float16 or bfloat16."
        )
        assert cos.is_cuda and sin.is_cuda, "RoPE frequencies must be CUDA tensors."
        assert (
            cos.device == hidden_states.device
            and sin.device == hidden_states.device
        ), (
            "RoPE frequencies must be on the input device."
        )
        assert cos.dtype == hidden_states.dtype and sin.dtype == hidden_states.dtype, (
            "RoPE frequencies must match the input dtype."
        )
        assert cos.is_contiguous() and sin.is_contiguous(), (
            "RoPE frequencies must be contiguous."
        )
        assert cos.shape == (sequence_length, head_dim), (
            f"RoPE cosine frequencies must have shape ({sequence_length}, {head_dim})."
        )
        assert sin.shape == (sequence_length, head_dim), (
            f"RoPE sine frequencies must have shape ({sequence_length}, {head_dim})."
        )
        assert head_dim % 2 == 0, "RoPE head dimension must be even."
        assert head_dim == self.block_size, (
            f"RoPE head dimension must equal {self.block_size}; received {head_dim}."
        )
        output = torch.empty_like(hidden_states)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(batch_size * num_heads * sequence_length,)](
            hidden_states,
            cos,
            sin,
            output,
            sequence_length=sequence_length,
            head_dim=head_dim,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        hidden_states, cos, sin = inputs
        half_head_dim = hidden_states.shape[-1] // 2
        rotated = torch.cat(
            (-hidden_states[..., half_head_dim:], hidden_states[..., :half_head_dim]),
            dim=-1,
        )
        return hidden_states * cos + rotated * sin
