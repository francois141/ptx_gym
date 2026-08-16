import math

import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel

HEAD_DIM = 128
ROPE_THETA = 1_000_000
LOG_ROPE_THETA = tl.constexpr(math.log(ROPE_THETA))


class QwenRoPEKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = HEAD_DIM
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        input_ptr,
        output_ptr,
        sequence_length,
        position_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        feature_offsets = tl.arange(0, BLOCK_SIZE)
        half_head_dim = BLOCK_SIZE // 2
        first_half = feature_offsets < half_head_dim
        paired_offsets = tl.where(
            first_half,
            feature_offsets + half_head_dim,
            feature_offsets - half_head_dim,
        )
        input_offsets = row * BLOCK_SIZE + feature_offsets
        values = tl.load(input_ptr + input_offsets).to(tl.float32)
        paired_values = tl.load(input_ptr + row * BLOCK_SIZE + paired_offsets).to(
            tl.float32
        )
        rotated = tl.where(first_half, -paired_values, paired_values)
        frequency_index = feature_offsets % half_head_dim
        inverse_frequency = tl.exp(
            -LOG_ROPE_THETA * (2.0 * frequency_index.to(tl.float32) / BLOCK_SIZE)
        )
        position = row % sequence_length + position_offset
        angle = position.to(tl.float32) * inverse_frequency
        tl.store(
            output_ptr + input_offsets,
            values * tl.cos(angle) + rotated * tl.sin(angle),
        )

    def get_random_input(self, fixed=False):
        return torch.rand(
            (1, 32, 1, HEAD_DIM),
            device="cuda",
            dtype=torch.bfloat16,
        )

    def get_shape_information(self):
        return (
            "- input_ptr: bfloat16 tensor with shape "
            "(1, 32, 1, 128)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 32, 1, 128)"
        )

    def forward_triton(self, inputs, ptx=False, position_offset=0):
        hidden_states = inputs
        batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
        assert hidden_states.is_cuda, "RoPE requires CUDA input."
        assert hidden_states.is_contiguous(), "RoPE input must be contiguous."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
            "RoPE input must use float16 or bfloat16."
        )
        assert head_dim == HEAD_DIM, f"RoPE head dimension must be {HEAD_DIM}."
        assert position_offset >= 0, "RoPE position offset must be non-negative."
        output = torch.empty_like(hidden_states)
        use_custom_ptx = ptx and position_offset == 0
        launch_kernel = (
            self.compiled_kernel_ptx if use_custom_ptx else self.compiled_kernel
        )
        launch_kwargs = (
            self.ptx_launch_kwargs()
            if use_custom_ptx
            else {"num_warps": self.num_warps}
        )
        kernel_args = (hidden_states, output, sequence_length)
        if not use_custom_ptx:
            kernel_args += (position_offset,)
        kernel = launch_kernel[(batch_size * num_heads * sequence_length,)](
            *kernel_args,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs, position_offset=0):
        hidden_states = inputs
        sequence_length = hidden_states.shape[-2]
        inverse_frequencies = 1.0 / (
            ROPE_THETA
            ** (
                torch.arange(
                    0,
                    HEAD_DIM,
                    2,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                / HEAD_DIM
            )
        )
        frequencies = torch.outer(
            torch.arange(
                sequence_length,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            + position_offset,
            inverse_frequencies,
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)
        half_head_dim = HEAD_DIM // 2
        rotated = torch.cat(
            (
                -hidden_states[..., half_head_dim:],
                hidden_states[..., :half_head_dim],
            ),
            dim=-1,
        )
        cosine = angles.cos().to(hidden_states.dtype)
        sine = angles.sin().to(hidden_states.dtype)
        return hidden_states * cosine + rotated * sine
