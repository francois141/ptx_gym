import math

import torch
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel

HEAD_DIM = 128
ROPE_THETA = 500_000
ROPE_FACTOR = 8.0
ROPE_LOW_FREQUENCY_FACTOR = 1.0
ROPE_HIGH_FREQUENCY_FACTOR = 4.0
ROPE_ORIGINAL_CONTEXT_LENGTH = 8_192
LOG_ROPE_THETA = tl.constexpr(math.log(ROPE_THETA))
KERNEL_TWO_PI = tl.constexpr(2.0 * math.pi)
KERNEL_ROPE_FACTOR = tl.constexpr(ROPE_FACTOR)
KERNEL_LOW_FREQUENCY_WAVELENGTH = tl.constexpr(
    ROPE_ORIGINAL_CONTEXT_LENGTH / ROPE_LOW_FREQUENCY_FACTOR
)
KERNEL_HIGH_FREQUENCY_WAVELENGTH = tl.constexpr(
    ROPE_ORIGINAL_CONTEXT_LENGTH / ROPE_HIGH_FREQUENCY_FACTOR
)
KERNEL_LOW_FREQUENCY_FACTOR = tl.constexpr(ROPE_LOW_FREQUENCY_FACTOR)
KERNEL_HIGH_FREQUENCY_FACTOR = tl.constexpr(ROPE_HIGH_FREQUENCY_FACTOR)
KERNEL_ORIGINAL_CONTEXT_LENGTH = tl.constexpr(ROPE_ORIGINAL_CONTEXT_LENGTH)


class MarinRoPEKernel(TritonPTXKernel):
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
        wavelength = KERNEL_TWO_PI / inverse_frequency
        scaled_frequency = tl.where(
            wavelength > KERNEL_LOW_FREQUENCY_WAVELENGTH,
            inverse_frequency / KERNEL_ROPE_FACTOR,
            inverse_frequency,
        )
        smooth_factor = (
            KERNEL_ORIGINAL_CONTEXT_LENGTH / wavelength - KERNEL_LOW_FREQUENCY_FACTOR
        ) / (KERNEL_HIGH_FREQUENCY_FACTOR - KERNEL_LOW_FREQUENCY_FACTOR)
        smoothed_frequency = (
            1.0 - smooth_factor
        ) * inverse_frequency / KERNEL_ROPE_FACTOR + smooth_factor * inverse_frequency
        is_medium_frequency = (wavelength >= KERNEL_HIGH_FREQUENCY_WAVELENGTH) & (
            wavelength <= KERNEL_LOW_FREQUENCY_WAVELENGTH
        )
        inverse_frequency = tl.where(
            is_medium_frequency,
            smoothed_frequency,
            scaled_frequency,
        )
        position = row % sequence_length
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

    def forward_triton(self, inputs, ptx=False):
        hidden_states = inputs
        batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
        assert hidden_states.is_cuda, "RoPE requires CUDA input."
        assert hidden_states.is_contiguous(), "RoPE input must be contiguous."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
            "RoPE input must use float16 or bfloat16."
        )
        assert head_dim == HEAD_DIM, f"RoPE head dimension must be {HEAD_DIM}."
        output = torch.empty_like(hidden_states)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(batch_size * num_heads * sequence_length,)](
            hidden_states,
            output,
            sequence_length,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        hidden_states = inputs
        sequence_length = hidden_states.shape[-2]
        inverse_frequency = 1.0 / (
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
        wavelength = 2 * math.pi / inverse_frequency
        low_frequency_wavelength = (
            ROPE_ORIGINAL_CONTEXT_LENGTH / ROPE_LOW_FREQUENCY_FACTOR
        )
        high_frequency_wavelength = (
            ROPE_ORIGINAL_CONTEXT_LENGTH / ROPE_HIGH_FREQUENCY_FACTOR
        )
        scaled_frequency = torch.where(
            wavelength > low_frequency_wavelength,
            inverse_frequency / ROPE_FACTOR,
            inverse_frequency,
        )
        smooth_factor = (
            ROPE_ORIGINAL_CONTEXT_LENGTH / wavelength - ROPE_LOW_FREQUENCY_FACTOR
        ) / (ROPE_HIGH_FREQUENCY_FACTOR - ROPE_LOW_FREQUENCY_FACTOR)
        smoothed_frequency = (
            1 - smooth_factor
        ) * inverse_frequency / ROPE_FACTOR + smooth_factor * inverse_frequency
        is_medium_frequency = (wavelength >= high_frequency_wavelength) & (
            wavelength <= low_frequency_wavelength
        )
        inverse_frequency = torch.where(
            is_medium_frequency,
            smoothed_frequency,
            scaled_frequency,
        )
        frequencies = torch.outer(
            torch.arange(
                sequence_length,
                device=hidden_states.device,
                dtype=torch.float32,
            ),
            inverse_frequency,
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
