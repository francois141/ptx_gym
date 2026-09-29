from typing import ClassVar

import torch
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin


class RoPEFloat16Kernel(TritonPTXKernel):
    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {"num_warps": (4, 8, 16)}

    def __init__(self, *, ptx=None):
        self.batch_size = 256
        self.num_heads = 64
        self.sequence_length = 256
        self.head_dim = 128
        self.block_size = self.head_dim
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        input_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        sequence_length: tl.constexpr,
        head_dim: tl.constexpr,
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
        values = tl.load(input_ptr + input_offsets).to(tl.float32)
        paired_values = tl.load(input_ptr + row * head_dim + paired_offsets).to(
            tl.float32
        )
        rotated_values = tl.where(is_first_half, -paired_values, paired_values)
        position = row % sequence_length
        rope_offsets = position * head_dim + feature_offsets
        cos_values = tl.load(cos_ptr + rope_offsets).to(tl.float32)
        sin_values = tl.load(sin_ptr + rope_offsets).to(tl.float32)
        tl.store(
            output_ptr + input_offsets,
            (values * cos_values + rotated_values * sin_values).to(tl.float16),
        )

    def get_random_input(self, fixed: bool = False):
        hidden_states = torch.rand(
            (
                self.batch_size,
                self.num_heads,
                self.sequence_length,
                self.head_dim,
            ),
            device="cuda",
            dtype=torch.float16,
        )
        cos = torch.rand(
            (self.sequence_length, self.head_dim), device="cuda", dtype=torch.float16
        )
        sin = torch.rand(
            (self.sequence_length, self.head_dim), device="cuda", dtype=torch.float16
        )
        return hidden_states, cos, sin

    def get_shape_information(self) -> str:
        return (
            "- input_ptr: float16 tensor with shape "
            f"({self.batch_size}, {self.num_heads}, {self.sequence_length}, "
            f"{self.head_dim})\n"
            f"- cos_ptr: float16 tensor with shape ({self.sequence_length}, "
            f"{self.head_dim})\n"
            f"- sin_ptr: float16 tensor with shape ({self.sequence_length}, "
            f"{self.head_dim})\n"
            "- output_ptr: float16 tensor with shape "
            f"({self.batch_size}, {self.num_heads}, {self.sequence_length}, "
            f"{self.head_dim})"
        )

    def forward_triton(self, inputs, ptx=False):
        hidden_states, cos, sin = inputs
        batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
        if head_dim != self.block_size:
            raise ValueError(
                f"RoPE head dimension must equal {self.block_size}; received {head_dim}."
            )
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even.")
        if cos.shape != (sequence_length, head_dim):
            raise ValueError(
                "RoPE cosine frequencies must have shape "
                f"({sequence_length}, {head_dim})."
            )
        if sin.shape != (sequence_length, head_dim):
            raise ValueError(
                "RoPE sine frequencies must have shape "
                f"({sequence_length}, {head_dim})."
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


class RoPEFloat8Kernel(Float8KernelMixin, RoPEFloat16Kernel):
    autotune_tolerance = 1e-2
    verification_tolerance = 1e-2

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 3)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 3)

    def forward_triton(self, inputs, ptx=False):
        hidden_states, cos, sin = inputs
        batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
        if head_dim != self.block_size:
            raise ValueError(
                f"RoPE head dimension must equal {self.block_size}; received {head_dim}."
            )
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even.")
        if cos.shape != (sequence_length, head_dim):
            raise ValueError(
                "RoPE cosine frequencies must have shape "
                f"({sequence_length}, {head_dim})."
            )
        if sin.shape != (sequence_length, head_dim):
            raise ValueError(
                "RoPE sine frequencies must have shape "
                f"({sequence_length}, {head_dim})."
            )
        output = torch.empty_like(hidden_states, dtype=torch.float16)
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
        hidden_states, cos, sin = self.float32_inputs(inputs)
        half_head_dim = hidden_states.shape[-1] // 2
        rotated = torch.cat(
            (-hidden_states[..., half_head_dim:], hidden_states[..., :half_head_dim]),
            dim=-1,
        )
        return (hidden_states * cos + rotated * sin).to(torch.float16)
