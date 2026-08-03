"""Triton XIELU activation kernel for Apertus."""

import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class XIELUKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 1024
        self.num_warps = 4
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.init_compiled_kernels(ptx=ptx, autotune=True)

    @staticmethod
    def kernel(
        input_ptr,
        alpha_p_ptr,
        alpha_n_ptr,
        beta_ptr,
        eps_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block = tl.program_id(axis=0)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(input_ptr + offsets).to(tl.float32)
        alpha_p = tl.load(alpha_p_ptr).to(tl.float32)
        alpha_n = tl.load(alpha_n_ptr).to(tl.float32)
        beta = tl.load(beta_ptr).to(tl.float32)
        eps = tl.load(eps_ptr).to(tl.float32)
        positive_alpha = tl.log(1.0 + tl.exp(alpha_p))
        negative_alpha = beta + tl.log(1.0 + tl.exp(alpha_n))
        positive = positive_alpha * values * values + beta * values
        negative = (tl.exp(tl.minimum(values, eps)) - 1.0 - values) * negative_alpha
        negative += beta * values
        tl.store(output_ptr + offsets, tl.where(values > 0.0, positive, negative))

    def get_random_input(self, fixed: bool = False):
        return tuple(
            torch.rand(4096, device="cuda", dtype=torch.float16)
            if index == 0
            else torch.rand(1, device="cuda", dtype=torch.float16)
            for index in range(5)
        )

    def get_shape_information(self) -> str:
        return (
            "- input_ptr: float16 tensor with shape (4096,)\n"
            "- alpha_p_ptr, alpha_n_ptr, beta_ptr, eps_ptr: float16 scalar tensors\n"
            "- output_ptr: float16 tensor with shape (4096,)"
        )

    def forward_triton(self, inputs, ptx=False):
        hidden_states, alpha_p, alpha_n, beta, eps = inputs
        assert hidden_states.is_cuda, "XIELU requires CUDA input."
        assert hidden_states.is_contiguous(), "XIELU input must be contiguous."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
            "XIELU input must use float16 or bfloat16."
        )
        for name, value in (
            ("alpha_p", alpha_p),
            ("alpha_n", alpha_n),
            ("beta", beta),
            ("eps", eps),
        ):
            assert value.device == hidden_states.device, (
                f"{name} must be on the input device."
            )
            assert value.dtype == hidden_states.dtype, (
                f"{name} must match the input dtype."
            )
            assert value.numel() == 1, f"{name} must be a scalar tensor."
        assert hidden_states.numel() > 0, "XIELU input must not be empty."
        assert hidden_states.numel() % self.block_size == 0, (
            f"XIELU input size must be divisible by {self.block_size}; "
            f"received {hidden_states.numel()} elements."
        )
        flattened_input = hidden_states.reshape(-1)
        output = torch.empty_like(flattened_input)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[
            (triton.cdiv(flattened_input.numel(), self.block_size),)
        ](
            flattened_input,
            alpha_p,
            alpha_n,
            beta,
            eps,
            output,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output.reshape_as(hidden_states), kernel

    def forward_torch(self, inputs):
        hidden_states, alpha_p, alpha_n, beta, eps = inputs
        values = hidden_states.float()
        positive_alpha = torch.log1p(torch.exp(alpha_p.float()))
        negative_alpha = beta.float() + torch.log1p(torch.exp(alpha_n.float()))
        positive = positive_alpha * values.square() + beta.float() * values
        negative = torch.exp(torch.minimum(values, eps.float())) - 1.0 - values
        negative = negative * negative_alpha + beta.float() * values
        return torch.where(values > 0.0, positive, negative).to(hidden_states.dtype)
