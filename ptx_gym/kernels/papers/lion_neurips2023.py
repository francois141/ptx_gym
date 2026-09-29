"""
Fused Lion optimizer kernel from the NeurIPS 2023 optimizer.
https://github.com/lucidrains/lion-pytorch
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class LionNeurIPS2023Optimizer(TritonPTXKernel):
    """Apply one fused Lion parameter and momentum update."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_size": (128, 256, 512, 1024, 2048, 4096),
        "num_warps": (4, 8, 16, 32),
    }

    def __init__(self, *, ptx=None):
        self.num_elements = 1_048_576
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.block_size = 1024
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        p_ptr,
        grad_ptr,
        exp_avg_ptr,
        lr:  tl.constexpr,
        wd:  tl.constexpr,
        beta1:  tl.constexpr,
        beta2:  tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis = 0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # offsetted pointers
        offset_p_ptr = p_ptr + offsets
        offset_grad_ptr = grad_ptr + offsets
        offset_exp_avg_ptr = exp_avg_ptr + offsets

        # load
        p = tl.load(offset_p_ptr)
        grad = tl.load(offset_grad_ptr)
        exp_avg = tl.load(offset_exp_avg_ptr)

        # stepweight decay
        p = p * (1 - lr * wd)

        # diff between momentum running average and grad
        diff = exp_avg - grad

        # weight update
        update = diff * beta1 + grad

        # torch.sign
        can_update = update != 0
        update_sign = tl.where(update > 0, -lr, lr)

        p = p + update_sign * can_update

        # decay the momentum running average coefficient
        exp_avg = diff * beta2 + grad

        # store new params and momentum running average coefficient
        tl.store(offset_p_ptr, p)
        tl.store(offset_exp_avg_ptr, exp_avg)

    def get_random_input(self, fixed: bool = False):
        del fixed
        parameter = torch.randn(self.num_elements, device="cuda", dtype=torch.float32)
        gradient = torch.randn_like(parameter)
        exp_avg = torch.randn_like(parameter)
        return parameter, gradient, exp_avg

    def get_shape_information(self) -> str:
        return (
            f"- parameter_ptr: float32 tensor with shape ({self.num_elements},)\n"
            f"- gradient_ptr: float32 tensor with shape ({self.num_elements},)\n"
            f"- exp_avg_ptr: float32 tensor with shape ({self.num_elements},)"
        )

    def forward_triton(self, inputs, ptx=False):
        parameter, gradient, exp_avg = (tensor.clone() for tensor in inputs)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (triton.cdiv(parameter.numel(), meta["BLOCK_SIZE"]),)
        kernel = launch_kernel[grid](
            parameter,
            gradient,
            exp_avg,
            self.learning_rate,
            self.weight_decay,
            self.beta1,
            self.beta2,
            parameter.numel(),
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return (parameter, exp_avg), kernel

    def forward_torch(self, inputs):
        parameter, gradient, exp_avg = (tensor.clone() for tensor in inputs)
        parameter.mul_(1.0 - self.learning_rate * self.weight_decay)
        update = self.beta1 * exp_avg + (1.0 - self.beta1) * gradient
        parameter.add_(update.sign(), alpha=-self.learning_rate)
        exp_avg.mul_(self.beta2).add_(gradient, alpha=1.0 - self.beta2)
        return parameter, exp_avg
