"""
Selective log-probability in-place backward kernel from Liger-Kernel.
https://github.com/linkedin/Liger-Kernel/pull/1548
https://github.com/linkedin/Liger-Kernel/pull/1549
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel

LOG2_E = tl.constexpr(1.4426950408889634)


class LigerSelectiveLogprobBackward(TritonPTXKernel):
    """Overwrite logits in place with the gradient of the selected logprobs."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_n": (1024, 2048, 4096),
        "num_warps": (4, 8, 16, 32),
    }
    verification_tolerance = 2e-2

    def __init__(self, *, ptx=None):
        self.num_rows = 4096
        self.vocab_size = 32000
        self.temperature = 0.8
        self.ignore_index = -100
        self.block_n = 4096
        self.num_warps = 32
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        LOGITS_ptr,
        LOGITS_stride,
        TARGETS_ptr,
        GRAD_LOGPROBS_ptr,
        BIAS_ptr,
        n_cols,
        temperature,
        HAS_BIAS: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        logits_row_ptr = LOGITS_ptr + row_idx * LOGITS_stride

        target = tl.load(TARGETS_ptr + row_idx)
        g = tl.load(GRAD_LOGPROBS_ptr + row_idx).to(tl.float32)

        is_invalid = (g == 0.0) | (target == IGNORE_INDEX) | (target < 0) | (target >= n_cols)
        if is_invalid:
            for offset in range(0, n_cols, BLOCK_N):
                cols = offset + tl.arange(0, BLOCK_N)
                tl.store(logits_row_ptr + cols, 0.0, mask=cols < n_cols)
            return

        inv_temp = 1.0 / temperature

        # Pass 1: compute m and d for online softmax
        m = float("-inf")
        d = 0.0
        for offset in range(0, n_cols, BLOCK_N):
            cols = offset + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            val = tl.load(logits_row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
            if HAS_BIAS:
                val += tl.load(BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            val = val * inv_temp
            new_m = tl.maximum(m, tl.max(val))
            d = d * tl.exp2((m - new_m) * LOG2_E) + tl.sum(tl.exp2((val - new_m) * LOG2_E))
            m = new_m

        scaled_g = g * inv_temp

        # Pre-read target logit to compute stable true-class gradient
        val_target = tl.load(logits_row_ptr + target).to(tl.float32)
        if HAS_BIAS:
            val_target += tl.load(BIAS_ptr + target).to(tl.float32)
        val_target = val_target * inv_temp

        # Pass 2: grad_logits = -scaled_g * softmax(val)
        for offset in range(0, n_cols, BLOCK_N):
            cols = offset + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            val = tl.load(logits_row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
            if HAS_BIAS:
                val += tl.load(BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            val = val * inv_temp
            prob = tl.exp2((val - m) * LOG2_E) / d
            grad_val = -scaled_g * prob
            tl.store(logits_row_ptr + cols, grad_val, mask=mask)

        tl.debug_barrier()

        # Stable true-class gradient: scaled_g * (1 - prob_target)
        prob_target = tl.exp2((val_target - m) * LOG2_E) / d
        dx_target = scaled_g * (1.0 - prob_target)
        tl.store(logits_row_ptr + target, dx_target)

    def get_random_input(self, fixed: bool = False):
        del fixed
        logits = torch.randn(
            (self.num_rows, self.vocab_size), device="cuda", dtype=torch.bfloat16
        )
        bias = torch.randn(self.vocab_size, device="cuda", dtype=torch.bfloat16)
        targets = torch.randint(
            0, self.vocab_size, (self.num_rows,), device="cuda", dtype=torch.int64
        )
        targets[torch.rand(self.num_rows, device="cuda") < 0.1] = self.ignore_index
        grad_logprobs = torch.randn(self.num_rows, device="cuda", dtype=torch.float32)
        return logits, targets, grad_logprobs, bias

    def get_shape_information(self) -> str:
        return (
            f"- LOGITS_ptr: bfloat16 tensor with shape "
            f"({self.num_rows}, {self.vocab_size}), overwritten in place with "
            "the logit gradients\n"
            f"- TARGETS_ptr: int64 tensor with shape ({self.num_rows},), "
            f"about 10% equal to {self.ignore_index}\n"
            f"- GRAD_LOGPROBS_ptr: float32 tensor with shape ({self.num_rows},)\n"
            f"- BIAS_ptr: bfloat16 tensor with shape ({self.vocab_size},)"
        )

    def forward_triton(self, inputs, ptx=False):
        logits, targets, grad_logprobs, bias = inputs
        grad_logits = logits.clone()
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(self.num_rows,)](
            grad_logits,
            grad_logits.stride(0),
            targets,
            grad_logprobs,
            bias,
            self.vocab_size,
            self.temperature,
            HAS_BIAS=True,
            IGNORE_INDEX=self.ignore_index,
            BLOCK_N=self.block_n,
            **launch_kwargs,
        )
        return grad_logits, kernel

    def forward_torch(self, inputs):
        logits, targets, grad_logprobs, bias = inputs
        scaled = (logits.float() + bias.float()) / self.temperature
        valid = targets != self.ignore_index
        scaled_grad = torch.where(valid, grad_logprobs, 0.0) / self.temperature
        grad_logits = -scaled_grad.unsqueeze(1) * torch.softmax(scaled, dim=-1)
        grad_logits.scatter_add_(
            1, targets.clamp(min=0).unsqueeze(1), scaled_grad.unsqueeze(1)
        )
        return grad_logits.to(logits.dtype)
