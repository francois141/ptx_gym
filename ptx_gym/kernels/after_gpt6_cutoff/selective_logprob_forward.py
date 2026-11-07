"""
Selective log-probability forward kernel from Liger-Kernel.
https://github.com/linkedin/Liger-Kernel/pull/1548
https://github.com/linkedin/Liger-Kernel/pull/1549
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel

LOG2_E = tl.constexpr(1.4426950408889634)


class LigerSelectiveLogprobForward(TritonPTXKernel):
    """Per-row log-softmax gathered at the target index, plus the row LSE."""

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
        LOGPROBS_ptr,
        LOG_Z_ptr,
        BIAS_ptr,
        n_cols,
        temperature,
        HAS_BIAS: tl.constexpr,
        STORE_LOG_Z: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        logits_row_ptr = LOGITS_ptr + row_idx * LOGITS_stride

        target = tl.load(TARGETS_ptr + row_idx)
        if (target == IGNORE_INDEX) | (target < 0) | (target >= n_cols):
            tl.store(LOGPROBS_ptr + row_idx, 0.0)
            if STORE_LOG_Z:
                tl.store(LOG_Z_ptr + row_idx, 0.0)
            return

        inv_temp = 1.0 / temperature

        # Pass 1: compute online softmax (max and sum-exp)
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

        lse = m + tl.log(d)
        if STORE_LOG_Z:
            tl.store(LOG_Z_ptr + row_idx, lse)

        target_logit = tl.load(logits_row_ptr + target).to(tl.float32)
        if HAS_BIAS:
            target_logit += tl.load(BIAS_ptr + target).to(tl.float32)
        target_logit = target_logit * inv_temp
        tl.store(LOGPROBS_ptr + row_idx, target_logit - lse)

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
        return logits, targets, bias

    def get_shape_information(self) -> str:
        return (
            f"- LOGITS_ptr: bfloat16 tensor with shape "
            f"({self.num_rows}, {self.vocab_size})\n"
            f"- TARGETS_ptr: int64 tensor with shape ({self.num_rows},), "
            f"about 10% equal to {self.ignore_index}\n"
            f"- LOGPROBS_ptr: float32 tensor with shape ({self.num_rows},)\n"
            f"- LOG_Z_ptr: float32 tensor with shape ({self.num_rows},)\n"
            f"- BIAS_ptr: bfloat16 tensor with shape ({self.vocab_size},)"
        )

    def forward_triton(self, inputs, ptx=False):
        logits, targets, bias = inputs
        logprobs = torch.empty(self.num_rows, device="cuda", dtype=torch.float32)
        log_z = torch.empty_like(logprobs)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(self.num_rows,)](
            logits,
            logits.stride(0),
            targets,
            logprobs,
            log_z,
            bias,
            self.vocab_size,
            self.temperature,
            HAS_BIAS=True,
            STORE_LOG_Z=True,
            IGNORE_INDEX=self.ignore_index,
            BLOCK_N=self.block_n,
            **launch_kwargs,
        )
        return (logprobs, log_z), kernel

    def forward_torch(self, inputs):
        logits, targets, bias = inputs
        scaled = (logits.float() + bias.float()) / self.temperature
        log_z = torch.logsumexp(scaled, dim=-1)
        valid = targets != self.ignore_index
        target_logits = scaled.gather(1, targets.clamp(min=0).unsqueeze(1)).squeeze(1)
        zero = torch.zeros_like(log_z)
        return (
            torch.where(valid, target_logits - log_z, zero),
            torch.where(valid, log_z, zero),
        )
