from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class LinearSigmoidSumKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        input_size=32,
        hidden_size=32,
        batch=128,
        block_m=32,
        block_n=32,
        block_k=32,
        num_warps=4,
        ptx=None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch = batch
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.constexpr_values = {
            "INPUT_SIZE": input_size,
            "HIDDEN_SIZE": hidden_size,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        }
        self.num_warps = num_warps
        self.linear = nn.Linear(input_size, hidden_size).cuda()
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        stride_xm,
        stride_wn,
        INPUT_SIZE: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        sum_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        offs_k = tl.arange(0, BLOCK_K)

        for n_start in range(0, HIDDEN_SIZE, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
            weight_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_start in range(0, tl.cdiv(INPUT_SIZE, BLOCK_K)):
                k_offsets = k_start * BLOCK_K + offs_k
                x = tl.load(
                    x_ptrs,
                    mask=(offs_m[:, None] < batch) & (k_offsets[None, :] < INPUT_SIZE),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptrs,
                    mask=(offs_n[None, :] < HIDDEN_SIZE) & (k_offsets[:, None] < INPUT_SIZE),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc=acc, out_dtype=tl.float32, input_precision="ieee")
                x_ptrs += BLOCK_K
                weight_ptrs += BLOCK_K

            bias = tl.load(bias_ptr + offs_n, mask=offs_n < HIDDEN_SIZE, other=0.0)
            acc += bias[None, :]
            acc = 1.0 / (1.0 + tl.exp(-acc))
            sum_acc += tl.sum(acc, axis=1)

        tl.store(output_ptr + offs_m, sum_acc, mask=offs_m < batch)

    def get_random_input(self):
        return torch.rand((self.batch, self.input_size), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        output = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(x.shape[0], self.block_m),)
        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            self.linear.weight,
            self.linear.bias,
            output,
            x.shape[0],
            x.stride(0),
            self.linear.weight.stride(0),
            INPUT_SIZE=self.input_size,
            HIDDEN_SIZE=self.hidden_size,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x = F.linear(inputs, self.linear.weight, self.linear.bias)
        x = torch.sigmoid(x)
        return torch.sum(x, dim=1, keepdim=True)
