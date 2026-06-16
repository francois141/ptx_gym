from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class LinearMulLeakyReLUKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        in_features=32,
        out_features=32,
        multiplier=2.0,
        negative_slope=0.1,
        batch=1024,
        block_m=32,
        block_n=32,
        block_k=32,
        num_warps=4,
        ptx=None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.batch = batch
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.constexpr_values = {
            "IN_FEATURES": in_features,
            "OUT_FEATURES": out_features,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        }
        self.num_warps = num_warps
        self.gemm = nn.Linear(in_features, out_features).cuda()
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr, weight_ptr, bias_ptr, output_ptr, batch, multiplier, negative_slope, stride_xm, stride_wn, stride_om,
        IN_FEATURES: tl.constexpr, OUT_FEATURES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
        w_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(IN_FEATURES, BLOCK_K)):
            k_offsets = k_start * BLOCK_K + offs_k
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < batch) & (k_offsets[None, :] < IN_FEATURES), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_n[None, :] < OUT_FEATURES) & (k_offsets[:, None] < IN_FEATURES), other=0.0)
            acc = tl.dot(x, w, acc=acc, out_dtype=tl.float32, input_precision="ieee")
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=0.0)
        acc = (acc + bias[None, :]) * multiplier
        acc = tl.where(acc >= 0, acc, acc * negative_slope)
        out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < batch) & (offs_n[None, :] < OUT_FEATURES))

    def get_random_input(self):
        return torch.rand((self.batch, self.in_features), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        output = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_M"]), triton.cdiv(self.out_features, meta["BLOCK_N"]))
        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            self.gemm.weight,
            self.gemm.bias,
            output,
            x.shape[0],
            self.multiplier,
            self.negative_slope,
            x.stride(0),
            self.gemm.weight.stride(0),
            output.stride(0),
            IN_FEATURES=self.in_features,
            OUT_FEATURES=self.out_features,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x = F.linear(inputs, self.gemm.weight, self.gemm.bias)
        x = x * self.multiplier
        return F.leaky_relu(x, negative_slope=self.negative_slope)
