from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class LinearScaleBatchNormKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        in_features=32,
        out_features=32,
        eps=1e-5,
        momentum=0.1,
        batch=32,
        block_n=32,
        block_k=32,
        num_warps=4,
        ptx=None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        self.batch = batch
        self.block_n = block_n
        self.block_k = block_k
        self.constexpr_values = {
            "IN_FEATURES": in_features,
            "OUT_FEATURES": out_features,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        }
        self.num_warps = num_warps
        self.gemm = nn.Linear(in_features, out_features).cuda()
        self.scale = nn.Parameter(torch.randn((out_features,), device="cuda"))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum).cuda()
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        scale_ptr,
        bn_weight_ptr,
        bn_bias_ptr,
        output_ptr,
        batch,
        stride_xm,
        stride_wn,
        stride_om,
        eps,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mean = tl.zeros((BLOCK_N,), dtype=tl.float32)
        sq_mean = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(0, batch):
            x_ptrs = x_ptr + m * stride_xm + offs_k
            w_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k_start in range(0, tl.cdiv(IN_FEATURES, BLOCK_K)):
                k_offsets = k_start * BLOCK_K + offs_k
                x = tl.load(x_ptrs, mask=k_offsets < IN_FEATURES, other=0.0)
                w = tl.load(
                    w_ptrs,
                    mask=(offs_n[None, :] < OUT_FEATURES)
                    & (k_offsets[:, None] < IN_FEATURES),
                    other=0.0,
                )
                acc += tl.sum(w * x[:, None], axis=0)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=0.0)
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=1.0)
            values = (acc + bias) * scale
            mean += values
            sq_mean += values * values
        mean = mean / batch
        var = sq_mean / batch - mean * mean
        gamma = tl.load(bn_weight_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=1.0)
        beta = tl.load(bn_bias_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=0.0)
        inv_std = 1.0 / tl.sqrt(var + eps)
        for m in range(0, batch):
            x_ptrs = x_ptr + m * stride_xm + offs_k
            w_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k_start in range(0, tl.cdiv(IN_FEATURES, BLOCK_K)):
                k_offsets = k_start * BLOCK_K + offs_k
                x = tl.load(x_ptrs, mask=k_offsets < IN_FEATURES, other=0.0)
                w = tl.load(
                    w_ptrs,
                    mask=(offs_n[None, :] < OUT_FEATURES)
                    & (k_offsets[:, None] < IN_FEATURES),
                    other=0.0,
                )
                acc += tl.sum(w * x[:, None], axis=0)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=0.0)
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=1.0)
            values = (acc + bias) * scale
            normed = (values - mean) * inv_std
            out = normed * gamma + beta
            out_ptrs = output_ptr + m * stride_om + offs_n
            tl.store(out_ptrs, out, mask=offs_n < OUT_FEATURES)

    def get_random_input(self):
        return torch.rand(
            (self.batch, self.in_features), device="cuda", dtype=torch.float32
        )

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        output = torch.empty(
            (x.shape[0], self.out_features), device=x.device, dtype=x.dtype
        )
        grid = (triton.cdiv(self.out_features, self.block_n),)
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
            self.scale,
            self.bn.weight,
            self.bn.bias,
            output,
            x.shape[0],
            x.stride(0),
            self.gemm.weight.stride(0),
            output.stride(0),
            self.eps,
            IN_FEATURES=self.in_features,
            OUT_FEATURES=self.out_features,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x = F.linear(inputs, self.gemm.weight, self.gemm.bias)
        x = x * self.scale
        return self.bn(x)
