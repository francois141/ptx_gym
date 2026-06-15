from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from triton_ptx.kernels.base import TritonPTXKernel


class LinearSwishDivideClampTanhClampKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        in_features=32,
        out_features=32,
        batch=32,
        bias=True,
        block_m=32,
        block_n=32,
        block_k=32,
        num_warps=4,
        ptx=None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch = batch
        self.bias = bias
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.constexpr_values = {
            "IN_FEATURES": in_features,
            "OUT_FEATURES": out_features,
            "HAS_BIAS": bias,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        }
        self.num_warps = num_warps
        self.gemm = nn.Linear(in_features, out_features, bias=bias).cuda()
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
        stride_om,
        IN_FEATURES: tl.constexpr,
        OUT_FEATURES: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
        weight_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(IN_FEATURES, BLOCK_K)):
            k_offsets = k_start * BLOCK_K + offs_k
            x = tl.load(
                x_ptrs,
                mask=(offs_m[:, None] < batch) & (k_offsets[None, :] < IN_FEATURES),
                other=0.0,
            )
            w = tl.load(
                weight_ptrs,
                mask=(offs_n[None, :] < OUT_FEATURES) & (k_offsets[:, None] < IN_FEATURES),
                other=0.0,
            )
            acc = tl.dot(x, w, acc=acc, out_dtype=tl.float32, input_precision="ieee")
            x_ptrs += BLOCK_K
            weight_ptrs += BLOCK_K

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_FEATURES, other=0.0)
            acc += bias[None, :]

        sig = 1.0 / (1.0 + tl.exp(-acc))
        acc = acc * sig
        acc = acc / 2.0
        acc = tl.maximum(tl.minimum(acc, 1.0), -1.0)
        acc = libdevice.tanh(acc)
        acc = tl.maximum(tl.minimum(acc, 1.0), -1.0)

        output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
        out_mask = (offs_m[:, None] < batch) & (offs_n[None, :] < OUT_FEATURES)
        tl.store(output_ptrs, acc, mask=out_mask)

    def get_random_input(self):
        return torch.rand((self.batch, self.in_features), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        output = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        bias = self.gemm.bias if self.gemm.bias is not None else self.gemm.weight
        grid = lambda meta: (
            triton.cdiv(x.shape[0], meta["BLOCK_M"]),
            triton.cdiv(self.out_features, meta["BLOCK_N"]),
        )
        launch_kwargs = dict(
            IN_FEATURES=self.in_features,
            OUT_FEATURES=self.out_features,
            HAS_BIAS=self.gemm.bias is not None,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
        )

        if not ptx:
            kernel = self.compiled_kernel[grid](
                x,
                self.gemm.weight,
                bias,
                output,
                x.shape[0],
                x.stride(0),
                self.gemm.weight.stride(0),
                output.stride(0),
                **launch_kwargs,
                num_warps=self.num_warps,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                x,
                self.gemm.weight,
                bias,
                output,
                x.shape[0],
                x.stride(0),
                self.gemm.weight.stride(0),
                output.stride(0),
                **self.ptx_launch_kwargs(**launch_kwargs, num_warps=self.num_warps),
            )

        return output, kernel

    def forward_torch(self, inputs):
        x = F.linear(inputs, self.gemm.weight, self.gemm.bias)
        x = x * torch.sigmoid(x)
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.tanh(x)
        return torch.clamp(x, min=-1.0, max=1.0)
