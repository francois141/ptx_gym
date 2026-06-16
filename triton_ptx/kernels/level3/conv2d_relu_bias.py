from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class Conv2dReLUBiasKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        batch=32,
        height=128,
        width=128,
        block_size=128,
        num_warps=4,
        ptx=None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch = batch
        self.height = height
        self.width = width
        self.block_size = block_size
        self.num_warps = num_warps
        self.constexpr_values = {
            "IN_CHANNELS": in_channels,
            "OUT_CHANNELS": out_channels,
            "KERNEL_SIZE": kernel_size,
            "BLOCK_SIZE": block_size,
        }

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.bias = nn.Parameter(torch.randn((out_channels, 1, 1), device="cuda"))
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        weight_ptr,
        conv_bias_ptr,
        add_bias_ptr,
        output_ptr,
        total,
        height,
        width,
        out_h,
        out_w,
        IN_CHANNELS: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
        KERNEL_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total

        out_w_idx = offsets % out_w
        out_h_idx = (offsets // out_w) % out_h
        out_channel = (offsets // (out_h * out_w)) % OUT_CHANNELS
        batch_idx = offsets // (OUT_CHANNELS * out_h * out_w)

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for ic in range(0, IN_CHANNELS):
            for kh in range(0, KERNEL_SIZE):
                for kw in range(0, KERNEL_SIZE):
                    in_h = out_h_idx + kh
                    in_w = out_w_idx + kw
                    x_idx = ((batch_idx * IN_CHANNELS + ic) * height + in_h) * width + in_w
                    w_idx = ((out_channel * IN_CHANNELS + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw
                    acc += tl.load(x_ptr + x_idx, mask=mask, other=0.0) * tl.load(
                        weight_ptr + w_idx, mask=mask, other=0.0
                    )

        acc += tl.load(conv_bias_ptr + out_channel, mask=mask, other=0.0)
        acc = tl.maximum(acc, 0.0)
        acc += tl.load(add_bias_ptr + out_channel, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, acc, mask=mask)

    def get_random_input(self):
        return torch.rand(
            self.batch,
            self.in_channels,
            self.height,
            self.width,
            device="cuda",
            dtype=torch.float32,
        )

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        out_h = x.shape[2] - self.kernel_size + 1
        out_w = x.shape[3] - self.kernel_size + 1
        output = torch.empty((x.shape[0], self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
        total = output.numel()
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            self.conv.weight,
            self.conv.bias,
            self.bias,
            output,
            total,
            x.shape[2],
            x.shape[3],
            out_h,
            out_w,
            IN_CHANNELS=self.in_channels,
            OUT_CHANNELS=self.out_channels,
            KERNEL_SIZE=self.kernel_size,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x = F.conv2d(inputs, self.conv.weight, self.conv.bias)
        x = torch.relu(x)
        return x + self.bias
