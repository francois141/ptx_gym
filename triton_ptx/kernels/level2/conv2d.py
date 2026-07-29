from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class Conv2dKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        in_channels=3,
        out_channels=96,
        kernel_size=11,
        stride=4,
        padding=2,
        bias=True,
        batch=8,
        height=64,
        width=64,
        block_size=128,
        num_warps=4,
        ptx=None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.batch = batch
        self.height = height
        self.width = width
        self.block_size = block_size
        self.constexpr_values = {
            "IN_CHANNELS": in_channels,
            "OUT_CHANNELS": out_channels,
            "KERNEL_SIZE": kernel_size,
            "STRIDE": stride,
            "PADDING": padding,
            "HAS_BIAS": bias,
            "BLOCK_SIZE": block_size,
        }
        self.num_warps = num_warps
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        ).cuda()
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total,
        height,
        width,
        out_h,
        out_w,
        IN_CHANNELS: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
        KERNEL_SIZE: tl.constexpr,
        STRIDE: tl.constexpr,
        PADDING: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        out_w_idx = offsets % out_w
        out_h_idx = (offsets // out_w) % out_h
        out_channel = (offsets // (out_h * out_w)) % OUT_CHANNELS
        batch = offsets // (OUT_CHANNELS * out_h * out_w)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for ic in range(0, IN_CHANNELS):
            for kh in range(0, KERNEL_SIZE):
                for kw in range(0, KERNEL_SIZE):
                    in_h = out_h_idx * STRIDE - PADDING + kh
                    in_w = out_w_idx * STRIDE - PADDING + kw
                    valid = (
                        (offsets < total)
                        & (in_h >= 0)
                        & (in_h < height)
                        & (in_w >= 0)
                        & (in_w < width)
                    )
                    x_idx = ((batch * IN_CHANNELS + ic) * height + in_h) * width + in_w
                    w_idx = (
                        (out_channel * IN_CHANNELS + ic) * KERNEL_SIZE + kh
                    ) * KERNEL_SIZE + kw
                    acc += tl.load(x_ptr + x_idx, mask=valid, other=0.0) * tl.load(
                        weight_ptr + w_idx, mask=offsets < total, other=0.0
                    )

        if HAS_BIAS:
            acc += tl.load(bias_ptr + out_channel, mask=offsets < total, other=0.0)

        tl.store(output_ptr + offsets, acc, mask=offsets < total)

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
        out_h = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty(
            (x.shape[0], self.out_channels, out_h, out_w),
            device=x.device,
            dtype=x.dtype,
        )
        bias = self.conv1.bias if self.conv1.bias is not None else self.conv1.weight
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
            self.conv1.weight,
            bias,
            output,
            total,
            x.shape[2],
            x.shape[3],
            out_h,
            out_w,
            IN_CHANNELS=self.in_channels,
            OUT_CHANNELS=self.out_channels,
            KERNEL_SIZE=self.kernel_size,
            STRIDE=self.stride,
            PADDING=self.padding,
            HAS_BIAS=self.conv1.bias is not None,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        return F.conv2d(
            inputs,
            self.conv1.weight,
            self.conv1.bias,
            stride=self.stride,
            padding=self.padding,
        )
