from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from triton_ptx.kernels.base import TritonPTXKernel


class ConvTranspose2dSubTanhKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        in_channels=32,
        out_channels=32,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=1,
        batch=32,
        height=32,
        width=32,
        block_size=128,
        num_warps=4,
        ptx=None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.batch = batch
        self.height = height
        self.width = width
        self.block_size = block_size
        self.num_warps = num_warps
        self.constexpr_values = {
            "IN_CHANNELS": in_channels,
            "OUT_CHANNELS": out_channels,
            "KERNEL_SIZE": kernel_size,
            "STRIDE": stride,
            "PADDING": padding,
            "OUTPUT_PADDING": output_padding,
            "BLOCK_SIZE": block_size,
        }
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        ).cuda()
        self.bias = nn.Parameter(torch.randn((out_channels, 1, 1), device="cuda"))
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr, weight_ptr, conv_bias_ptr, bias_ptr, output_ptr, total, in_h, in_w, out_h, out_w,
        IN_CHANNELS: tl.constexpr, OUT_CHANNELS: tl.constexpr, KERNEL_SIZE: tl.constexpr,
        STRIDE: tl.constexpr, PADDING: tl.constexpr, OUTPUT_PADDING: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        out_x = offsets % out_w
        out_y = (offsets // out_w) % out_h
        out_c = (offsets // (out_h * out_w)) % OUT_CHANNELS
        batch_idx = offsets // (OUT_CHANNELS * out_h * out_w)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for in_c in range(0, IN_CHANNELS):
            for kh in range(0, KERNEL_SIZE):
                numer_h = out_y + PADDING - kh
                for kw in range(0, KERNEL_SIZE):
                    numer_w = out_x + PADDING - kw
                    valid_h = numer_h >= 0
                    valid_w = numer_w >= 0
                    div_h = numer_h // STRIDE
                    div_w = numer_w // STRIDE
                    exact_h = div_h * STRIDE == numer_h
                    exact_w = div_w * STRIDE == numer_w
                    valid = mask & valid_h & valid_w & exact_h & exact_w & (div_h < in_h) & (div_w < in_w)
                    x_idx = ((batch_idx * IN_CHANNELS + in_c) * in_h + div_h) * in_w + div_w
                    w_idx = ((in_c * OUT_CHANNELS + out_c) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw
                    acc += tl.load(x_ptr + x_idx, mask=valid, other=0.0) * tl.load(weight_ptr + w_idx, mask=mask, other=0.0)
        acc += tl.load(conv_bias_ptr + out_c, mask=mask, other=0.0)
        acc -= tl.load(bias_ptr + out_c, mask=mask, other=0.0)
        acc = libdevice.tanh(acc)
        tl.store(output_ptr + offsets, acc, mask=mask)

    def get_random_input(self):
        return torch.rand((self.batch, self.in_channels, self.height, self.width), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        out_h = (x.shape[2] - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_w = (x.shape[3] - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output = torch.empty((x.shape[0], self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
        total = output.numel()
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
        kwargs = dict(
            IN_CHANNELS=self.in_channels, OUT_CHANNELS=self.out_channels, KERNEL_SIZE=self.kernel_size,
            STRIDE=self.stride, PADDING=self.padding, OUTPUT_PADDING=self.output_padding, BLOCK_SIZE=self.block_size,
        )
        if not ptx:
            kernel = self.compiled_kernel[grid](
                x, self.conv_transpose.weight, self.conv_transpose.bias, self.bias, output, total,
                x.shape[2], x.shape[3], out_h, out_w, **kwargs, num_warps=self.num_warps,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                x, self.conv_transpose.weight, self.conv_transpose.bias, self.bias, output, total,
                x.shape[2], x.shape[3], out_h, out_w, **self.ptx_launch_kwargs(**kwargs, num_warps=self.num_warps),
            )
        return output, kernel

    def forward_torch(self, inputs):
        x = F.conv_transpose2d(
            inputs, self.conv_transpose.weight, self.conv_transpose.bias,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
        )
        x = x - self.bias
        return torch.tanh(x)
