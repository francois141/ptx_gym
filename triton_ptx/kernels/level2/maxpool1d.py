from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MaxPool1dKernel(TritonPTXKernel):
    def __init__(self, *, kernel_size=2, stride=2, padding=0, dilation=1, block_size=256, num_warps=4, ptx=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.block_size = block_size
        self.constexpr_values = {
            "KERNEL_SIZE": kernel_size,
            "STRIDE": stride,
            "PADDING": padding,
            "DILATION": dilation,
            "BLOCK_SIZE": block_size,
        }
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, indices_ptr, total, length, out_length, KERNEL_SIZE: tl.constexpr, STRIDE: tl.constexpr, PADDING: tl.constexpr, DILATION: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        out_l = offsets % out_length
        channel_batch = offsets // out_length
        base = channel_batch * length
        best = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        for k in range(0, KERNEL_SIZE):
            in_l = out_l * STRIDE - PADDING + k * DILATION
            valid = (offsets < total) & (in_l >= 0) & (in_l < length)
            value = tl.load(x_ptr + base + in_l, mask=valid, other=-float("inf"))
            update = value > best
            best = tl.where(update, value, best)
            best_idx = tl.where(update, in_l, best_idx)

        tl.store(output_ptr + offsets, best, mask=offsets < total)

    def get_random_input(self, size=128):
        length = max(8, min(int(size), 2048))
        return torch.rand((4, 4, length), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        out_length = (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output = torch.empty((x.shape[0], x.shape[1], out_length), device=x.device, dtype=x.dtype)
        indices = torch.empty(output.shape, device=x.device, dtype=torch.int64)
        total = output.numel()
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
        kwargs = dict(KERNEL_SIZE=self.kernel_size, STRIDE=self.stride, PADDING=self.padding, DILATION=self.dilation, BLOCK_SIZE=self.block_size)

        if not ptx:
            kernel = self.compiled_kernel[grid](x, output, indices, total, x.shape[2], out_length, **kwargs, num_warps=self.num_warps)
        else:
            kernel = self.require_compiled_ptx()[grid](x, output, indices, total, x.shape[2], out_length, **self.ptx_launch_kwargs(**kwargs, num_warps=self.num_warps))
        return output, kernel

    def forward_torch(self, inputs):
        return F.max_pool1d(
            inputs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )
