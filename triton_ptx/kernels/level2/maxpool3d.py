from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MaxPool3dKernel(TritonPTXKernel):
    def __init__(self, *, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, block_size=256, num_warps=4, ptx=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
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
    def kernel(x_ptr, output_ptr, indices_ptr, total, depth, height, width, out_d, out_h, out_w, KERNEL_SIZE: tl.constexpr, STRIDE: tl.constexpr, PADDING: tl.constexpr, DILATION: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        out_w_idx = offsets % out_w
        out_h_idx = (offsets // out_w) % out_h
        out_d_idx = (offsets // (out_h * out_w)) % out_d
        channel_batch = offsets // (out_d * out_h * out_w)
        base = channel_batch * depth * height * width
        best = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        for kd in range(0, KERNEL_SIZE):
            for kh in range(0, KERNEL_SIZE):
                for kw in range(0, KERNEL_SIZE):
                    in_d = out_d_idx * STRIDE - PADDING + kd * DILATION
                    in_h = out_h_idx * STRIDE - PADDING + kh * DILATION
                    in_w = out_w_idx * STRIDE - PADDING + kw * DILATION
                    valid = (offsets < total) & (in_d >= 0) & (in_d < depth) & (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
                    value = tl.load(x_ptr + base + (in_d * height + in_h) * width + in_w, mask=valid, other=-float("inf"))
                    update = value > best
                    best = tl.where(update, value, best)
                    best_idx = tl.where(update, (in_d * height + in_h) * width + in_w, best_idx)

        tl.store(output_ptr + offsets, best, mask=offsets < total)

    def get_random_input(self, size=128):
        side = max(4, min(round(int(size) ** (1 / 3)), 32))
        return torch.rand((2, 4, side, side, side), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        extra = self.stride - 1 if self.ceil_mode else 0
        out_d = (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1 + extra) // self.stride + 1
        out_h = (x.shape[3] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1 + extra) // self.stride + 1
        out_w = (x.shape[4] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1 + extra) // self.stride + 1
        output = torch.empty((x.shape[0], x.shape[1], out_d, out_h, out_w), device=x.device, dtype=x.dtype)
        indices = torch.empty(output.shape, device=x.device, dtype=torch.int64)
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
            output,
            indices,
            total,
            x.shape[2],
            x.shape[3],
            x.shape[4],
            out_d,
            out_h,
            out_w,
            KERNEL_SIZE=self.kernel_size,
            STRIDE=self.stride,
            PADDING=self.padding,
            DILATION=self.dilation,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        return F.max_pool3d(
            inputs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )
