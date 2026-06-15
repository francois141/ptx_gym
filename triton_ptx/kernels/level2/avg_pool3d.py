from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class AvgPool3dKernel(TritonPTXKernel):
    def __init__(self, *, kernel_size=2, stride=2, padding=0, block_size=256, num_warps=4, ptx=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.block_size = block_size
        self.constexpr_values = {
            "KERNEL_SIZE": kernel_size,
            "STRIDE": stride,
            "PADDING": padding,
            "BLOCK_SIZE": block_size,
        }
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, total, depth, height, width, out_d, out_h, out_w, KERNEL_SIZE: tl.constexpr, STRIDE: tl.constexpr, PADDING: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        out_w_idx = offsets % out_w
        out_h_idx = (offsets // out_w) % out_h
        out_d_idx = (offsets // (out_h * out_w)) % out_d
        channel_batch = offsets // (out_d * out_h * out_w)
        base = channel_batch * depth * height * width
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for kd in range(0, KERNEL_SIZE):
            for kh in range(0, KERNEL_SIZE):
                for kw in range(0, KERNEL_SIZE):
                    in_d = out_d_idx * STRIDE - PADDING + kd
                    in_h = out_h_idx * STRIDE - PADDING + kh
                    in_w = out_w_idx * STRIDE - PADDING + kw
                    valid = (offsets < total) & (in_d >= 0) & (in_d < depth) & (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
                    acc += tl.load(x_ptr + base + (in_d * height + in_h) * width + in_w, mask=valid, other=0.0)

        tl.store(output_ptr + offsets, acc / (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE), mask=offsets < total)

    def get_random_input(self, size=128):
        side = max(4, min(round(int(size) ** (1 / 3)), 32))
        return torch.rand((2, 4, side, side, side), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        out_d = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_h = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (x.shape[4] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty((x.shape[0], x.shape[1], out_d, out_h, out_w), device=x.device, dtype=x.dtype)
        total = output.numel()
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
        kwargs = dict(KERNEL_SIZE=self.kernel_size, STRIDE=self.stride, PADDING=self.padding, BLOCK_SIZE=self.block_size)

        if not ptx:
            kernel = self.compiled_kernel[grid](x, output, total, x.shape[2], x.shape[3], x.shape[4], out_d, out_h, out_w, **kwargs, num_warps=self.num_warps)
        else:
            kernel = self.require_compiled_ptx()[grid](x, output, total, x.shape[2], x.shape[3], x.shape[4], out_d, out_h, out_w, **self.ptx_launch_kwargs(**kwargs, num_warps=self.num_warps))
        return output, kernel

    def forward_torch(self, inputs):
        return F.avg_pool3d(inputs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
