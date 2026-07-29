from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class AvgPool1dKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        kernel_size=2,
        stride=2,
        padding=0,
        block_size=256,
        num_warps=4,
        ptx=None,
    ):
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
    def kernel(
        x_ptr,
        output_ptr,
        total,
        length,
        out_length,
        KERNEL_SIZE: tl.constexpr,
        STRIDE: tl.constexpr,
        PADDING: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        out_l = offsets % out_length
        channel_batch = offsets // out_length
        base = channel_batch * length
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for k in range(0, KERNEL_SIZE):
            in_l = out_l * STRIDE - PADDING + k
            valid = (offsets < total) & (in_l >= 0) & (in_l < length)
            acc += tl.load(x_ptr + base + in_l, mask=valid, other=0.0)

        tl.store(output_ptr + offsets, acc / KERNEL_SIZE, mask=offsets < total)

    def get_random_input(self):
        size = 128
        length = max(8, min(int(size), 2048))
        return torch.rand((4, 4, length), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        out_length = (
            x.shape[2] + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        output = torch.empty(
            (x.shape[0], x.shape[1], out_length), device=x.device, dtype=x.dtype
        )
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
            total,
            x.shape[2],
            out_length,
            KERNEL_SIZE=self.kernel_size,
            STRIDE=self.stride,
            PADDING=self.padding,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        return F.avg_pool1d(
            inputs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
