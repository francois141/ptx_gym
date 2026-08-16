from __future__ import annotations

import torch
import torch.nn.functional as F
import triton.language as tl

import triton
from triton_ptx.kernels.base import TritonPTXKernel


class Convolution2DKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.batch_size = 32
        self.input_channels = 64
        self.output_channels = 128
        self.input_height = 56
        self.input_width = 56
        self.kernel_height = 3
        self.kernel_width = 3
        self.output_height = self.input_height - self.kernel_height + 1
        self.output_width = self.input_width - self.kernel_width + 1
        self.output_pixels = self.output_height * self.output_width
        self.k_dim = self.input_channels * self.kernel_height * self.kernel_width
        self.block_m = 128
        self.block_n = 128
        self.block_k = 32
        self.constexpr_values = {
            "INPUT_CHANNELS": self.input_channels,
            "INPUT_HEIGHT": self.input_height,
            "INPUT_WIDTH": self.input_width,
            "OUTPUT_CHANNELS": self.output_channels,
            "OUTPUT_HEIGHT": self.output_height,
            "OUTPUT_WIDTH": self.output_width,
            "K_DIM": self.k_dim,
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
        }
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        INPUT_CHANNELS: tl.constexpr,
        INPUT_HEIGHT: tl.constexpr,
        INPUT_WIDTH: tl.constexpr,
        OUTPUT_CHANNELS: tl.constexpr,
        OUTPUT_HEIGHT: tl.constexpr,
        OUTPUT_WIDTH: tl.constexpr,
        K_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        output_pixels = OUTPUT_HEIGHT * OUTPUT_WIDTH
        batch = offsets_m // output_pixels
        output_pixel = offsets_m % output_pixels
        output_row = output_pixel // OUTPUT_WIDTH
        output_col = output_pixel % OUTPUT_WIDTH
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K_DIM, BLOCK_K):
            k_offsets = k_start + offsets_k
            input_channel = k_offsets // (3 * 3)
            kernel_offset = k_offsets % (3 * 3)
            kernel_row = kernel_offset // 3
            kernel_col = kernel_offset % 3
            input_ptrs = (
                x_ptr
                + batch[:, None] * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH
                + input_channel[None, :] * INPUT_HEIGHT * INPUT_WIDTH
                + (output_row[:, None] + kernel_row[None, :]) * INPUT_WIDTH
                + output_col[:, None]
                + kernel_col[None, :]
            )
            weight_ptrs = weight_ptr + offsets_n[None, :] * K_DIM + k_offsets[:, None]
            accumulator = tl.dot(
                tl.load(input_ptrs),
                tl.load(weight_ptrs),
                acc=accumulator,
                out_dtype=tl.float32,
                input_precision="ieee",
            )

        output_ptrs = (
            output_ptr
            + batch[:, None] * OUTPUT_CHANNELS * output_pixels
            + offsets_n[None, :] * output_pixels
            + output_pixel[:, None]
        )
        tl.store(output_ptrs, accumulator)

    def get_random_input(self, fixed: bool = False):
        x = torch.randn(
            (
                self.batch_size,
                self.input_channels,
                self.input_height,
                self.input_width,
            ),
            device="cuda",
            dtype=torch.float32,
        )
        weight = torch.randn(
            (
                self.output_channels,
                self.input_channels,
                self.kernel_height,
                self.kernel_width,
            ),
            device="cuda",
            dtype=torch.float32,
        )
        output = torch.empty(
            (
                self.batch_size,
                self.output_channels,
                self.output_height,
                self.output_width,
            ),
            device="cuda",
            dtype=torch.float32,
        )
        return x, weight, output

    def get_shape_information(self) -> str:
        return (
            "- x_ptr: float32 tensor with shape (32, 64, 56, 56)\n"
            "- weight_ptr: float32 tensor with shape (128, 64, 3, 3)\n"
            "- output_ptr: float32 tensor with shape (32, 128, 54, 54)"
        )

    def forward_triton(self, inputs, ptx=False):
        x, weight, output = inputs

        def grid(meta):
            return (
                triton.cdiv(self.batch_size * self.output_pixels, meta["BLOCK_M"]),
                triton.cdiv(self.output_channels, meta["BLOCK_N"]),
            )

        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[grid](
            x,
            weight,
            output,
            INPUT_CHANNELS=self.input_channels,
            INPUT_HEIGHT=self.input_height,
            INPUT_WIDTH=self.input_width,
            OUTPUT_CHANNELS=self.output_channels,
            OUTPUT_HEIGHT=self.output_height,
            OUTPUT_WIDTH=self.output_width,
            K_DIM=self.k_dim,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x, weight, _ = inputs
        allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            return F.conv2d(x, weight)
        finally:
            torch.backends.cudnn.allow_tf32 = allow_tf32
