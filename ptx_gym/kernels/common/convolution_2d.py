from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin

CONVOLUTION_2D_SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "conv2d.spec"


class Convolution2DFloat16Kernel(TritonPTXKernel):
    input_element_width = 2
    tuning_options = {
        "block_m": (16, 32, 64, 128),
        "block_n": (16, 32, 64, 128),
        "block_k": (16, 32, 64),
        "num_warps": (4, 8, 16),
    }

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
        self.block_k = 64
        self.num_warps = 8
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
        input_base = (
            batch * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH
            + output_row * INPUT_WIDTH
            + output_col
        )
        # Keep spatial pixels contiguous in the right operand and the output.
        accumulator = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        weight_ptrs = weight_ptr + offsets_n[:, None] * K_DIM + offsets_k[None, :]

        for k_start in tl.range(0, K_DIM, BLOCK_K):
            k_offsets = k_start + offsets_k
            input_channel = k_offsets // 9
            kernel_offset = k_offsets % 9
            input_offsets = (
                input_channel * INPUT_HEIGHT * INPUT_WIDTH
                + (kernel_offset // 3) * INPUT_WIDTH
                + kernel_offset % 3
            )
            input_ptrs = x_ptr + input_offsets[:, None] + input_base[None, :]
            accumulator = tl.dot(
                tl.load(weight_ptrs),
                tl.load(input_ptrs, eviction_policy="evict_last"),
                acc=accumulator,
                out_dtype=tl.float32,
            )
            weight_ptrs += BLOCK_K

        output_ptrs = (
            output_ptr
            + batch[None, :] * OUTPUT_CHANNELS * output_pixels
            + offsets_n[:, None] * output_pixels
            + output_pixel[None, :]
        )
        tl.store(output_ptrs, accumulator.to(tl.float16))

    def get_random_input(self, fixed: bool = False):
        x = torch.randn(
            (
                self.batch_size,
                self.input_channels,
                self.input_height,
                self.input_width,
            ),
            device="cuda",
            dtype=torch.float16,
        )
        weight = torch.randn(
            (
                self.output_channels,
                self.input_channels,
                self.kernel_height,
                self.kernel_width,
            ),
            device="cuda",
            dtype=torch.float16,
        )
        output = torch.empty(
            (
                self.batch_size,
                self.output_channels,
                self.output_height,
                self.output_width,
            ),
            device="cuda",
            dtype=torch.float16,
        )
        return x, weight, output

    def get_shape_information(self) -> str:
        return (
            "- x_ptr: float16 tensor with shape (32, 64, 56, 56)\n"
            "- weight_ptr: float16 tensor with shape (128, 64, 3, 3)\n"
            "- output_ptr: float16 tensor with shape (32, 128, 54, 54)"
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
        return F.conv2d(x, weight)

    def volta_arguments(self):
        input_elements = (
            self.batch_size * self.input_channels * self.input_height * self.input_width
        )
        weight_elements = self.output_channels * self.k_dim
        output_elements = self.batch_size * self.output_channels * self.output_pixels
        return CONVOLUTION_2D_SPEC_PATH, [
            "-g",
            f"{triton.cdiv(self.batch_size * self.output_pixels, self.block_m)},"
            f"{triton.cdiv(self.output_channels, self.block_n)}",
            "--array",
            f"x_ptr:0x100000000:{self.input_element_width}:{input_elements}:in",
            "--array",
            f"weight_ptr:0x200000000:{self.input_element_width}:{weight_elements}:in",
            "--array",
            f"output_ptr:0x300000000:2:{output_elements}:out",
            "--param",
            "ptr:x_ptr",
            "--param",
            "ptr:weight_ptr",
            "--param",
            "ptr:output_ptr",
            "--param",
            "int:0",
            "--param",
            "int:0",
            "--dim",
            f"BATCH={self.batch_size}",
            "--dim",
            f"CIN={self.input_channels}",
            "--dim",
            f"IH={self.input_height}",
            "--dim",
            f"IW={self.input_width}",
            "--dim",
            f"COUT={self.output_channels}",
            "--dim",
            f"OH={self.output_height}",
            "--dim",
            f"OW={self.output_width}",
            "--dim",
            f"KH={self.kernel_height}",
            "--dim",
            f"KW={self.kernel_width}",
        ]


class Convolution2DFloat8Kernel(Float8KernelMixin, Convolution2DFloat16Kernel):
    input_element_width = 1
    autotune_tolerance = 1e-1
    verification_tolerance = 1e-1

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 2)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 2)

    def forward_torch(self, inputs):
        x, weight, _ = inputs
        return F.conv2d(x.to(torch.float32), weight.to(torch.float32)).to(torch.float16)
