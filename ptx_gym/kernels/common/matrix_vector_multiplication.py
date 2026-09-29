from __future__ import annotations

from pathlib import Path

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel
from ptx_gym.kernels.common.float8 import Float8KernelMixin

GEMV_SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "gemv.spec"


class MatrixVectorMultiplicationFloat16Kernel(TritonPTXKernel):
    input_element_width = 2
    tuning_options = {
        "block_m": (32, 64, 128, 256),
        "block_k": (32, 64, 128, 256, 512, 1024),
        "num_warps": (4, 8, 16),
    }

    def __init__(self, *, ptx=None):
        self.matrix_rows = 4096
        self.matrix_cols = 4096
        self.block_m = 16
        self.block_k = 256
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        a_ptr,
        x_ptr,
        y_ptr,
        stride_am: tl.constexpr,
        MATRIX_COLS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_start in range(0, MATRIX_COLS, BLOCK_K):
            matrix = tl.load(
                a_ptr + offsets_m[:, None] * stride_am + k_start + offsets_k[None, :]
            )
            vector = tl.load(x_ptr + k_start + offsets_k)
            accumulator += tl.reshape(
                tl.dot(
                    matrix,
                    vector[:, None],
                    out_dtype=tl.float32,
                ),
                (BLOCK_M,),
            )

        tl.store(y_ptr + offsets_m, accumulator.to(tl.float16))

    def get_random_input(self, fixed: bool = False):
        a = torch.randn(
            (self.matrix_rows, self.matrix_cols),
            device="cuda",
            dtype=torch.float16,
        )
        x = torch.randn(self.matrix_cols, device="cuda", dtype=torch.float16)
        y = torch.empty(self.matrix_rows, device="cuda", dtype=torch.float16)
        return a, x, y

    def get_shape_information(self) -> str:
        return (
            f"- a_ptr: float16 tensor with shape "
            f"({self.matrix_rows}, {self.matrix_cols})\n"
            f"- x_ptr: float16 tensor with shape ({self.matrix_cols},)\n"
            f"- y_ptr: float16 tensor with shape ({self.matrix_rows},)"
        )

    def forward_triton(self, inputs, ptx=False):
        a, x, y = inputs
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (triton.cdiv(self.matrix_rows, meta["BLOCK_M"]),)
        kernel = launch_kernel[grid](
            a,
            x,
            y,
            a.stride(0),
            MATRIX_COLS=self.matrix_cols,
            BLOCK_M=self.block_m,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return y, kernel

    def forward_torch(self, inputs):
        a, x, _ = inputs
        return torch.mv(a, x)

    def volta_arguments(self):
        elements = self.matrix_rows * self.matrix_cols
        return GEMV_SPEC_PATH, [
            "-g",
            str(triton.cdiv(self.matrix_rows, self.block_m)),
            "--array",
            f"a:0x100000000:{self.input_element_width}:{elements}:in",
            "--array",
            f"x:0x200000000:{self.input_element_width}:{self.matrix_cols}:in",
            "--array",
            f"y:0x300000000:2:{self.matrix_rows}:out",
            "--param",
            "ptr:a",
            "--param",
            "ptr:x",
            "--param",
            "ptr:y",
            "--param",
            "int:0",
            "--param",
            "int:0",
            "--dim",
            f"M={self.matrix_rows}",
            "--dim",
            f"K={self.matrix_cols}",
        ]


class MatrixVectorMultiplicationFloat8Kernel(
    Float8KernelMixin, MatrixVectorMultiplicationFloat16Kernel
):
    input_element_width = 1
    autotune_tolerance = 1e-1
    verification_tolerance = 1e-1

    def get_random_input(self, fixed: bool = False):
        return self.float8_inputs(super().get_random_input(fixed), 2)

    def get_shape_information(self) -> str:
        return super().get_shape_information().replace("float16", "float8_e4m3fn", 2)

    def forward_torch(self, inputs):
        a, x, _ = inputs
        return torch.mv(a.to(torch.float32), x.to(torch.float32)).to(torch.float16)
