"""Floating-point kernel variants, using tensor cores for matrix operations."""

from .add import AddFloat8Kernel, AddFloat16Kernel
from .convolution_2d import Convolution2DFloat8Kernel, Convolution2DFloat16Kernel
from .fused_gemm_add_gelu import (
    FusedGEMMAddGELUFloat8Kernel,
    FusedGEMMAddGELUFloat16Kernel,
)
from .fused_gemm_add_silu import (
    FusedGEMMAddSiLUFloat8Kernel,
    FusedGEMMAddSiLUFloat16Kernel,
)
from .gelu import GELUFloat8Kernel, GELUFloat16Kernel
from .matrix_multiplication import (
    MatrixMultiplicationFloat8,
    MatrixMultiplicationFloat16,
)
from .matrix_vector_multiplication import (
    MatrixVectorMultiplicationFloat8Kernel,
    MatrixVectorMultiplicationFloat16Kernel,
)
from .reduction_sum import ReductionSumFloat8Kernel, ReductionSumFloat16Kernel
from .relu import ReLUFloat8Kernel, ReLUFloat16Kernel
from .rms_norm import RMSNormFloat8Kernel, RMSNormFloat16Kernel
from .rope import RoPEFloat8Kernel, RoPEFloat16Kernel
from .sigmoid import SigmoidFloat8Kernel, SigmoidFloat16Kernel
from .silu import SiLUFloat8Kernel, SiLUFloat16Kernel
from .softmax import SoftmaxFloat8Kernel, SoftmaxFloat16Kernel
from .swiglu import SwiGLUFloat8Kernel, SwiGLUFloat16Kernel

MatrixMultiplicationFloat16Kernel = MatrixMultiplicationFloat16
MatrixMultiplicationFloat8Kernel = MatrixMultiplicationFloat8

__all__ = [
    "AddFloat8Kernel",
    "AddFloat16Kernel",
    "Convolution2DFloat8Kernel",
    "Convolution2DFloat16Kernel",
    "FusedGEMMAddGELUFloat8Kernel",
    "FusedGEMMAddGELUFloat16Kernel",
    "FusedGEMMAddSiLUFloat8Kernel",
    "FusedGEMMAddSiLUFloat16Kernel",
    "GELUFloat8Kernel",
    "GELUFloat16Kernel",
    "MatrixMultiplicationFloat8Kernel",
    "MatrixMultiplicationFloat16Kernel",
    "MatrixVectorMultiplicationFloat8Kernel",
    "MatrixVectorMultiplicationFloat16Kernel",
    "RMSNormFloat8Kernel",
    "RMSNormFloat16Kernel",
    "ReLUFloat8Kernel",
    "ReLUFloat16Kernel",
    "ReductionSumFloat8Kernel",
    "ReductionSumFloat16Kernel",
    "RoPEFloat8Kernel",
    "RoPEFloat16Kernel",
    "SiLUFloat8Kernel",
    "SiLUFloat16Kernel",
    "SigmoidFloat8Kernel",
    "SigmoidFloat16Kernel",
    "SoftmaxFloat8Kernel",
    "SoftmaxFloat16Kernel",
    "SwiGLUFloat8Kernel",
    "SwiGLUFloat16Kernel",
]
