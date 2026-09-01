"""Float16 kernel variants, using tensor cores for matrix operations."""

from .convolution_2d import Convolution2DFloat16Kernel
from .dot_product_attention import DotProductAttentionFloat16Kernel
from .flash_attention import FlashAttentionFloat16Kernel
from .fused_gemm_add_gelu import FusedGEMMAddGELUFloat16Kernel
from .gelu import GELUFloat16Kernel
from .matrix_multiplication import MatrixMultiplicationFloat16
from .matrix_vector_multiplication import MatrixVectorMultiplicationFloat16Kernel
from .reduction_sum import ReductionSumFloat16Kernel
from .relu import ReLUFloat16Kernel
from .rms_norm import RMSNormFloat16Kernel
from .silu import SiLUFloat16Kernel
from .softmax import SoftmaxFloat16Kernel
from .swiglu import SwiGLUFloat16Kernel

MatrixMultiplicationFloat16Kernel = MatrixMultiplicationFloat16

__all__ = [
    "Convolution2DFloat16Kernel",
    "DotProductAttentionFloat16Kernel",
    "FlashAttentionFloat16Kernel",
    "FusedGEMMAddGELUFloat16Kernel",
    "GELUFloat16Kernel",
    "MatrixMultiplicationFloat16Kernel",
    "MatrixVectorMultiplicationFloat16Kernel",
    "RMSNormFloat16Kernel",
    "ReLUFloat16Kernel",
    "ReductionSumFloat16Kernel",
    "SiLUFloat16Kernel",
    "SoftmaxFloat16Kernel",
    "SwiGLUFloat16Kernel",
]
