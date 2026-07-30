"""Float16 kernel variants, using tensor cores for matrix operations."""

from .dot_product_attention import DotProductAttentionFloat16Kernel
from .gelu import GELUFloat16Kernel
from .matrix_multiplication import MatrixMultiplicationFloat16
from .relu import ReLUFloat16Kernel
from .rms_norm import RMSNormFloat16Kernel
from .silu import SiLUFloat16Kernel
from .softmax import SoftmaxFloat16Kernel
from .swiglu import SwiGLUFloat16Kernel

MatrixMultiplicationFloat16Kernel = MatrixMultiplicationFloat16

__all__ = [
    "DotProductAttentionFloat16Kernel",
    "GELUFloat16Kernel",
    "MatrixMultiplicationFloat16Kernel",
    "ReLUFloat16Kernel",
    "RMSNormFloat16Kernel",
    "SiLUFloat16Kernel",
    "SoftmaxFloat16Kernel",
    "SwiGLUFloat16Kernel",
]
