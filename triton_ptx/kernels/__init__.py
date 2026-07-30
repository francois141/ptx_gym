from .dot_product_attention import DotProductAttentionKernel
from .gelu import GELUKernel
from .matrix_multiplication import MatrixMultiplicationKernel
from .matrix_multiplication_float16 import MatrixMultiplicationFloat16
from .relu import ReLUKernel
from .rms_norm import RMSNormKernel
from .silu import SiLUKernel
from .softmax import SoftmaxKernel
from .swiglu import SwiGLUKernel


def available_kernels() -> dict[str, type]:
    return {operator.__name__: operator for operator in kernel_list}


def resolve_kernel(name: str) -> type:
    kernels = available_kernels()
    if name not in kernels:
        available = ", ".join(sorted(kernels))
        raise ValueError(f"Unknown kernel {name!r}. Available kernels: {available}")
    return kernels[name]


kernel_list = [
    DotProductAttentionKernel,
    GELUKernel,
    MatrixMultiplicationFloat16,
    MatrixMultiplicationKernel,
    ReLUKernel,
    RMSNormKernel,
    SiLUKernel,
    SoftmaxKernel,
    SwiGLUKernel,
]
