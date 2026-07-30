from .level1_float32 import (
    DotProductAttentionKernel,
    GELUKernel,
    MatrixMultiplicationKernel,
    ReLUKernel,
    RMSNormKernel,
    SiLUKernel,
    SoftmaxKernel,
    SwiGLUKernel,
)
from .level2_float16 import (
    DotProductAttentionFloat16Kernel,
    GELUFloat16Kernel,
    MatrixMultiplicationFloat16Kernel,
    ReLUFloat16Kernel,
    RMSNormFloat16Kernel,
    SiLUFloat16Kernel,
    SoftmaxFloat16Kernel,
    SwiGLUFloat16Kernel,
)

MatrixMultiplicationFloat16 = MatrixMultiplicationFloat16Kernel


def available_kernels() -> dict[str, type]:
    kernels = {operator.__name__: operator for operator in kernel_list}
    kernels["MatrixMultiplicationFloat16"] = MatrixMultiplicationFloat16
    return kernels


def resolve_kernel(name: str) -> type:
    kernels = available_kernels()
    if name not in kernels:
        from triton_ptx.LLMs.apertus.model import APERTUS_KERNEL_CLASSES

        kernels.update(
            {kernel.__name__: kernel for kernel in APERTUS_KERNEL_CLASSES.values()}
        )
    if name not in kernels:
        available = ", ".join(sorted(kernels))
        raise ValueError(f"Unknown kernel {name!r}. Available kernels: {available}")
    return kernels[name]


level1_float32_kernel_list = [
    DotProductAttentionKernel,
    GELUKernel,
    MatrixMultiplicationKernel,
    ReLUKernel,
    RMSNormKernel,
    SiLUKernel,
    SoftmaxKernel,
    SwiGLUKernel,
]

level2_float16_kernel_list = [
    DotProductAttentionFloat16Kernel,
    GELUFloat16Kernel,
    MatrixMultiplicationFloat16Kernel,
    ReLUFloat16Kernel,
    RMSNormFloat16Kernel,
    SiLUFloat16Kernel,
    SoftmaxFloat16Kernel,
    SwiGLUFloat16Kernel,
]

kernel_list = level1_float32_kernel_list + level2_float16_kernel_list
