from .level1_float32 import (
    Convolution2DKernel,
    DotProductAttentionKernel,
    FusedGEMMAddGELUKernel,
    GELUKernel,
    MatrixMultiplicationKernel,
    MatrixVectorMultiplicationKernel,
    ReductionSumKernel,
    ReLUKernel,
    RMSNormKernel,
    SiLUKernel,
    SoftmaxKernel,
    SwiGLUKernel,
)
from .level2_float16 import (
    Convolution2DFloat16Kernel,
    DotProductAttentionFloat16Kernel,
    FlashAttentionFloat16Kernel,
    FusedGEMMAddGELUFloat16Kernel,
    GELUFloat16Kernel,
    MatrixMultiplicationFloat16Kernel,
    MatrixVectorMultiplicationFloat16Kernel,
    ReductionSumFloat16Kernel,
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
        from triton_ptx.LLMs.gemma.model import GEMMA_KERNEL_CLASSES
        from triton_ptx.LLMs.marin.model import MARIN_KERNEL_CLASSES
        from triton_ptx.LLMs.qwen.model import QWEN_KERNEL_CLASSES

        kernels.update(
            {kernel.__name__: kernel for kernel in APERTUS_KERNEL_CLASSES.values()}
        )
        kernels.update(
            {kernel.__name__: kernel for kernel in GEMMA_KERNEL_CLASSES.values()}
        )
        kernels.update(
            {kernel.__name__: kernel for kernel in MARIN_KERNEL_CLASSES.values()}
        )
        kernels.update(
            {kernel.__name__: kernel for kernel in QWEN_KERNEL_CLASSES.values()}
        )
    if name not in kernels:
        available = ", ".join(sorted(kernels))
        raise ValueError(f"Unknown kernel {name!r}. Available kernels: {available}")
    return kernels[name]


level1_float32_kernel_list = [
    Convolution2DKernel,
    DotProductAttentionKernel,
    FusedGEMMAddGELUKernel,
    GELUKernel,
    MatrixMultiplicationKernel,
    MatrixVectorMultiplicationKernel,
    ReLUKernel,
    ReductionSumKernel,
    RMSNormKernel,
    SiLUKernel,
    SoftmaxKernel,
    SwiGLUKernel,
]

level2_float16_kernel_list = [
    Convolution2DFloat16Kernel,
    DotProductAttentionFloat16Kernel,
    FlashAttentionFloat16Kernel,
    FusedGEMMAddGELUFloat16Kernel,
    GELUFloat16Kernel,
    MatrixMultiplicationFloat16Kernel,
    MatrixVectorMultiplicationFloat16Kernel,
    ReLUFloat16Kernel,
    ReductionSumFloat16Kernel,
    RMSNormFloat16Kernel,
    SiLUFloat16Kernel,
    SoftmaxFloat16Kernel,
    SwiGLUFloat16Kernel,
]

kernel_list = level1_float32_kernel_list + level2_float16_kernel_list
