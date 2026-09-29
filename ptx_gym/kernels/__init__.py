from .adversial import (
    AdversialFlashAttentionKernel,
    AdversialMatrixMultiplicationKernel,
    AdversialReLUKernel,
    MatrixMultiplicationFloat32,
)
from .common import (
    AddFloat8Kernel,
    AddFloat16Kernel,
    Convolution2DFloat8Kernel,
    Convolution2DFloat16Kernel,
    FusedGEMMAddGELUFloat8Kernel,
    FusedGEMMAddGELUFloat16Kernel,
    FusedGEMMAddSiLUFloat8Kernel,
    FusedGEMMAddSiLUFloat16Kernel,
    GELUFloat8Kernel,
    GELUFloat16Kernel,
    MatrixMultiplicationFloat8Kernel,
    MatrixMultiplicationFloat16Kernel,
    MatrixVectorMultiplicationFloat8Kernel,
    MatrixVectorMultiplicationFloat16Kernel,
    ReductionSumFloat8Kernel,
    ReductionSumFloat16Kernel,
    ReLUFloat8Kernel,
    ReLUFloat16Kernel,
    RMSNormFloat8Kernel,
    RMSNormFloat16Kernel,
    RoPEFloat8Kernel,
    RoPEFloat16Kernel,
    SigmoidFloat8Kernel,
    SigmoidFloat16Kernel,
    SiLUFloat8Kernel,
    SiLUFloat16Kernel,
    SoftmaxFloat8Kernel,
    SoftmaxFloat16Kernel,
    SwiGLUFloat8Kernel,
    SwiGLUFloat16Kernel,
)
from .papers.bitdelta_neurips2024 import (
    BitDeltaNeurIPS2024BatchedMatmul,
    BitDeltaNeurIPS2024Matmul,
)
from .papers.chanmix_iclr2026 import (
    ChanMixICLR2026GetCache,
    ChanMixICLR2026SetCache,
)
from .papers.dion2_triton import Dion2TritonPostOrthogonalize
from .papers.flash_attention_neurips2022 import (
    FlashAttentionNeurIPS2022Backward,
    FlashAttentionNeurIPS2022Forward,
)
from .papers.flashsinkhorn_icml2026 import FlashSinkhornFusedSchurMatvec
from .papers.forgetting_attention_iclr2025 import (
    ForgettingAttentionICLR2025Backward,
    ForgettingAttentionICLR2025Forward,
)
from .papers.householder_diagonalized_linear_attention_iclr2026 import (
    HouseholderDiagonalizedLinearAttentionICLR2026Backward,
    HouseholderDiagonalizedLinearAttentionICLR2026Forward,
)
from .papers.lion_neurips2023 import LionNeurIPS2023Optimizer
from .papers.mamba2 import Mamba2ChunkStateForward
from .papers.mamba2_chunk_scan import Mamba2ChunkScanForward
from .papers.mamba_iclr2026 import MambaICLR2026Forward, MambaICLR2026Step
from .papers.sageattention_iclr2025 import SageAttentionICLR2025
from .random import (
    RandomKernel1,
    RandomKernel2,
    RandomKernel3,
    RandomKernel4,
    RandomKernel5,
)

MatrixMultiplicationFloat16 = MatrixMultiplicationFloat16Kernel
MatrixMultiplicationFloat8 = MatrixMultiplicationFloat8Kernel


def available_kernels() -> dict[str, type]:
    kernels = {operator.__name__: operator for operator in kernel_list}
    kernels["MatrixMultiplicationFloat16"] = MatrixMultiplicationFloat16
    kernels["MatrixMultiplicationFloat8"] = MatrixMultiplicationFloat8
    kernels["MatrixMultiplicationFloat32"] = MatrixMultiplicationFloat32
    return kernels


def resolve_kernel(name: str) -> type:
    kernels = available_kernels()
    if name not in kernels:
        available = ", ".join(sorted(kernels))
        raise ValueError(f"Unknown kernel {name!r}. Available kernels: {available}")
    return kernels[name]


common_list = [
    AddFloat8Kernel,
    AddFloat16Kernel,
    Convolution2DFloat8Kernel,
    Convolution2DFloat16Kernel,
    FusedGEMMAddGELUFloat8Kernel,
    FusedGEMMAddGELUFloat16Kernel,
    FusedGEMMAddSiLUFloat8Kernel,
    FusedGEMMAddSiLUFloat16Kernel,
    GELUFloat8Kernel,
    GELUFloat16Kernel,
    MatrixMultiplicationFloat8Kernel,
    MatrixMultiplicationFloat16Kernel,
    MatrixVectorMultiplicationFloat8Kernel,
    MatrixVectorMultiplicationFloat16Kernel,
    ReLUFloat8Kernel,
    ReLUFloat16Kernel,
    ReductionSumFloat8Kernel,
    ReductionSumFloat16Kernel,
    RMSNormFloat8Kernel,
    RMSNormFloat16Kernel,
    RoPEFloat8Kernel,
    RoPEFloat16Kernel,
    SiLUFloat8Kernel,
    SiLUFloat16Kernel,
    SigmoidFloat8Kernel,
    SigmoidFloat16Kernel,
    SoftmaxFloat8Kernel,
    SoftmaxFloat16Kernel,
    SwiGLUFloat8Kernel,
    SwiGLUFloat16Kernel,
]

random_kernel_list = [
    RandomKernel1,
    RandomKernel2,
    RandomKernel3,
    RandomKernel4,
    RandomKernel5,
]

adversial_kernel_list = [
    AdversialFlashAttentionKernel,
    AdversialMatrixMultiplicationKernel,
    AdversialReLUKernel,
    MatrixMultiplicationFloat32,
]

paper_kernel_list = [
    BitDeltaNeurIPS2024Matmul,
    BitDeltaNeurIPS2024BatchedMatmul,
    ChanMixICLR2026SetCache,
    ChanMixICLR2026GetCache,
    Dion2TritonPostOrthogonalize,
    FlashSinkhornFusedSchurMatvec,
    FlashAttentionNeurIPS2022Forward,
    FlashAttentionNeurIPS2022Backward,
    ForgettingAttentionICLR2025Forward,
    ForgettingAttentionICLR2025Backward,
    HouseholderDiagonalizedLinearAttentionICLR2026Forward,
    HouseholderDiagonalizedLinearAttentionICLR2026Backward,
    LionNeurIPS2023Optimizer,
    Mamba2ChunkStateForward,
    Mamba2ChunkScanForward,
    MambaICLR2026Forward,
    MambaICLR2026Step,
    SageAttentionICLR2025,
]

kernel_list = (
    common_list + random_kernel_list + adversial_kernel_list + paper_kernel_list
)
