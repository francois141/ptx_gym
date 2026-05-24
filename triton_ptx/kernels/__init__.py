from .add import AddKernel
from .fused import FancyFusedKernel
from .matmul import MatrixMultiplicationKernel
from .matrix_scalar_multiplication import MatrixScalarMultiplicationKernel
from .max_pooling_2d import MaxPooling2DKernel
from .mse_loss import MSELossKernel
from .reduce_sum import ReduceSumKernel
from .relu import ReLUKernel
from .relu_reduction import ReLUReductionKernel
from .sigmoid import SigmoidKernel

__all__ = [
    "AddKernel",
    "FancyFusedKernel",
    "MSELossKernel",
    "MatrixMultiplicationKernel",
    "MatrixScalarMultiplicationKernel",
    "MaxPooling2DKernel",
    "ReLUKernel",
    "ReLUReductionKernel",
    "ReduceSumKernel",
    "SigmoidKernel",
]

def available_kernels() -> dict[str, type]:
    return {operator.__name__: operator for operator in kernel_list}


def resolve_kernel(name: str) -> type:
    kernels = available_kernels()
    if name not in kernels:
        available = ", ".join(sorted(kernels))
        raise ValueError(f"Unknown kernel {name!r}. Available kernels: {available}")
    return kernels[name]


kernel_list = [
    AddKernel,
    FancyFusedKernel,
    MSELossKernel,
    MaxPooling2DKernel,
    MatrixScalarMultiplicationKernel,
    ReduceSumKernel,
    ReLUKernel,
    ReLUReductionKernel,
    MatrixMultiplicationKernel,
    SigmoidKernel,
]

test_kernel = AddKernel
