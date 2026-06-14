from .level1 import AddKernel
from .level1 import ArgmaxKernel
from .level1 import ArgminKernel
from .level1 import CrossEntropyLossKernel
from .level1 import CumprodKernel
from .level1 import CumsumKernel
from .level1 import ELUKernel
from .level1 import FancyFusedKernel
from .level1 import GELUKernel
from .level1 import HardSigmoidKernel
from .level1 import HardtanhKernel
from .level1 import HingeLossKernel
from .level1 import KLDivBatchMeanKernel
from .level1 import L1NormKernel
from .level1 import L2NormKernel
from .level1 import LeakyReLUKernel
from .level1 import LogSoftmaxKernel
from .level1 import MaskedCumsumKernel
from .level1 import MeanSquaredErrorKernel
from .level1 import MinDimKernel
from .level1 import MSELossKernel
from .level1 import ReduceSumKernel
from .level1 import ReLUKernel
from .level1 import ReLUReductionKernel
from .level1 import ReverseCumsumKernel
from .level1 import SELUKernel
from .level1 import SigmoidKernel
from .level1 import SmoothL1LossKernel
from .level1 import SoftmaxKernel
from .level1 import SoftplusKernel
from .level1 import SoftsignKernel
from .level1 import SumDimKernel
from .level1 import SwishKernel
from .level1 import TanhKernel
from .level1 import TripletMarginLossKernel
from .level2 import MatrixAdditionKernel
from .level2 import MatrixMultiplicationKernel
from .level2 import MatrixScalarAdditionKernel

__all__ = [
    "AddKernel",
    "ArgmaxKernel",
    "ArgminKernel",
    "CrossEntropyLossKernel",
    "CumprodKernel",
    "CumsumKernel",
    "ELUKernel",
    "FancyFusedKernel",
    "GELUKernel",
    "HardSigmoidKernel",
    "HardtanhKernel",
    "HingeLossKernel",
    "KLDivBatchMeanKernel",
    "L1NormKernel",
    "L2NormKernel",
    "LeakyReLUKernel",
    "LogSoftmaxKernel",
    "MaskedCumsumKernel",
    "MatrixAdditionKernel",
    "MatrixMultiplicationKernel",
    "MatrixScalarAdditionKernel",
    "MeanSquaredErrorKernel",
    "MinDimKernel",
    "MSELossKernel",
    "ReLUKernel",
    "ReLUReductionKernel",
    "ReduceSumKernel",
    "ReverseCumsumKernel",
    "SELUKernel",
    "SigmoidKernel",
    "SoftmaxKernel",
    "SoftplusKernel",
    "SoftsignKernel",
    "SmoothL1LossKernel",
    "SumDimKernel",
    "SwishKernel",
    "TanhKernel",
    "TripletMarginLossKernel",
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
    ArgmaxKernel,
    ArgminKernel,
    CrossEntropyLossKernel,
    CumprodKernel,
    CumsumKernel,
    ELUKernel,
    FancyFusedKernel,
    GELUKernel,
    HardSigmoidKernel,
    HardtanhKernel,
    HingeLossKernel,
    KLDivBatchMeanKernel,
    L1NormKernel,
    L2NormKernel,
    LeakyReLUKernel,
    LogSoftmaxKernel,
    MaskedCumsumKernel,
    MatrixAdditionKernel,
    MatrixMultiplicationKernel,
    MatrixScalarAdditionKernel,
    MeanSquaredErrorKernel,
    MinDimKernel,
    MSELossKernel,
    ReduceSumKernel,
    ReLUKernel,
    ReLUReductionKernel,
    ReverseCumsumKernel,
    SELUKernel,
    SigmoidKernel,
    SoftmaxKernel,
    SoftplusKernel,
    SoftsignKernel,
    SmoothL1LossKernel,
    SumDimKernel,
    SwishKernel,
    TanhKernel,
    TripletMarginLossKernel,
]

test_kernel = AddKernel
