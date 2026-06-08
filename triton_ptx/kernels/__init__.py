from .add import AddKernel
from .argmax import ArgmaxKernel
from .argmin import ArgminKernel
from .cross_entropy_loss import CrossEntropyLossKernel
from .cumprod import CumprodKernel
from .cumsum import CumsumKernel
from .elu import ELUKernel
from .fused import FancyFusedKernel
from .gelu import GELUKernel
from .hard_sigmoid import HardSigmoidKernel
from .hardtanh import HardtanhKernel
from .hinge_loss import HingeLossKernel
from .kl_div_batchmean import KLDivBatchMeanKernel
from .l1_norm import L1NormKernel
from .l2_norm import L2NormKernel
from .leaky_relu import LeakyReLUKernel
from .log_softmax import LogSoftmaxKernel
from .masked_cumsum import MaskedCumsumKernel
from .matrix_multiplication import MatrixMultiplicationKernel
from .mean_squared_error import MeanSquaredErrorKernel
from .min_dim import MinDimKernel
from .mse_loss import MSELossKernel
from .reduce_sum import ReduceSumKernel
from .relu import ReLUKernel
from .relu_reduction import ReLUReductionKernel
from .reverse_cumsum import ReverseCumsumKernel
from .selu import SELUKernel
from .sigmoid import SigmoidKernel
from .softmax import SoftmaxKernel
from .softplus import SoftplusKernel
from .softsign import SoftsignKernel
from .smooth_l1_loss import SmoothL1LossKernel
from .sum_dim import SumDimKernel
from .swish import SwishKernel
from .tanh import TanhKernel
from .triplet_margin_loss import TripletMarginLossKernel

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
    "MatrixMultiplicationKernel",
    "MatrixScalarMultiplicationKernel",
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
    MatrixMultiplicationKernel,
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
