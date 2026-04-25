from .add import AddOperator
from .fused import FancyFusedOperator
from .mse_loss import MSELossOperator
from .max_pooling_2d import MaxPooling2DOperator
from .matrix_scalar_multiplication import MatrixScalarMultiplicationOperator
from .relu import ReLUOperator
from .matmul import MatrixMultiplicationOperator
from .sigmoid import SigmoidOperator

__all__ = [
    "AddOperator",
    "FancyFusedOperator",
    "MSELossOperator",
    "MaxPooling2DOperator",
    "MatrixScalarMultiplicationOperator",
    "ReLUOperator",
    "MatrixMultiplicationOperator",
    "SigmoidOperator",
]


operator_list = [
    AddOperator,
    FancyFusedOperator,
    MSELossOperator,
    MaxPooling2DOperator,
    MatrixScalarMultiplicationOperator,
    ReLUOperator,
    MatrixMultiplicationOperator,
    SigmoidOperator,
]
