from .add import AddOperator
from .fused import FancyFusedOperator
from .matmul import MatrixMultiplicationOperator
from .matrix_scalar_multiplication import MatrixScalarMultiplicationOperator
from .max_pooling_2d import MaxPooling2DOperator
from .mse_loss import MSELossOperator
from .reduce_sum import ReduceSumOperator
from .relu import ReLUOperator
from .relu_reduction import ReLUReductionOperator
from .sigmoid import SigmoidOperator

__all__ = [
    "AddOperator",
    "FancyFusedOperator",
    "MSELossOperator",
    "MatrixMultiplicationOperator",
    "MatrixScalarMultiplicationOperator",
    "MaxPooling2DOperator",
    "ReLUOperator",
    "ReLUReductionOperator",
    "ReduceSumOperator",
    "SigmoidOperator",
]

def available_operators() -> dict[str, type]:
    return {operator.__name__: operator for operator in operator_list}


def resolve_operator(name: str) -> type:
    operators = available_operators()
    if name not in operators:
        available = ", ".join(sorted(operators))
        raise ValueError(f"Unknown operator {name!r}. Available operators: {available}")
    return operators[name]


operator_list = [
    AddOperator,
    FancyFusedOperator,
    MSELossOperator,
    MaxPooling2DOperator,
    MatrixScalarMultiplicationOperator,
    ReduceSumOperator,
    ReLUOperator,
    ReLUReductionOperator,
    MatrixMultiplicationOperator,
    SigmoidOperator,
]

test_operator = AddOperator
