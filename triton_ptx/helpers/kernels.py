from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Callable, cast

EMPTY = inspect.Parameter.empty

PTX_LAUNCH_KEYS: frozenset[str] = frozenset(
    {
        "num_threads_x",
        "num_threads_y",
        "num_threads_z",
    }
)

def get_ptx_code(ptx: object) -> object | None:
    """Return PTX source from a payload mapping or a raw PTX value."""
    if isinstance(ptx, dict):
        return ptx.get("ptx")
    return ptx


def has_ptx_code(ptx: object) -> bool:
    """Return whether a PTX value or payload contains PTX source."""
    return get_ptx_code(ptx) is not None



@dataclass(frozen=True)
class KernelParameter:
    name: str
    annotation: object = EMPTY
    default: object = EMPTY
    kind: object = inspect.Parameter.POSITIONAL_OR_KEYWORD


@dataclass(frozen=True)
class KernelSpec:
    operator_name: str
    kernel_name: str
    source: str
    parameters: tuple[KernelParameter, ...]
    shape_information: str
    constexpr_values: dict[str, object] = field(default_factory=dict)
    num_warps: int = 4


def is_constexpr_annotation(annotation: object) -> bool:
    """Return whether an annotation denotes a Triton compile-time constant."""
    annotation_text = str(annotation).lower()
    return (
        annotation_text == "constexpr"
        or annotation_text.endswith(".constexpr")
        or "triton.language.core.constexpr" in annotation_text
        or ("triton.language" in annotation_text and "constexpr" in annotation_text)
    )


def parse_parameters(kernel: Callable[..., object]) -> tuple[KernelParameter, ...]:
    """Extract non-variadic parameters from a kernel callable."""
    parameters = tuple(inspect.signature(kernel).parameters.values())
    variadic_parameters = tuple(
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    if variadic_parameters:
        names = ", ".join(variadic_parameters)
        raise ValueError(
            f"Kernel {kernel.__qualname__} cannot use *args or **kwargs: {names}"
        )

    return tuple(
        KernelParameter(
            name=parameter.name,
            annotation=parameter.annotation,
            default=parameter.default,
            kind=parameter.kind,
        )
        for parameter in parameters
    )


def _extract_kernel_function(
    kernel: Callable[..., object],
) -> tuple[str, str, tuple[KernelParameter, ...]]:
    """Extract a kernel's name, undecorated source, and parameters."""
    source = textwrap.dedent(inspect.getsource(kernel))
    filename = inspect.getsourcefile(kernel) or kernel.__name__
    module = ast.parse(source, filename=filename)
    function_nodes = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if len(function_nodes) != 1:
        raise ValueError(
            f"Expected exactly one kernel function in {filename}, "
            f"found {len(function_nodes)}"
        )

    kernel_node = function_nodes[0]
    if kernel_node.name != kernel.__name__:
        raise ValueError(
            f"Expected kernel function {kernel.__name__!r}, "
            f"found {kernel_node.name!r}"
        )

    kernel_source = ast.get_source_segment(source, kernel_node)
    if kernel_source is None:
        raise ValueError(f"Unable to read kernel function source from {filename}")

    return kernel_node.name, kernel_source.strip(), parse_parameters(kernel)


def extract_specification_from_operator(operator: object) -> KernelSpec:
    """Build a kernel specification from an operator class or instance."""
    if inspect.isclass(operator):
        operator = cast(type[object], operator)()

    operator_cls = operator.__class__
    kernel = getattr(operator, "kernel", None)
    if not callable(kernel):
        raise TypeError(f"Operator {operator_cls.__name__} has no callable kernel")

    kernel_name, kernel_source, parameters = _extract_kernel_function(kernel)

    constexpr_values = getattr(operator, "constexpr_values", None)
    num_warps = getattr(operator, "num_warps", None)
    get_shape_information = getattr(operator, "get_shape_information", None)
    if not callable(get_shape_information):
        raise TypeError(
            f"Operator {operator_cls.__name__} has no callable get_shape_information"
        )
    shape_information = get_shape_information()
    if not isinstance(shape_information, str) or not shape_information.strip():
        raise ValueError(
            f"Operator {operator_cls.__name__} returned invalid shape information"
        )

    return KernelSpec(
        operator_name=operator_cls.__name__,
        kernel_name=kernel_name,
        source=kernel_source,
        parameters=parameters,
        shape_information=shape_information.strip(),
        constexpr_values=(
            dict(constexpr_values) if isinstance(constexpr_values, dict) else {}
        ),
        num_warps=num_warps if isinstance(num_warps, int) else 4,
    )
