from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

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
    supporting_source: str
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
            f"Expected kernel function {kernel.__name__!r}, found {kernel_node.name!r}"
        )

    kernel_source = ast.get_source_segment(source, kernel_node)
    if kernel_source is None:
        raise ValueError(f"Unable to read kernel function source from {filename}")

    return kernel_node.name, kernel_source.strip(), parse_parameters(kernel)


def _extract_supporting_source(kernel: Callable[..., object], source: str) -> str:
    """Return module-level helper functions directly called by a kernel."""
    module = inspect.getmodule(kernel)
    if module is None:
        return ""

    source_tree = ast.parse(source)
    helper_names = {
        node.func.id
        for node in ast.walk(source_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    helper_sources = []
    for helper_name in sorted(helper_names):
        helper = getattr(module, helper_name, None)
        if helper is None or inspect.getmodule(helper) is not module:
            continue
        source_helper = getattr(helper, "fn", helper)
        try:
            helper_source = textwrap.dedent(inspect.getsource(source_helper)).strip()
        except (OSError, TypeError):
            continue
        helper_tree = ast.parse(helper_source)
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == helper_name
            for node in helper_tree.body
        ):
            helper_sources.append(helper_source)

    return "\n\n".join(helper_sources)


def extract_specification_from_operator(operator: object) -> KernelSpec:
    """Build a kernel specification from an operator class or instance."""
    if inspect.isclass(operator):
        operator = cast(type[object], operator)()

    operator_cls = operator.__class__
    kernel = getattr(operator, "kernel", None)
    if not callable(kernel):
        raise TypeError(f"Operator {operator_cls.__name__} has no callable kernel")

    kernel_name, kernel_source, parameters = _extract_kernel_function(kernel)

    constexpr_values = _extract_constexpr_values(operator, kernel)
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
        supporting_source=_extract_supporting_source(kernel, kernel_source),
        parameters=parameters,
        shape_information=shape_information.strip(),
        constexpr_values=(
            dict(constexpr_values) if isinstance(constexpr_values, dict) else {}
        ),
        num_warps=num_warps if isinstance(num_warps, int) else 4,
    )


class _ConstexprLaunchRecorder:
    def __init__(self, signature: inspect.Signature) -> None:
        self._signature = signature
        self.values: dict[str, object] = {}

    def __getitem__(self, _grid: object) -> _ConstexprLaunchRecorder:
        return self

    def __call__(self, *args: object, **kwargs: object) -> None:
        kernel_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in self._signature.parameters
        }
        bound_arguments = self._signature.bind_partial(*args, **kernel_kwargs)
        self.values.update(bound_arguments.arguments)


def _extract_constexpr_values(operator: object, kernel: object) -> dict[str, object]:
    """Capture the constexpr arguments used by the operator's fixed launch."""
    signature = inspect.signature(kernel)
    constexpr_names = {
        parameter.name
        for parameter in signature.parameters.values()
        if _is_constexpr_annotation(parameter.annotation)
    }
    declared_values = getattr(operator, "constexpr_values", {})
    values = dict(declared_values) if isinstance(declared_values, dict) else {}
    missing_names = constexpr_names.difference(values)
    if not missing_names:
        return values

    get_random_input = getattr(operator, "get_random_input", None)
    forward_triton = getattr(operator, "forward_triton", None)
    if not callable(get_random_input) or not callable(forward_triton):
        return values

    recorder = _ConstexprLaunchRecorder(signature)
    compiled_kernel = getattr(operator, "compiled_kernel", None)
    compiled_kernel_ptx = getattr(operator, "compiled_kernel_ptx", None)
    try:
        operator.compiled_kernel = recorder
        operator.compiled_kernel_ptx = recorder
        forward_triton(get_random_input(fixed=True))
    finally:
        operator.compiled_kernel = compiled_kernel
        operator.compiled_kernel_ptx = compiled_kernel_ptx

    values.update(
        (name, recorder.values[name])
        for name in constexpr_names
        if name in recorder.values
    )
    operator.constexpr_values = values
    return values


def _is_constexpr_annotation(annotation: object) -> bool:
    annotation_text = str(annotation).lower()
    return (
        annotation_text == "constexpr"
        or annotation_text.endswith(".constexpr")
        or "triton.language.core.constexpr" in annotation_text
        or ("triton.language" in annotation_text and "constexpr" in annotation_text)
    )
