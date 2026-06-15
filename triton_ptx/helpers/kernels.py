import ast
import importlib.util
import inspect
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path

EMPTY = inspect.Parameter.empty
BASE_DIR = Path(__file__).resolve().parent

PTX_LAUNCH_KEYS = frozenset(
    {
        "num_threads_x",
        "num_threads_y",
        "num_threads_z",
    }
)



def get_ptx_code(ptx):
    if isinstance(ptx, dict):
        return ptx.get("ptx")
    return ptx

def has_ptx_code(ptx):
    return get_ptx_code(ptx) is not None

def get_ptx_extra_payload_keys(ptx):
    if not isinstance(ptx, dict):
        return {}
    return {
        key: value
        for key, value in ptx.items()
        if key != "ptx" and key not in PTX_LAUNCH_KEYS
    }

def get_ptx_constexprs(ptx):
    return get_ptx_extra_payload_keys(ptx)

def get_ptx_constexpr(ptx, name, default=None):
    if isinstance(ptx, dict) and name not in ptx:
        return default
    return get_ptx_extra_payload_keys(ptx).get(name, default)


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
    constexpr_values: dict[str, object] = field(default_factory=dict)
    num_warps: int = 4


def is_constexpr_annotation(annotation) -> bool:
    annotation_text = str(annotation).lower()
    return (
        annotation_text == "constexpr"
        or annotation_text.endswith(".constexpr")
        or "triton.language.core.constexpr" in annotation_text
        or ("triton.language" in annotation_text and "constexpr" in annotation_text)
    )

def parse_parameters(func, source):
    params = []
    args = func.args

    positional_args = [*args.posonlyargs, *args.args]
    defaults = [EMPTY] * (len(positional_args) - len(args.defaults)) + args.defaults

    for arg, default_node in zip(positional_args, defaults):
        annotation = ast.get_source_segment(source, arg.annotation) if arg.annotation else EMPTY

        try:
            default = ast.literal_eval(default_node) if default_node is not EMPTY else EMPTY
        except Exception:
            default = EMPTY

        params.append(
            KernelParameter(
                name=arg.arg,
                annotation=annotation.strip() if isinstance(annotation, str) else annotation,
                default=default,
            )
        )

    if args.vararg:
        annotation = ast.get_source_segment(source, args.vararg.annotation) if args.vararg.annotation else EMPTY

        params.append(
            KernelParameter(
                name=args.vararg.arg,
                annotation=annotation.strip() if isinstance(annotation, str) else annotation,
                kind=inspect.Parameter.VAR_POSITIONAL,
            )
        )

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        annotation = ast.get_source_segment(source, arg.annotation) if arg.annotation else EMPTY

        try:
            default = ast.literal_eval(default_node) if default_node else EMPTY
        except Exception:
            default = EMPTY

        params.append(
            KernelParameter(
                name=arg.arg,
                annotation=annotation.strip() if isinstance(annotation, str) else annotation,
                default=default,
                kind=inspect.Parameter.KEYWORD_ONLY,
            )
        )

    if args.kwarg:
        annotation = ast.get_source_segment(source, args.kwarg.annotation) if args.kwarg.annotation else EMPTY

        params.append(
            KernelParameter(
                name=args.kwarg.arg,
                annotation=annotation.strip() if isinstance(annotation, str) else annotation,
                kind=inspect.Parameter.VAR_KEYWORD,
            )
        )

    return tuple(params)


def _extract_specification_from_module_source(source, *, filename, operator=None):
    lines = tuple(source.splitlines(keepends=True))
    module = ast.parse(source, filename=filename)
    operator_nodes = [
        class_node
        for class_node in module.body
        if isinstance(class_node, ast.ClassDef)
        and class_node.name.endswith("Kernel")
    ]

    if not operator_nodes:
        raise ValueError(f"No operator found in {filename}")

    if len(operator_nodes) != 1:
        raise ValueError(
            f"Expected exactly one operator in {filename}, found {len(operator_nodes)}"
        )

    class_node = operator_nodes[0]
    kernel_node = next(
        (
            item
            for item in class_node.body
            if isinstance(item, ast.FunctionDef) and item.name == "kernel"
        ),
        None,
    )

    if kernel_node is None:
        raise ValueError(
            f"Kernel {class_node.name} in {filename} is missing a kernel method"
        )

    kernel_source = "".join(
        lines[kernel_node.lineno - 1 : kernel_node.end_lineno]
    )

    parameters = parse_parameters(kernel_node, source)

    spec = KernelSpec(
        operator_name=class_node.name,
        kernel_name=kernel_node.name,
        source=textwrap.dedent(kernel_source).strip(),
        parameters=parameters,
    )

    if operator is None:
        return spec

    updates = {}

    constexpr_values = getattr(operator, "constexpr_values", None)
    if isinstance(constexpr_values, dict):
        updates["constexpr_values"] = dict(constexpr_values)

    num_warps = getattr(operator, "num_warps", None)
    if isinstance(num_warps, int):
        updates["num_warps"] = num_warps

    if updates:
        spec = replace(spec, **updates)

    return spec


def _load_operator_from_path(path, operator_name):
    module_name = f"triton_ptx_dynamic_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    operator_cls = getattr(module, operator_name)
    return operator_cls()


def extract_specification(path):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    operator_names = [
        class_node.name
        for class_node in module.body
        if isinstance(class_node, ast.ClassDef) and class_node.name.endswith("Kernel")
    ]

    if len(operator_names) != 1:
        return _extract_specification_from_module_source(source, filename=str(path))

    operator = _load_operator_from_path(path, operator_names[0])
    return _extract_specification_from_module_source(
        source,
        filename=str(path),
        operator=operator,
    )


def extract_specification_from_operator(operator):
    if inspect.isclass(operator):
        operator = operator()

    operator_cls = operator.__class__
    source = textwrap.dedent(inspect.getsource(operator_cls))
    spec = _extract_specification_from_module_source(
        source,
        filename=inspect.getsourcefile(operator_cls) or operator_cls.__name__,
        operator=operator,
    )

    if spec.operator_name != operator_cls.__name__:
        raise ValueError(
            f"Expected operator {operator_cls.__name__!r}, found {spec.operator_name!r}"
        )

    return spec
