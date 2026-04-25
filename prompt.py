from __future__ import annotations

import ast
import inspect
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelParameter:
    name: str
    annotation: Any
    default: Any = inspect.Parameter.empty
    kind: Any = inspect.Parameter.POSITIONAL_OR_KEYWORD


@dataclass(frozen=True)
class KernelSpec:
    operator_name: str
    kernel_name: str
    source: str
    parameters: tuple[KernelParameter, ...]


def _ascii_repr(value: Any) -> str:
    """
    Return an ASCII-safe representation of a Python value.
    Useful because non-ASCII characters can break PTX compilation
    if they accidentally appear in generated comments or code.
    """
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return repr(value).encode("ascii", "backslashreplace").decode("ascii")


def _annotation_name(annotation: Any) -> str:
    """
    Convert a function annotation to a stable string name.
    """
    if annotation is inspect.Parameter.empty:
        return ""

    if isinstance(annotation, str):
        return annotation

    module = getattr(annotation, "__module__", "")
    qualname = getattr(annotation, "__qualname__", "")
    name = getattr(annotation, "__name__", "")

    if module and qualname:
        return f"{module}.{qualname}"

    return qualname or name or repr(annotation)


def _is_tl_constexpr(annotation: Any) -> bool:
    """
    Detect Triton tl.constexpr annotations.
    """
    annotation_name = _annotation_name(annotation).lower()

    return (
        annotation_name == "constexpr"
        or annotation_name.endswith(".constexpr")
        or ("triton.language" in annotation_name and "constexpr" in annotation_name)
        or "triton.language.core.constexpr" in annotation_name
    )


def _format_param_default(param: KernelParameter) -> str:
    """
    Format a default parameter value for the prompt.
    """
    if param.default is inspect.Parameter.empty:
        return "None"

    return _ascii_repr(param.default)


def _format_argument_list(parameters: tuple[KernelParameter, ...]) -> str:
    """
    Build a readable list of Triton arguments and whether each one
    should appear in the PTX signature.
    """
    lines: list[str] = []

    for param in parameters:
        annotation = _annotation_name(param.annotation)
        is_constexpr = _is_tl_constexpr(param.annotation)

        if is_constexpr:
            ptx_status = (
                "compile-time constexpr, omit from PTX signature, "
                "you must choose the appropriate value"
            )
        else:
            ptx_status = "runtime argument, include in PTX signature"

        line = (
            f"- {param.name}: "
            f"default={_format_param_default(param)}, "
            f"kind={param.kind}, "
            f"annotation={annotation or 'None'}, "
            f"{ptx_status}"
        )
        lines.append(line)

    return "\n".join(lines) if lines else "None."


def _literal_value(node: ast.AST | None) -> Any:
    if node is None:
        return inspect.Parameter.empty

    try:
        return ast.literal_eval(node)
    except Exception:
        return inspect.Parameter.empty


def _source_segment(lines: list[str], start: int, end: int) -> str:
    segment = "".join(lines[start - 1 : end])
    return textwrap.dedent(segment).strip()


def _annotation_segment(source: str, node: ast.AST | None) -> Any:
    if node is None:
        return inspect.Parameter.empty

    text = ast.get_source_segment(source, node)
    if text is None:
        try:
            text = ast.unparse(node)
        except Exception:
            return inspect.Parameter.empty

    return text.strip()


def _discover_kernel_specs(kernels_dir: str | Path = "kernels") -> list[KernelSpec]:
    root = Path(kernels_dir)
    specs: list[KernelSpec] = []

    for path in sorted(root.glob("*.py")):
        if path.name == "__init__.py":
            continue

        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(path))
        lines = source.splitlines(keepends=True)

        for node in module.body:
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("Operator"):
                continue

            kernel_node = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "kernel":
                    kernel_node = item
                    break

            if kernel_node is None:
                continue

            parameters: list[KernelParameter] = []
            args = kernel_node.args
            positional_args = list(args.posonlyargs) + list(args.args)
            defaults = [inspect.Parameter.empty] * (
                len(positional_args) - len(args.defaults)
            ) + [_literal_value(default) for default in args.defaults]

            for arg, default in zip(positional_args, defaults, strict=True):
                parameters.append(
                    KernelParameter(
                        name=arg.arg,
                        annotation=_annotation_segment(source, arg.annotation),
                        default=default,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                )

            if args.vararg is not None:
                parameters.append(
                    KernelParameter(
                        name=args.vararg.arg,
                        annotation=_annotation_segment(source, args.vararg.annotation),
                        default=inspect.Parameter.empty,
                        kind=inspect.Parameter.VAR_POSITIONAL,
                    )
                )

            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                parameters.append(
                    KernelParameter(
                        name=arg.arg,
                        annotation=_annotation_segment(source, arg.annotation),
                        default=_literal_value(default),
                        kind=inspect.Parameter.KEYWORD_ONLY,
                    )
                )

            if args.kwarg is not None:
                parameters.append(
                    KernelParameter(
                        name=args.kwarg.arg,
                        annotation=_annotation_segment(source, args.kwarg.annotation),
                        default=inspect.Parameter.empty,
                        kind=inspect.Parameter.VAR_KEYWORD,
                    )
                )

            specs.append(
                KernelSpec(
                    operator_name=node.name,
                    kernel_name=kernel_node.name,
                    source=_source_segment(lines, kernel_node.lineno, kernel_node.end_lineno),
                    parameters=tuple(parameters),
                )
            )

    return specs


def prompt_builder(
    spec: KernelSpec,
    *,
    version: str = "8.0",
    target: str = "sm_89",
    address_size: int = 64,
    add_dummy_ptrs: bool = True,
    extra_instructions: str | None = None,
) -> str:
    """
    Build a Triton-to-PTX prompt from a kernel spec.
    """
    argument_block = _format_argument_list(spec.parameters)

    if add_dummy_ptrs:
        dummy_ptr_rule = """
- After all runtime arguments, append these two dead parameters exactly:
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
""".strip()
    else:
        dummy_ptr_rule = "- Do not append dummy pointer parameters."

    extra_block = ""
    if extra_instructions and extra_instructions.strip():
        extra_block = f"\n\n## Extra Instructions\n\n{extra_instructions.strip()}"

    prompt = f"""
# Triton to Optimized PTX Conversion

You are given a Triton kernel. Generate one equivalent, compile-ready PTX kernel.

## PTX Header

Use exactly this PTX header:

.version {version}
.target {target}
.address_size {address_size}

## Extracted Triton Signature Information

{argument_block}

## Signature Rules

- The PTX kernel name must be exactly the same as the Triton kernel function name.
- Preserve the same order of all Triton runtime arguments in the PTX .entry signature.
- Runtime arguments are the parameters listed above as "include in PTX signature".
- Do not include tl.constexpr parameters in the PTX .entry signature. They are compile-time constants.
- You are responsible for finding the appropriate value for each tl.constexpr parameter.
- Use the Triton kernel body, defaults, and surrounding code to infer sensible constexpr values.
- Pointer arguments must be passed as .param .u64.
- Scalar argument types must match the Triton argument type when it is explicit.
- If a scalar type is ambiguous:
  - use .u64 for pointers, offsets, sizes, strides, and address-like values;
  - use .u32 or .s32 for ordinary 32-bit integer values;
  - use .f32 for float32 values.
{dummy_ptr_rule}

## Correctness Rules

- Implement the same computation and control flow as the Triton kernel.
- Respect all masks and boundary conditions exactly.
- Assume pointer inputs refer to contiguous GPU global memory unless the Triton code says otherwise.
- Treat tl.constexpr values as compile-time constants after you infer them.
- Do not add, remove, reorder, or reinterpret runtime arguments.
- Do not emit host code, CUDA C, Triton, LLVM IR, explanations, or pseudocode.

## Performance Rules

Optimize the PTX for runtime speed on {target}.

Priority order:

1. Use coalesced global memory accesses whenever the Triton indexing permits it.
2. Use vectorized loads and stores when they are safe, aligned, and preserve masking semantics.
3. Use predicated PTX instructions for masks and boundary checks where possible.
4. Avoid divergent branches unless they are clearly cheaper than predication.
5. Keep temporary values in registers.
6. Avoid local memory, stack usage, and register spills where possible.
7. Hoist invariant arithmetic out of repeated computations.
8. Fold tl.constexpr values into immediates.
9. Prefer efficient address arithmetic such as mad, mad.lo, mad.wide, shl, and add over slower sequences.
10. Replace division or modulo by powers of two with shifts and masks when valid.
11. Avoid redundant conversions, redundant loads, and redundant stores.
12. Do not add debugging code, asserts, printf, comments, or unused computations, except for the required dummy parameters.

## Triton Kernel

```python
{spec.source}
```

## Output Contract

Return only the PTX code.

The output must be:

- enclose it in ptx = \"\"\" solution \"\"\"
- valid PTX;
- ASCII-only;
- free of markdown fences;
- free of comments;
- free of explanations.
{extra_block}
"""
    return textwrap.dedent(prompt).strip()


def generate_prompts(output_dir: str | Path = "generated_prompts") -> list[Path]:
    """
    Generate a markdown prompt file for each operator.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for spec in _discover_kernel_specs():
        prompt = prompt_builder(spec)
        path = out_dir / f"{spec.operator_name}.md"
        path.write_text(prompt + "\n", encoding="utf-8")
        written_files.append(path)

    return written_files


if __name__ == "__main__":
    for path in generate_prompts():
        print(path)
