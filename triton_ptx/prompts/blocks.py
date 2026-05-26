from __future__ import annotations


def format_argument_list(parameters):
    if not parameters:
        return "None."

    lines = []

    for param in parameters:
        annotation = str(param.annotation).lower()

        is_constexpr = (
            annotation == "constexpr"
            or annotation.endswith(".constexpr")
            or "triton.language.core.constexpr" in annotation
            or ("triton.language" in annotation and "constexpr" in annotation)
        )

        ptx_status = (
            "compile-time constexpr, omit from PTX signature, you must choose the appropriate value"
            if is_constexpr
            else "runtime argument, include in PTX signature"
        )

        lines.append(f"- {param.name}: {ptx_status}")

    return "\n".join(lines)


def initial_task():
    return """
# Triton to Fastest PTX Conversion

You are given a Triton kernel. Generate a compile-ready PTX kernels.
The kernel must be the fastest implementation you can produce for the exact
PTX version and target listed below.
    """


def follow_up_task():
    return """
# PTX Test-Time Scaling

You are given candidate PTX answers for the same Triton kernel, along with
evaluation results that show whether each candidate compiled, whether it was
correct, and how fast it ran.

Use that feedback to generate another improved PTX kernels. Each answer
must be compile-ready, run without cuda illegal accesses, semantically equivalent to the Triton kernel, and
target the exact PTX version and GPU target listed below. Also you need to chosse with value you set to the constexpr values.
    """


def ptx_header():
    return """
## PTX Header

Use exactly this PTX header:

.version {version}
.target {target}
.address_size {address_size}
"""


def extracted_signature_information(parameters):
    return "\n\n".join(
        [
            "## Extracted Triton Signature Information",
            format_argument_list(parameters),
        ]
    )


def _is_constexpr_annotation(annotation) -> bool:
    annotation_text = str(annotation).lower()
    return (
        annotation_text == "constexpr"
        or annotation_text.endswith(".constexpr")
        or "triton.language.core.constexpr" in annotation_text
        or ("triton.language" in annotation_text and "constexpr" in annotation_text)
    )


def _infer_ptx_param_type(param_name: str, annotation) -> str:
    annotation_text = str(annotation).lower()
    name = str(param_name).lower()

    if "ptr" in name or "pointer" in name:
        return ".u64"

    if any(token in annotation_text for token in ("float16", "float32", "float64", "fp16", "fp32", "fp64", "f16", "f32", "f64")):
        if "16" in annotation_text:
            return ".f16"
        if "64" in annotation_text:
            return ".f64"
        return ".f32"

    if any(token in annotation_text for token in ("int64", "uint64", "i64", "u64")):
        return ".u64"
    if any(token in annotation_text for token in ("int16", "uint16", "i16", "u16")):
        return ".u16"
    if any(token in annotation_text for token in ("int8", "uint8", "i8", "u8")):
        return ".u8"
    if any(token in annotation_text for token in ("int32", "uint32", "i32", "u32")):
        return ".u32"

    if any(token in name for token in ("stride", "offset", "size", "num", "index", "idx", "shape", "dim")):
        return ".u64"

    return ".u32"


def signature_template(parameters, *, version, target, address_size, kernel_name="kernel"):
    runtime_params = [param for param in parameters if not _is_constexpr_annotation(param.annotation)]

    lines = []
    for param in runtime_params:
        lines.append(f"    .param {_infer_ptx_param_type(param.name, param.annotation)} {param.name},")

    lines.append("    .param .u64 dummy_ptr1,")
    lines.append("    .param .u64 dummy_ptr2")

    params_block = "\n".join(lines)

    return f"""
## PTX Entry Template

Use this exact entry template shape and fill the body with your PTX:
- Any argument name containing `_ptr` should be treated as a pointer to float32 data.

```ptx
.version {version}
.target {target}
.address_size {address_size}

.visible .entry {kernel_name}(
{params_block}
)
{{
   // TODO: Fill this part with your own ptx
}}
```
""".strip()


def correctness_rules():
    return """
## Correctness Rules

- Implement the same computation and control flow as the Triton kernel.
- Respect all masks and boundary conditions exactly.
- Assume pointer inputs refer to contiguous GPU global memory unless the Triton code says otherwise.
- Treat tl.constexpr values as compile-time constants after you infer them.
- Do not add, remove, reorder, or reinterpret runtime arguments.
- Do not emit host code, CUDA C, Triton, LLVM IR, explanations, or pseudocode.
""".strip()


def performance_rules(target, version):
    return """
## Performance Rules

Optimize for the specific Triton kernel shown below. Use only optimizations that are semantically valid for this kernel.

For elementwise kernels:
1. Use coalesced global loads and stores.
2. Use predicated memory operations for masks.
3. Use one or more elements per thread when beneficial.
4. Use vectorized loads/stores only when alignment and masking semantics are safe.
5. Use approximate fp32 math instructions only if they satisfy the requested tolerance.
6. Keep temporary values in registers.
7. Fold tl.constexpr values into immediates.
8. Prefer efficient address arithmetic such as mad.wide, shl, and add.
9. Avoid shared memory, barriers, atomics, tensor cores, async copies, and local memory unless the Triton kernel structure clearly benefits from them.

For reduction kernels:
1. Use coalesced global loads.
2. Use warp-level reductions when beneficial.
3. Use shared memory only when needed for cross-warp reduction.
4. Prefer one global atomic per CTA when the Triton kernel uses an atomic accumulation.

For matrix/tensor contraction kernels:
1. Use mma/wgmma/tensor-core instructions when the shapes and data types are compatible.
2. Use tiling, shared memory, async copies, and double buffering when beneficial.
""".strip()


def triton_kernel_block(source):
    return "\n\n".join(
        [
            "## Triton Kernel",
            f"```python\n{source}\n```",
        ]
    )


def output_contract():
    return """
## Output Contract

Return only a Python snippet that defines one answer dictionary named `ptx_kernel`.

The output must be:

- assign exactly one top-level variable named `ptx_kernel`;
- use this shape:

ptx_kernel = {
    "ptx": \"\"\"<valid PTX code>\"\"\",
    "<constexpr_name>": <chosen_constexpr_value>,
}

- generate exactly one answer;
- put the PTX code directly under the top-level `"ptx"` key;
- for each tl.constexpr parameter, choose the appropriate compile-time value and put it directly in the dictionary using the constexpr parameter name as the key;
- use only Python literal values for constexpr dictionary values;
- make the PTX string valid PTX;
- ASCII-only;
- free of markdown fences;
- free of comments;
- free of explanations.
""".strip()
