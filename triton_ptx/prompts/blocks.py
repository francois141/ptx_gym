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


def initial_task(num_answers: int = 5):
    return f"""
# Triton to Fastest PTX Conversion

You are given a Triton kernel. Generate {num_answers} distinct equivalent, compile-ready PTX kernels.
The kernel must be the fastest implementation you can produce for the exact
PTX version and target listed below.
    """


def follow_up_task(num_answers: int = 5):
    return f"""
# PTX Test-Time Scaling

You are given candidate PTX answers for the same Triton kernel, along with
evaluation results that show whether each candidate compiled, whether it was
correct, and how fast it ran.

Use that feedback to generate {num_answers} distinct improved PTX kernels. Each answer
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
    return f"""
## Performance Rules

Optimize the PTX for runtime speed on {target} using PTX ISA version {version}.
Use target-specific instructions aggressively when they are semantically valid
for this kernel and supported by the requested PTX version/target.

Priority order:

1. Use coalesced global memory accesses whenever the Triton indexing permits it.
2. Use vectorized loads and stores when they are safe, aligned, and preserve masking semantics.
3. Use async global-to-shared copies, prefetching, double buffering, and barriers when they reduce latency for the kernel.
4. Use tensor core instructions such as mma/wgmma when the Triton computation is a matrix/tensor contraction with compatible data types and tile shapes.
5. Use predicated PTX instructions for masks and boundary checks where possible.
6. Avoid divergent branches unless they are clearly cheaper than predication.
7. Keep temporary values in registers.
8. Avoid local memory, stack usage, and register spills where possible.
9. Hoist invariant arithmetic out of repeated computations.
10. Fold tl.constexpr values into immediates.
11. Prefer efficient address arithmetic such as mad, mad.lo, mad.wide, shl, and add over slower sequences.
12. Replace division or modulo by powers of two with shifts and masks when valid.
13. Avoid redundant conversions, redundant loads, and redundant stores.
14. Do not add debugging code, asserts, printf, comments, or unused computations, except for the required dummy parameters.
""".strip()


def triton_kernel_block(source):
    return "\n\n".join(
        [
            "## Triton Kernel",
            f"```python\n{source}\n```",
        ]
    )


def output_contract(num_answers: int = 5):
    answers_template = ",\n".join(
        """        {
            "ptx": \"\"\"<valid PTX code>\"\"\",
            "<constexpr_name>": <chosen_constexpr_value>,
        }"""
        for _ in range(num_answers)
    )

    return f"""
## Output Contract

Return only a Python snippet that defines a dictionary named `ptx_kernel` containing {num_answers} answers.

The output must be:

- assign exactly one top-level variable named `ptx_kernel`;
- use this shape:

ptx_kernel = {{
    "answers": [
{answers_template}
    ],
}}

- generate exactly {num_answers} answers;
- put the PTX code directly under the top-level `"ptx"` key in each answer;
- for each tl.constexpr parameter, choose the appropriate compile-time value and put it directly in each answer dictionary using the constexpr parameter name as the key;
- use only Python literal values for constexpr dictionary values;
- make each PTX string valid PTX;
- ASCII-only;
- free of markdown fences;
- free of comments;
- free of explanations.
""".strip()
