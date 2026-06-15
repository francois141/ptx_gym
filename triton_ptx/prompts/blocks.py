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
            "compile-time constexpr, omit from PTX signature; the operator default will be used"
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
target the exact PTX version and GPU target listed below. The operator defaults will be used for tl.constexpr values.
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


def constexpr_values_block(spec):
    constexpr_params = [
        param
        for param in spec.parameters
        if _is_constexpr_annotation(param.annotation)
    ]
    if not constexpr_params:
        return "\n\n".join(
            [
                "## Operator Constexpr Values",
                "None.",
            ]
        )

    lines = []
    for param in constexpr_params:
        if param.name in spec.constexpr_values:
            lines.append(f"- {param.name}: {spec.constexpr_values[param.name]!r}")
        else:
            lines.append(f"- {param.name}: unavailable from operator constexpr_values")

    return "\n\n".join(
        [
            "## Operator Constexpr Values",
            "\n".join(
                [
                    "These tl.constexpr parameters are fixed by the operator constexpr_values dict and will be passed at launch.",
                    "Use these exact values when folding constants and writing PTX indexing logic.",
                    "Omit them from the PTX signature and do not include them in the answer dictionary.",
                    *lines,
                ]
            ),
        ]
    )


def num_warps_block(spec):
    total_threads = 32 * spec.num_warps
    return "\n\n".join(
        [
            "## Operator Warp Configuration",
            "\n".join(
                [
                    f"- `num_warps` from the kernel: {spec.num_warps}",
                    f"- The total CTA thread count must equal `32 * num_warps = {total_threads}`.",
                    "- Query this warp count from the kernel metadata instead of assuming a fixed thread count.",
                ]
            ),
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

def signature_template(parameters, *, version, target, address_size, kernel_name="kernel", ptx_signature=None):
    runtime_params = [param for param in parameters if not _is_constexpr_annotation(param.annotation)]

    lines = []
    for index, param in enumerate(runtime_params):
        ptx_type = ptx_signature[index].ptx_type if ptx_signature is not None else ".u64"
        lines.append(f"    .param {ptx_type} {param.name},")

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
- Treat tl.constexpr values as compile-time constants supplied by the operator defaults.
- The launch grid is computed from the operator constexpr values, not from `num_threads_x`.
- If PTX uses one thread for one element in a constexpr-sized tile, set `num_threads_x` to the matching tile size; otherwise explicitly loop each CTA's threads over the full constexpr tile.
- Do not add, remove, reorder, or reinterpret runtime arguments.
- Do not emit host code, CUDA C, Triton, LLVM IR, explanations, or pseudocode.
""".strip()


def commenting_rules():
    return """
## PTX Commenting Rules

- Document the PTX logic with concise `//` comments written in plain human language.
- Add a short comment before each logical group of PTX instructions that explains the purpose of that group.
- Use comments to explain important indexing, masking, data movement, reductions, and stores.
- Keep comments accurate and tightly coupled to the PTX they describe.
""".strip()


def performance_rules(target, version, spec):
    total_threads = 32 * spec.num_warps
    return """
## Performance Rules

Optimize for the specific Triton kernel shown below. Use only optimizations that are semantically valid for this kernel.

Launch tuning guidance:
- `num_threads_x` should be explicitly defined for this kernel and determines the number of threads launched in the CTA's x dimension.
- If you need a multi-dimensional CTA shape, you may also define `num_threads_y` and `num_threads_z` to specify the y and z dimensions.
- Query `num_warps` from the kernel metadata for this operator.
- The sum of the provided thread dimensions must be exactly `32 * num_warps = {total_threads}`: `num_threads_x + num_threads_y + num_threads_z == {total_threads}`, treating omitted `num_threads_y` and `num_threads_z` as 0.
- Choose the launch dimensions deliberately to achieve the best performance while preserving correctness.
- It is important to evaluate a range of grid and block sizes, as these parameters can significantly impact performance.
- Avoid assuming that the current best-performing configuration is optimal. In practice, seemingly unexpected block sizes or thread counts can sometimes deliver superior performance.

For elementwise kernels:
1. Use coalesced global loads and stores.
2. Use predicated memory operations for masks.
3. Use one or more elements per thread when beneficial.
4. Use vectorized loads/stores only when alignment and masking semantics are safe.
5. Use approximate fp32 math instructions only if they satisfy the requested tolerance.
6. Keep temporary values in registers.
7. Fold operator-default tl.constexpr values into immediates.
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
""".format(total_threads=total_threads).strip()


def triton_kernel_block(source):
    return "\n\n".join(
        [
            "## Triton Kernel",
            f"```python\n{source}\n```",
        ]
    )


def output_contract(spec):
    total_threads = 32 * spec.num_warps
    return """
## Output Contract

Return only a Python snippet that defines one answer dictionary named `ptx_kernel`.

The output must be:

- assign exactly one top-level variable named `ptx_kernel`;
- use this shape:

ptx_kernel = {
    "ptx": \"\"\"<valid PTX code>\"\"\",
    "num_threads_x": <required_threads_x>,
    "num_threads_y": <optional_threads_y>,
    "num_threads_z": <optional_threads_z>,
}

- generate exactly one answer;
- put the PTX code directly under the top-level `"ptx"` key;
- include `"num_threads_x"` as a positive Python integer literal for every answer;
- include `"num_threads_y"` and `"num_threads_z"` only when the kernel needs a multi-dimensional CTA shape;
- query `num_warps` from the kernel and make the sum of included thread dimensions exactly `32 * num_warps = __TOTAL_THREADS__`, treating omitted `"num_threads_y"` and `"num_threads_z"` as 0;
- do not include tl.constexpr parameters in the dictionary; the operator defaults are used when launching the PTX kernel;
- make the PTX string valid PTX;
- include concise human-readable PTX comments that explain the logic and each logical instruction group;
- ASCII-only;
- free of markdown fences;
- free of explanations.
-  Predicated execution does not support block syntax. Instead, the predicate must be applied individually to each instruction by placing it at the beginning of the instruction. 

The following syntax is invalid:

@p_warp0 {
    setp.lt.u32 pvalid, rLane, 8;
}

The following syntax is valid

@p_warp0 setp.lt.u32 pvalid, rLane, 8;
""".replace("__TOTAL_THREADS__", str(total_threads)).strip()
