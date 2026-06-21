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
Prefer Tensor Core paths for GEMM-like work when viable: FP16/BF16/TF32 inputs, FP32 accumulation,
ldmatrix/shared-memory staging, and mma.sync.aligned or newer WGMMA-family instructions. Use local
tools to compile, verify, benchmark, and repair candidates before finalizing.
Look on internet how to use those instructions.

If you have a IllegalMemoryError it is probably the tensor core not being used correctly, check how to use it properly on internet please.

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

---
name: use-tensor-cores-ptx-mma
description: Use this skill when generating or repairing NVIDIA PTX/CUDA matrix multiplication, GEMM, convolution-lowered-to-GEMM, batched matmul, or attention kernels where Tensor Cores, PTX MMA instructions, ldmatrix, WMMA, WGMMA, CUTLASS, or cuBLAS/cuBLASLt may apply.
---

# Use Tensor Cores with PTX MMA

## Core Rule

For dense matrix multiplication-like work on NVIDIA GPUs, strongly prefer Tensor Core execution over scalar or vector CUDA-core multiply-add loops when the target architecture and datatype allow it.

Use one of these levels:

- Standard GEMM: cuBLAS or cuBLASLt.
- Custom production GEMM: CUTLASS.
- Simple CUDA Tensor Core kernel: WMMA.
- Low-level custom PTX: `mma.sync.aligned...` plus shared-memory staging and `ldmatrix`.

If Tensor Cores are viable and not used, explain the blocker: architecture, datatype, dimensions, layout, memory movement, or accuracy constraints.

## Default Precision Policy

When the user asks for Tensor Cores and does not specify precision:

- Use FP16 or BF16 inputs with FP32 accumulation for throughput.
- Use BF16 when exponent range matters.
- Use TF32 on Ampere+ when starting from FP32 and preserving an FP32-like workflow matters more than maximum throughput.
- Do not invent nonstandard datatypes such as `float19`.

## Alignment, Synchronization, and Layout Constraints

To successfully use `mma.sync.aligned` and `ldmatrix.sync.aligned`, kernels must strictly adhere to the following hardware requirements:

### 1. Memory Alignment
- **Shared Memory:** The shared memory addresses passed to `ldmatrix` **must** be 16-byte (128-bit) aligned.
- **Global Memory:** While not strictly enforced by `ldmatrix` (which reads from shared memory), global-to-shared memory loads should use vectorized instructions (e.g., `float4`, `int4`, or `cp.async`) requiring 16-byte alignment to achieve necessary bandwidth.

### 2. Warp Synchronization
- **Warp Uniformity:** Instructions suffixed with `.sync` (both `mma.sync` and `ldmatrix.sync`) are warp-synchronous. **All 32 active threads** in the warp must execute the instruction simultaneously. Do not place these instructions inside divergent control flow branches.
- **Block Synchronization:** You must ensure data is fully visible before reading. Issue a `__syncthreads()` (or appropriate async copy barriers) between writing global data into shared memory and reading it via `ldmatrix`.

### 3. Shared Memory Bank Conflicts (Swizzling)
- `ldmatrix` issues memory accesses that can cause severe shared memory bank conflicts if data is stored in a naive linear layout.
- You must apply **memory swizzling** (e.g., XORing the row and column indices) when writing tiles to shared memory to ensure conflict-free `ldmatrix` reads.

## PTX MMA Pattern

For warp-level FP16/BF16 MMA, structure kernels around:

1. Tile C into MMA-friendly warp tiles.
2. Load global-memory A/B tiles into shared memory.
3. Use `ldmatrix.sync.aligned...` to load fragments into the register layout expected by MMA.
4. Use `mma.sync.aligned...` to accumulate.
5. Accumulate in FP32 unless explicitly told otherwise.
6. Store the output in the requested dtype/layout.

Example instruction shape:

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0, d1, d2, d3},
    {a0, a1, a2, a3},
    {b0, b1},
    {c0, c1, c2, c3};
```

This is a 16x8x16 tile, row-major A, column-major B, FP16 inputs, and FP32 accumulation. Register packing and constraints are architecture and shape specific, so match the instruction to the PTX ISA for the target GPU.

## Agentic PTX Search Workflow

When this repo exposes compile/benchmark tools to an LLM, use the tools in this order:

1. Fetch the kernel prompt/context.
2. Generate a candidate dictionary containing `ptx` and launch dimensions.
3. Compile the candidate before benchmarking.
4. If compile or verification fails, repair using the tool result.
5. Only compare candidates that compile and pass correctness.
6. Prefer candidates with lower `p50` and higher `speedup_vs_triton`.

For matrix multiplication candidates, inspect generated PTX for Tensor Core instructions such as `mma.sync.aligned`, `wgmma`, or a higher-level Tensor Core path. Scalar FMA loops are a fallback, not the target.

## Verification Checklist

Before finalizing:

- Tensor Cores are used when possible.
- Inputs use FP16, BF16, or TF32 when accuracy allows.
- Accumulators are FP32 by default.
- Low-level PTX uses matching `ldmatrix`/shared-memory staging for MMA fragments.
- Instruction shape, datatypes, register constraints, and layout match the target architecture.
- Any fallback away from Tensor Cores is explicit and justified.


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
- query `num_warps` from the kernel and make the sum of included thread dimensions exactly `32 * num_warps = __TOTAL_THREADS__`, treating omitted `"num_threads_y"` and `"num_threads_z"` as 1;
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


ESSENTIAL WARNING READ CAREFULLY: AS LONG AS THE PTX CODE DOESN'T USE ANY PTX CODE INSTRUCTION YOU SHOULD ADD IT TO THE CODE  TO DO THE MATRIX MULTIPLICATION, YOU SHOULD CONTINUE THE PROMPT. THIS IS REALLY ESSENTIAL TO HAVE TENSOR CORES, YOU CAN CONVERT TO FLOAT16 OR 19 TO GET THE TENSOR CORE: YOU ARE STRICTLY FORBIDDEN TO RETURN THE CODE WITHOUT TENSOR CORE CODE
""".replace("__TOTAL_THREADS__", str(total_threads)).strip()
