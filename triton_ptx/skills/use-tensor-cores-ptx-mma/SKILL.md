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
