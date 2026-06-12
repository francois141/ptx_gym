from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from triton_ptx.evaluation.sandbox import (
    OutputVerifier,
    PTXBenchmarkRunner,
    run_ptx_compilation,
)
from triton_ptx.kernels.matrix_scalar_addition import MatrixScalarAdditionKernel


kernel_config = {
    "ptx": r""".version 8.7
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 x_ptr,
    .param .u64 output_ptr,
    .param .f32 scalar,
    .param .u32 stride_xm,
    .param .u32 stride_ym,
    .param .u32 size,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
{
    // Registers
    .reg .pred  pN, pM, p, pLoop;
    .reg .u32   rTid, rPidM, rPidN, rSize, rStrideXm, rStrideYm;
    .reg .u32   rN, rMBase, rIdxX, rIdxY, rMIndex, rIter, rTmp32, rTmp32Y;
    .reg .u64   rXPtr, rYPtr, rAddrX, rAddrY, rOfsX, rOfsY;
    .reg .f32   fScalar, fVal, fOut;

    // Load kernel parameters
    ld.param.u64 rXPtr,      [x_ptr];
    ld.param.u64 rYPtr,      [output_ptr];
    ld.param.f32 fScalar,    [scalar];
    ld.param.u32 rStrideXm,  [stride_xm];
    ld.param.u32 rStrideYm,  [stride_ym];
    ld.param.u32 rSize,      [size];

    // Read CTA and thread indices
    mov.u32 rPidM, %ctaid.x;
    mov.u32 rPidN, %ctaid.y;
    mov.u32 rTid,  %tid.x;

    // Compute column index n = pid_n * BLOCK_N + tid.x  (BLOCK_N=128)
    mul.lo.u32 rN, rPidN, 128;
    add.u32    rN, rN, rTid;

    // Column bounds check: if n >= size, nothing to do for this thread
    setp.lt.u32 pN, rN, rSize;
    @!pN ret;

    // Compute row base m_base = pid_m * BLOCK_M  (BLOCK_M=64)
    mul.lo.u32 rMBase, rPidM, 64;

    // Prepare initial flat indices for x and y: idx = m_base*stride + n
    mul.lo.u32 rTmp32,  rMBase, rStrideXm;
    add.u32    rIdxX,   rTmp32, rN;
    mul.lo.u32 rTmp32Y, rMBase, rStrideYm;
    add.u32    rIdxY,   rTmp32Y, rN;

    // Initialize loop counters for BLOCK_M rows
    mov.u32 rMIndex, rMBase;
    mov.u32 rIter,   0;

L_loop:
    // Mask combines row and column bounds: (m_index < size) & (n < size)
    setp.lt.u32 pM, rMIndex, rSize;
    and.pred p, pM, pN;

    // Compute address for x: addr_x = x_ptr + (idx_x * 4)
    mul.wide.u32 rOfsX, rIdxX, 4;
    add.s64      rAddrX, rXPtr, rOfsX;

    // Masked load with other=0.0
    mov.f32 fVal, 0f00000000;
    @p ld.global.f32 fVal, [rAddrX];

    // Add scalar
    add.f32 fOut, fVal, fScalar;

    // Compute address for y: addr_y = y_ptr + (idx_y * 4)
    mul.wide.u32 rOfsY, rIdxY, 4;
    add.s64      rAddrY, rYPtr, rOfsY;

    // Masked store
    @p st.global.f32 [rAddrY], fOut;

    // Advance to next row
    add.u32 rIdxX,   rIdxX,   rStrideXm;
    add.u32 rIdxY,   rIdxY,   rStrideYm;
    add.u32 rMIndex, rMIndex, 1;
    add.u32 rIter,   rIter,   1;

    // Loop for BLOCK_M rows
    setp.lt.u32 pLoop, rIter, 64;
    @pLoop bra L_loop;

    // Done
    ret;
}
""",
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "num_threads_x": 128,
    "num_threads_y": 1,
    "num_threads_z": 1,
}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _serialize_timings(timings: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in timings.items():
        serialized[key] = _json_safe(value)
    return serialized


def evaluate_config(
    config: dict[str, Any],
    *,
    verify_sizes: tuple[int, ...],
    verify_iters: int,
    verify_seed: int,
    benchmark: bool,
    benchmark_size: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kernel": "MatrixScalarAdditionKernel",
        "config": {
            "BLOCK_M": config.get("BLOCK_M"),
            "BLOCK_N": config.get("BLOCK_N"),
            "num_threads_x": config.get("num_threads_x"),
            "num_threads_y": config.get("num_threads_y"),
            "num_threads_z": config.get("num_threads_z"),
            "ptx_length": len(str(config.get("ptx", ""))),
        },
    }

    compile_result = run_ptx_compilation(MatrixScalarAdditionKernel, ptx_code=config)
    report["compile"] = compile_result
    if not compile_result.get("success", False):
        return report

    operator = MatrixScalarAdditionKernel(
        block_m=int(config.get("BLOCK_M", 64)),
        block_n=int(config.get("BLOCK_N", 128)),
        ptx=config,
    )

    verifier = OutputVerifier(
        sizes=verify_sizes,
        iters_per_size=verify_iters,
        seed=verify_seed,
    )
    report["correct"] = verifier.verify(operator)
    report["verification"] = _json_safe(verifier.last_report)

    if benchmark and report["correct"]:
        runner = PTXBenchmarkRunner()
        inputs = operator.get_random_input(size=benchmark_size)
        report["benchmark"] = _serialize_timings(runner.evaluate(operator, inputs))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually evaluate a PTX kernel_config payload against MatrixScalarAdditionKernel.",
    )
    parser.add_argument(
        "--verify-sizes",
        type=int,
        nargs="*",
        default=[128, 1024, 4096, 16384],
        help="Input sizes used for correctness verification.",
    )
    parser.add_argument(
        "--verify-iters",
        type=int,
        default=10,
        help="Verification iterations per size.",
    )
    parser.add_argument(
        "--verify-seed",
        type=int,
        default=42,
        help="Random seed used during verification.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run Triton/PTX/Torch benchmarks after correctness passes.",
    )
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=1_000_000,
        help="Input size used for benchmarking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_config(
        kernel_config,
        verify_sizes=tuple(args.verify_sizes),
        verify_iters=args.verify_iters,
        verify_seed=args.verify_seed,
        benchmark=args.benchmark,
        benchmark_size=args.benchmark_size,
    )
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
