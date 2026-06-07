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
from triton_ptx.kernels.add import AddKernel


kernel_config = {
    "ptx": r""".version 8.7
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 output_ptr,
    .param .u32 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
{
    // Registers
    .reg .pred %p_full, %p0, %p1, %p2, %p3;
    .reg .b32 %rN, %rTid, %rCta, %rBlockStart, %rBase, %rIdx0, %rIdx1, %rIdx2, %rIdx3;
    .reg .b64 %rdX, %rdY, %rdOut, %rdAddr;
    .reg .f32 %fx, %fy, %fo;

    // Load parameters
    ld.param.u64 %rdX, [x_ptr];
    ld.param.u64 %rdY, [y_ptr];
    ld.param.u64 %rdOut, [output_ptr];
    ld.param.u32 %rN, [n_elements];

    // Thread/block indices
    mov.u32 %rTid, %tid.x;
    mov.u32 %rCta, %ctaid.x;

    // block_start = blockIdx.x * 1024
    mul.lo.u32 %rBlockStart, %rCta, 1024;

    // base = block_start + tid * 4
    mad.lo.u32 %rBase, %rTid, 4, %rBlockStart;

    // 4 consecutive indices
    mov.u32 %rIdx0, %rBase;
    add.u32 %rIdx1, %rBase, 1;
    add.u32 %rIdx2, %rBase, 2;
    add.u32 %rIdx3, %rBase, 3;

    // Fast path if all 4 are in bounds
    setp.lt.u32 %p_full, %rIdx3, %rN;
    @%p_full bra L_full;

L_tail:
    setp.lt.u32 %p0, %rIdx0, %rN;
    @%p0 mad.wide.u32 %rdAddr, %rIdx0, 4, %rdX;
    @%p0 ld.global.f32 %fx, [%rdAddr];
    @%p0 mad.wide.u32 %rdAddr, %rIdx0, 4, %rdY;
    @%p0 ld.global.f32 %fy, [%rdAddr];
    @%p0 add.f32 %fo, %fx, %fy;
    @%p0 mad.wide.u32 %rdAddr, %rIdx0, 4, %rdOut;
    @%p0 st.global.f32 [%rdAddr], %fo;

    setp.lt.u32 %p1, %rIdx1, %rN;
    @%p1 mad.wide.u32 %rdAddr, %rIdx1, 4, %rdX;
    @%p1 ld.global.f32 %fx, [%rdAddr];
    @%p1 mad.wide.u32 %rdAddr, %rIdx1, 4, %rdY;
    @%p1 ld.global.f32 %fy, [%rdAddr];
    @%p1 add.f32 %fo, %fx, %fy;
    @%p1 mad.wide.u32 %rdAddr, %rIdx1, 4, %rdOut;
    @%p1 st.global.f32 [%rdAddr], %fo;

    setp.lt.u32 %p2, %rIdx2, %rN;
    @%p2 mad.wide.u32 %rdAddr, %rIdx2, 4, %rdX;
    @%p2 ld.global.f32 %fx, [%rdAddr];
    @%p2 mad.wide.u32 %rdAddr, %rIdx2, 4, %rdY;
    @%p2 ld.global.f32 %fy, [%rdAddr];
    @%p2 add.f32 %fo, %fx, %fy;
    @%p2 mad.wide.u32 %rdAddr, %rIdx2, 4, %rdOut;
    @%p2 st.global.f32 [%rdAddr], %fo;

    setp.lt.u32 %p3, %rIdx3, %rN;
    @%p3 mad.wide.u32 %rdAddr, %rIdx3, 4, %rdX;
    @%p3 ld.global.f32 %fx, [%rdAddr];
    @%p3 mad.wide.u32 %rdAddr, %rIdx3, 4, %rdY;
    @%p3 ld.global.f32 %fy, [%rdAddr];
    @%p3 add.f32 %fo, %fx, %fy;
    @%p3 mad.wide.u32 %rdAddr, %rIdx3, 4, %rdOut;
    @%p3 st.global.f32 [%rdAddr], %fo;

    bra L_exit;

L_full:
    mad.wide.u32 %rdAddr, %rIdx0, 4, %rdX;
    ld.global.f32 %fx, [%rdAddr];
    mad.wide.u32 %rdAddr, %rIdx0, 4, %rdY;
    ld.global.f32 %fy, [%rdAddr];
    add.f32 %fo, %fx, %fy;
    mad.wide.u32 %rdAddr, %rIdx0, 4, %rdOut;
    st.global.f32 [%rdAddr], %fo;

    mad.wide.u32 %rdAddr, %rIdx1, 4, %rdX;
    ld.global.f32 %fx, [%rdAddr];
    mad.wide.u32 %rdAddr, %rIdx1, 4, %rdY;
    ld.global.f32 %fy, [%rdAddr];
    add.f32 %fo, %fx, %fy;
    mad.wide.u32 %rdAddr, %rIdx1, 4, %rdOut;
    st.global.f32 [%rdAddr], %fo;

    mad.wide.u32 %rdAddr, %rIdx2, 4, %rdX;
    ld.global.f32 %fx, [%rdAddr];
    mad.wide.u32 %rdAddr, %rIdx2, 4, %rdY;
    ld.global.f32 %fy, [%rdAddr];
    add.f32 %fo, %fx, %fy;
    mad.wide.u32 %rdAddr, %rIdx2, 4, %rdOut;
    st.global.f32 [%rdAddr], %fo;

    mad.wide.u32 %rdAddr, %rIdx3, 4, %rdX;
    ld.global.f32 %fx, [%rdAddr];
    mad.wide.u32 %rdAddr, %rIdx3, 4, %rdY;
    ld.global.f32 %fy, [%rdAddr];
    add.f32 %fo, %fx, %fy;
    mad.wide.u32 %rdAddr, %rIdx3, 4, %rdOut;
    st.global.f32 [%rdAddr], %fo;

L_exit:
    ret;
}
""",
    "num_threads_x": 256,
    "KERNEL_BLOCK_SIZE": 1024,
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
        "kernel": "AddKernel",
        "config": {
            "num_threads_x": config.get("num_threads_x"),
            "KERNEL_BLOCK_SIZE": config.get("KERNEL_BLOCK_SIZE"),
            "ptx_length": len(str(config.get("ptx", ""))),
        },
    }

    compile_result = run_ptx_compilation(AddKernel, ptx_code=config)
    report["compile"] = compile_result
    if not compile_result.get("success", False):
        return report

    operator = AddKernel(
        block_size=int(config.get("KERNEL_BLOCK_SIZE", 1024)),
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
        description="Manually evaluate a PTX kernel_config payload against AddKernel.",
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
