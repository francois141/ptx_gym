#!/usr/bin/env python3
from __future__ import annotations

import triton

from triton_ptx.evaluation.types import Timing
from triton_ptx.helpers.triton import clear_triton_cache


def benchmark(fn) -> Timing:
    p20, p50, p80, p90, p95, p99 = triton.testing.do_bench(
        fn,
        quantiles=[0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
    )
    return Timing(p20, p50, p80, p90, p95, p99)


def evaluate_ptx_performance(op, inputs) -> dict[str, Timing | float | None]:
    clear_triton_cache()

    triton_time = benchmark(lambda: op.forward_triton(inputs))
    ptx_time = benchmark(lambda: op.forward_triton(inputs, ptx=True))

    ptx_speedup = None if ptx_time.p50 <= 0 else triton_time.p50 / ptx_time.p50

    return {
        "triton": triton_time,
        "ptx": ptx_time,
        "ptx_speedup": ptx_speedup,
    }
