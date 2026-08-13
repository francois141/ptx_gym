from __future__ import annotations

from statistics import median

import triton
from triton_ptx.evaluation.types import Timing
from triton_ptx.helpers.triton import clear_triton_cache


def benchmark(fn) -> Timing:
    timing_rounds = [
        triton.testing.do_bench(
            fn,
            warmup=200,
            rep=2000,
            quantiles=[0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
        )
        for _ in range(5)
    ]
    return Timing(*(median(values) for values in zip(*timing_rounds, strict=True)))


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
