from __future__ import annotations

from statistics import mean

import triton
from ptx_gym.evaluation.types import Timing
from ptx_gym.helpers.triton import clear_triton_cache


def benchmark(fn) -> Timing:
    timings = triton.testing.do_bench(
        fn,
        warmup=200,
        rep=500,
        quantiles=[0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
    )
    return Timing(*timings)


def mean_timing(timing_rounds: list[Timing]) -> Timing:
    return Timing(
        *(
            mean(getattr(timing, field) for timing in timing_rounds)
            for field in Timing.__dataclass_fields__
        )
    )


def evaluate_ptx_performance(op, inputs) -> dict[str, Timing | float | None]:
    clear_triton_cache()

    ptx_rounds = []
    triton_rounds = []
    for round_index in range(50):
        if round_index % 2 == 0:
            ptx_rounds.append(benchmark(lambda: op.forward_triton(inputs, ptx=True)))
            triton_rounds.append(benchmark(lambda: op.forward_triton(inputs)))
        else:
            triton_rounds.append(benchmark(lambda: op.forward_triton(inputs)))
            ptx_rounds.append(benchmark(lambda: op.forward_triton(inputs, ptx=True)))

    ptx_time = mean_timing(ptx_rounds)
    triton_time = mean_timing(triton_rounds)

    ptx_speedup = None if ptx_time.p50 <= 0 else triton_time.p50 / ptx_time.p50

    return {
        "triton": triton_time,
        "ptx": ptx_time,
        "ptx_speedup": ptx_speedup,
    }
