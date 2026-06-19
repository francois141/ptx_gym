#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton

from triton_ptx.helpers.triton import clear_triton_cache


@dataclass(frozen=True)
class Timing:
    p20: float
    p50: float
    p80: float


def benchmark(fn, quantiles=(0.2, 0.5, 0.8)) -> Timing:
    p20, p50, p80 = triton.testing.do_bench(
        fn,
        quantiles=list(quantiles),
    )
    return Timing(p20, p50, p80)


def evaluate_ptx_performance(op, inputs) -> dict[str, Timing | float | None]:
    clear_triton_cache()
    compiled_torch = torch.compile(op.forward_torch)

    triton_time = benchmark(lambda: op.forward_triton(inputs))
    ptx_time = benchmark(lambda: op.forward_triton(inputs, ptx=True))
    torch_time = benchmark(lambda: compiled_torch(inputs))

    ptx_speedup = None if ptx_time.p50 <= 0 else triton_time.p50 / ptx_time.p50
    torch_speedup = None if torch_time.p50 <= 0 else triton_time.p50 / torch_time.p50

    return {
        "triton": triton_time,
        "ptx": ptx_time,
        "torch": torch_time,
        "ptx_speedup": ptx_speedup,
        "torch_speedup": torch_speedup,
    }
