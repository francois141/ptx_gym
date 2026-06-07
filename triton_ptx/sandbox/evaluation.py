#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton

from triton_ptx.helpers import clear_triton_cache


@dataclass(frozen=True)
class Timing:
    p20: float
    p50: float
    p80: float


class PTXBenchmarkRunner:
    """Benchmark helper for a single operator with and without PTX overrides."""

    def __init__(self, kernels=None):
        self.kernels = list(kernels or [])

    def benchmark(self, fn, quantiles=(0.2, 0.5, 0.8)) -> Timing:
        p20, p50, p80 = triton.testing.do_bench(
            fn,
            quantiles=list(quantiles),
        )
        return Timing(p20, p50, p80)

    @staticmethod
    def speedup(
        baseline: Timing,
        candidate: Timing | None,
    ) -> float | None:
        if candidate is None or candidate.p50 <= 0:
            return None
        return baseline.p50 / candidate.p50

    def evaluate(self, op, inputs) -> dict[str, Timing | float | None]:
        clear_triton_cache()
        compiled_torch = torch.compile(op.forward_torch)

        triton_time = self.benchmark(lambda: op.forward_triton(inputs))
        ptx_time = self.benchmark(lambda: op.forward_triton(inputs, ptx=True))
        torch_time = self.benchmark(lambda: compiled_torch(inputs))

        return {
            "triton": triton_time,
            "ptx": ptx_time,
            "torch": torch_time,
            "ptx_speedup": self.speedup(triton_time, ptx_time),
            "torch_speedup": self.speedup(triton_time, torch_time),
        }
