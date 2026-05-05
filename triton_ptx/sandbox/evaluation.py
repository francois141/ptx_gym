#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import triton
import torch
import matplotlib.pyplot as plt

from triton_ptx.helpers import (
    check_similarity,
    has_ptx_code,
    clear_triton_cache,
)
from triton_ptx.kernels import operator_list_single


COLUMNS = [
    "Operator Name", "Status",
    "Triton p20 (ms)", "Triton p50 (ms)", "Triton p80 (ms)",
    "Triton PTX p20 (ms)", "Triton PTX p50 (ms)", "Triton PTX p80 (ms)",
    "Torch p20 (ms)", "Torch p50 (ms)", "Torch p80 (ms)",
    "PTX vs Triton Speedup", "Torch vs Triton Speedup",
]


@dataclass(frozen=True)
class Timing:
    p20: float
    p50: float
    p80: float

    def rounded(self) -> tuple[float, float, float]:
        return round(self.p20, 4), round(self.p50, 4), round(self.p80, 4)


class BenchmarkRunnerBase(ABC):
    def __init__(self, operators):
        self.operators = list(operators)
        self.rows: list[dict[str, Any]] = []

    def run(self):
        clear_triton_cache()

        for operator_cls in self.operators:
            self.run_operator(operator_cls)

        table = pd.DataFrame(self.rows, columns=COLUMNS)
        self.plot_results(table)

        return table

    def run_operator(self, operator_cls) -> None:
        name = operator_cls.__name__

        try:
            op = operator_cls()

            if not self.should_run(op):
                return

            inputs = op.get_random_input()

            if not self.verify_outputs(op, inputs):
                print(f"Result: {name} FAILED (Correctness check failed)")
                self.rows.append(self.empty_row(name, "FAILED"))
                return

            metrics = self.evaluate(op, inputs)
            self.print_result(name, metrics)
            self.rows.append(self.result_row(name, metrics))

        except Exception as exc:
            print(f"Result: {name} ERROR ({exc})")
            self.rows.append(self.empty_row(name, "ERROR"))

    @abstractmethod
    def should_run(self, op) -> bool:
        pass

    @abstractmethod
    def evaluate(self, op, inputs) -> dict[str, Any]:
        pass

    def verify_outputs(self, op, inputs) -> bool:
        torch_output = op.forward_torch(inputs)
        triton_output, _ = op.forward_triton(inputs)
        if not check_similarity(torch_output, triton_output):
            return False

        if has_ptx_code(getattr(op, "ptx", None)):
            ptx_output, _ = op.forward_triton(inputs, ptx=True)
            return check_similarity(torch_output, ptx_output)

        return True

    def benchmark(self, fn, quantiles=(0.2, 0.5, 0.8)) -> Timing:
        p20, p50, p80 = triton.testing.do_bench(
            fn,
            quantiles=list(quantiles),
        )
        return Timing(p20, p50, p80)

    def speedup(
        self,
        baseline: Timing,
        candidate: Timing | None,
    ) -> float | None:
        if candidate is None or candidate.p50 <= 0:
            return None
        return baseline.p50 / candidate.p50

    def empty_row(self, name: str, status: str) -> dict[str, Any]:
        return dict.fromkeys(COLUMNS, 0.0) | {
            "Operator Name": name,
            "Status": status,
        }

    def timing_cols(
        self,
        prefix: str,
        timing: Timing | None,
    ) -> dict[str, float | None]:
        if timing is None:
            return {
                f"{prefix} p20 (ms)": None,
                f"{prefix} p50 (ms)": None,
                f"{prefix} p80 (ms)": None,
            }

        p20, p50, p80 = timing.rounded()
        return {
            f"{prefix} p20 (ms)": p20,
            f"{prefix} p50 (ms)": p50,
            f"{prefix} p80 (ms)": p80,
        }

    def result_row(self, name: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "Operator Name": name,
            "Status": "PASSED",
            **self.timing_cols("Triton", metrics["triton"]),
            **self.timing_cols("Triton PTX", metrics["ptx"]),
            **self.timing_cols("Torch", metrics["torch"]),
            "PTX vs Triton Speedup": (
                round(metrics["ptx_speedup"], 2)
                if metrics["ptx_speedup"] is not None
                else None
            ),
            "Torch vs Triton Speedup": (
                round(metrics["torch_speedup"], 2)
                if metrics["torch_speedup"] is not None
                else None
            ),
        }

    def print_result(self, name: str, metrics: dict[str, Any]) -> None:
        triton = metrics["triton"]
        ptx = metrics["ptx"]
        torch_time = metrics["torch"]

        ptx_text = (
            f", Triton PTX p50: {ptx.p50:.4f} ms "
            f"({metrics['ptx_speedup']:.2f}x vs Triton)"
            if ptx is not None
            else ""
        )

        torch_speedup = metrics["torch_speedup"]
        torch_text = f"{torch_speedup:.2f}" if torch_speedup is not None else "n/a"

        print(
            f"Result: {name} PASSED "
            f"(Triton p50: {triton.p50:.4f} ms"
            f"{ptx_text}, "
            f"Torch p50: {torch_time.p50:.4f} ms "
            f"({torch_text}x vs Triton))"
        )

    def plot_results(self, df, output_path: str | Path = "results.jpg") -> None:
        if not hasattr(df, "empty") or df.empty:
            return

        passed = df[df["Status"] == "PASSED"]
        if passed.empty:
            return

        cols = ["Triton p50 (ms)", "Torch p50 (ms)"]
        if passed["Triton PTX p50 (ms)"].notna().any():
            cols.insert(1, "Triton PTX p50 (ms)")

        ax = passed.plot(
            x="Operator Name",
            y=cols,
            kind="bar",
            figsize=(12, 6),
        )

        ax.set_title("Execution Time Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()


class PTXBenchmarkRunner(BenchmarkRunnerBase):
    def should_run(self, op) -> bool:
        return has_ptx_code(getattr(op, "ptx", None))

    def evaluate(self, op, inputs) -> dict[str, Any]:
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


def main():
    PTXBenchmarkRunner(operator_list_single).run()

if __name__ == "__main__":
    main()