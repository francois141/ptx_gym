#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NamedTuple


class WinnerRecord(NamedTuple):
    kernel_name: str
    run_dir: str
    source_path: Path
    round_index: int | None
    speedup_vs_triton: float
    triton_p50: float | None
    p50: float | None


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def load_winner(path: Path) -> WinnerRecord | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    kernel_name = payload.get("kernel_name")
    speedup = finite_float(payload.get("speedup_vs_triton"))

    if (
        not kernel_name
        or not payload.get("compiles")
        or not payload.get("correct")
        or speedup is None
    ):
        return None

    round_index = payload.get("round_index")

    return WinnerRecord(
        kernel_name=str(kernel_name),
        run_dir=path.parent.name,
        source_path=path,
        round_index=round_index if isinstance(round_index, int) else None,
        speedup_vs_triton=speedup,
        triton_p50=finite_float(payload.get("triton_p50")),
        p50=finite_float(payload.get("p50")),
    )


def collect_best_winners(database_dir: Path) -> dict[str, WinnerRecord]:
    best: dict[str, WinnerRecord] = {}

    for path in sorted(database_dir.glob("*/output_winner_*.json")):
        record = load_winner(path)
        if record is None:
            continue

        current = best.get(record.kernel_name)
        if current is None or record.speedup_vs_triton > current.speedup_vs_triton:
            best[record.kernel_name] = record

    return best


def print_table(best: dict[str, WinnerRecord]) -> None:
    headers = ("Kernel", "Best Speedup", "Run", "Round", "Winner File")

    rows = [
        (
            kernel,
            f"{record.speedup_vs_triton:.4f}x",
            record.run_dir,
            str(record.round_index) if record.round_index is not None else "-",
            str(record.source_path.relative_to(record.source_path.parents[2])),
        )
        for kernel, record in sorted(best.items())
    ]

    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]

    for row in (headers, tuple("-" * width for width in widths), *rows):
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the best valid PTX winner per kernel from the output database."
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("database"),
        help="Directory containing archived run folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.database_dir.is_dir():
        raise NotADirectoryError(
            f"Database directory not found: {args.database_dir}"
        )

    best = collect_best_winners(args.database_dir)

    if not best:
        print(f"No valid winner records found in {args.database_dir}")
        return

    print_table(best)


if __name__ == "__main__":
    main()