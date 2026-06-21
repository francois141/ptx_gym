#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_candidate_record(path: Path) -> dict[str, Any]:
    print("Loading the JSON")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    print("Json loaded")

    return data


def extract_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")

    if isinstance(payload, dict):
        return payload

    # Accept a raw payload file too, so users can remeasure either archived
    # EvaluatedCandidate JSON or just {"ptx": ..., "num_threads_x": ...}.
    if "ptx" in record:
        return record

    raise ValueError('Input JSON must contain a "payload" object or be a raw PTX payload.')


def resolve_kernel_name(record: dict[str, Any], explicit_kernel: str | None) -> str:
    kernel_name = explicit_kernel or record.get("kernel_name")
    if not isinstance(kernel_name, str) or not kernel_name.strip():
        raise ValueError(
            'Kernel name is missing. Include "kernel_name" in the JSON or pass --kernel.'
        )
    return kernel_name


def resolve_index(value: Any, default: int) -> int:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer-like index, got {value!r}") from exc


def remeasure_candidate(
    record: dict[str, Any],
    *,
    kernel_name: str | None = None,
    round_index: int | None = None,
    candidate_index: int | None = None,
    clear_cache: bool = False,
):
    from triton_ptx.evaluation import TritonPTXCandidateEvaluator
    from triton_ptx.kernels import resolve_kernel

    resolved_kernel_name = resolve_kernel_name(record, kernel_name)
    kernel_cls = resolve_kernel(resolved_kernel_name)
    payload = extract_payload(record)

    evaluator = TritonPTXCandidateEvaluator(kernel_cls, clear_cache=clear_cache)
    return evaluator.evaluate(
        payload,
        round_index=(
            round_index
            if round_index is not None
            else resolve_index(record.get("round_index"), 0)
        ),
        candidate_index=(
            candidate_index
            if candidate_index is not None
            else resolve_index(record.get("index"), 0)
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run compile, correctness, and timing measurement for an archived "
            "PTX candidate JSON."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to an EvaluatedCandidate JSON file or raw PTX payload JSON.",
    )
    parser.add_argument(
        "--kernel",
        help='Kernel class name to use when the input JSON has no "kernel_name".',
    )
    parser.add_argument(
        "--round-index",
        type=int,
        help="Override round_index in the output record.",
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        help="Override index in the output record.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Triton cache before remeasuring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the fresh measurement JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = load_candidate_record(args.json_path)
    result = remeasure_candidate(
        record,
        kernel_name=args.kernel,
        round_index=args.round_index,
        candidate_index=args.candidate_index,
        clear_cache=args.clear_cache,
    )
    output = result.to_json()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    print(output)


if __name__ == "__main__":
    main()
