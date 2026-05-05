from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from triton_ptx.evaluation import TritonPTXCandidateEvaluator
from triton_ptx.policy import select_winner
from triton_ptx.generator import OpenAIPrompt
from triton_ptx.helpers import JsonDatasetWriter
from triton_ptx.helpers import get_ptx_system_config
from triton_ptx.kernels import resolve_operator
from triton_ptx.prompts import build_prompt_for_operator
from triton_ptx.prompts import build_follow_up_prompt_for_operator


def run_test_time_scaling_loop(
    operator_name: str,
    *,
    rounds: int = 10,
    k: int = 3,
    output_root: Path,
) -> Path:

    operator_cls = resolve_operator(operator_name)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Keep an immutable per-run archive in database/<timestamp>.
    database_root = output_root.parent / "database"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_archive_root = database_root / run_timestamp
    run_archive_root.mkdir(parents=True, exist_ok=True)

    prompter = OpenAIPrompt()
    evaluator = TritonPTXCandidateEvaluator(
        operator_cls
    )
    output_writer = JsonDatasetWriter(dataset_dir=output_root)
    archive_writer = JsonDatasetWriter(dataset_dir=run_archive_root)

    current_prompt = build_prompt_for_operator(operator_cls, num_answers=k)
    (output_root / "initial_prompt.md").write_text(current_prompt + "\n", encoding="utf-8")
    (run_archive_root / "initial_prompt.md").write_text(current_prompt + "\n", encoding="utf-8")

    current_candidates = []

    for round_index in range(1, rounds + 1):
        print(f"=== Iteration {round_index} ===")
        payload = prompter.generate_response(current_prompt)
        answers = payload["answers"]

        round_results = [
            evaluator.evaluate(
                answer,
                round_index=round_index,
                candidate_index=index,
            )
            for index, answer in enumerate(answers, start=1)
        ]  

        output_writer.store(round_index, round_results)
        archive_writer.store(round_index, round_results)

        # Keep the best k only
        current_candidates += round_results
        current_candidates.sort()
        current_candidates = current_candidates[:k]

        winner = select_winner(current_candidates)
        output_writer.store(f"winner_{round_index}", [winner])
        archive_writer.store(f"winner_{round_index}", [winner])

        follow_up_prompt = build_follow_up_prompt_for_operator(
            current_candidates,
            operator_cls,
            num_answers=k,
        )

        follow_up_path = output_root / f"follow_up_iteration_{round_index}.md"
        follow_up_path.write_text(follow_up_prompt + "\n", encoding="utf-8")
        archive_follow_up_path = run_archive_root / f"follow_up_iteration_{round_index}.md"
        archive_follow_up_path.write_text(follow_up_prompt + "\n", encoding="utf-8")

        current_prompt = follow_up_prompt

    return output_root


def parse_args() -> argparse.Namespace:
    default_version, default_target, default_address_size = get_ptx_system_config()

    parser = argparse.ArgumentParser(
        description="Run a manual per-operator PTX test-time scaling loop."
    )
    parser.add_argument("operator", help="Operator class name, for example AddOperator.")
    parser.add_argument("--rounds", type=int, default=4, help="Number of test-time scaling rounds.")
    parser.add_argument("--k", type=int, default=2, help="Sample size for best-of-k candidate selection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="output",
        help="Root directory where run artifacts will be written.",
    )
    parser.add_argument("--keep-cache", action="store_true", help="Keep the Triton cache between candidates.")
    parser.add_argument("--version", default=default_version, help="PTX ISA version to request in prompts.")
    parser.add_argument("--target", default=default_target, help="PTX target to request in prompts.")
    parser.add_argument(
        "--address-size",
        type=int,
        default=default_address_size,
        help="PTX address size to request in prompts.",
    )
    parser.add_argument("--no-dummy-ptrs", action="store_true", help="Do not request dummy pointer parameters.")
    parser.add_argument("--extra-instructions", help="Additional instructions to append to prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.k <= 0:
        raise ValueError("--k must be positive")

    run_test_time_scaling_loop(
        args.operator,
        rounds=args.rounds,
        k=args.k,
        output_root=args.output_dir,
    )


if __name__ == "__main__":
    main()
