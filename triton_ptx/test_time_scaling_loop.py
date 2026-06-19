from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from triton_ptx.evaluation import TritonPTXCandidateEvaluator
from triton_ptx.generator import build_prompter
from triton_ptx.helpers.environment import get_ptx_system_config
from triton_ptx.helpers.ptx import parse_ptx_signature
from triton_ptx.helpers.storage import JsonDatasetWriter, ensure_safe_folder_name
from triton_ptx.helpers.triton import dump_kernel_ptx
from triton_ptx.kernels import resolve_kernel
from triton_ptx.prompts import build_follow_up_prompt_for_operator
from triton_ptx.prompts import build_prompt_for_operator
from triton_ptx.prompts import build_repair_prompt_for_operator

DEFAULT_CONFIG = {
    "loop": {
        "rounds": 10,
        "k": 2,
        "max_retries": 3,
    },
    "generator": {
        "provider": "openai",
        "model": None,
        "options": {},
    },
    "storage": {
        "database_dir": "database",
    },
}


def needs_compile_or_verification_retry(candidate) -> bool:
    if not candidate.compiles:
        return True

    message = (candidate.message or "").lower()
    return (
        candidate.compiles
        and not candidate.correct
        and (
            "correctness check" in message
            or "verification" in message
            or "verifier" in message
        )
    )


def load_config(config_path: Path | str | None = None) -> DictConfig:
    config = OmegaConf.create(DEFAULT_CONFIG)
    if config_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_path))
    return config


def validate_config(config: DictConfig) -> None:
    if config.loop.rounds <= 0:
        raise ValueError("loop.rounds must be positive")
    if config.loop.k <= 0:
        raise ValueError("loop.k must be positive")
    if config.loop.max_retries < 0:
        raise ValueError("loop.max_retries must be non-negative")


def run_test_time_scaling_loop(
    kernel_name: str,
    *,
    config: DictConfig,
) -> Path:

    kernel_cls = resolve_kernel(kernel_name)

    rounds = config.loop.rounds
    k = config.loop.k
    max_retries = config.loop.max_retries
    database_root = Path(config.storage.database_dir)
    database_root.mkdir(parents=True, exist_ok=True)

    # Keep an immutable per-run archive in database/<timestamp>_<kernel>.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_archive_root = database_root / f"{run_timestamp}_{ensure_safe_folder_name(kernel_cls.__name__)}"
    run_archive_root.mkdir(parents=True, exist_ok=True)

    print(f"Compiling baseline Triton PTX for {kernel_cls.__name__}")
    baseline_ptx = dump_kernel_ptx(kernel_cls())
    ptx_signature = parse_ptx_signature(baseline_ptx)
    version, target, address_size = get_ptx_system_config()
    (run_archive_root / "compiled_triton_kernel.ptx").write_text(baseline_ptx + "\n", encoding="utf-8")

    OmegaConf.save(config, run_archive_root / "config.yaml")

    prompter = build_prompter(
        config.generator.provider,
        model=config.generator.model,
        options=OmegaConf.to_container(config.generator.options, resolve=True) or {},
    )
    evaluator = TritonPTXCandidateEvaluator(
        kernel_cls,
    )
    archive_writer = JsonDatasetWriter(dataset_dir=run_archive_root)

    current_prompt = build_prompt_for_operator(
        kernel_cls,
        num_answers=k,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
    )

    (run_archive_root / "initial_prompt.md").write_text(current_prompt + "\n", encoding="utf-8")

    current_candidates = []

    for round_index in range(1, rounds + 1):
        print(f"=== Iteration {round_index} ===")
        answers = prompter.generate_response(current_prompt, num_answers=k)

        round_results = []
        repair_results = []

        for index, answer in enumerate(answers, start=1):
            result = evaluator.evaluate(
                answer,
                round_index=round_index,
                candidate_index=index,
            )
            original_result = result

            retry_index = 1
            while needs_compile_or_verification_retry(result) and retry_index <= max_retries:
                if retry_index == 1:
                    repair_results.append(original_result)

                repair_prompt = build_repair_prompt_for_operator(
                    result,
                    kernel_cls,
                    retry_index=retry_index,
                    max_retries=max_retries,
                    version=version,
                    target=target,
                    address_size=address_size,
                    ptx_signature=ptx_signature,
                )

                repair_prompt_name = (
                    f"repair_prompt_iteration_{round_index}"
                    f"_candidate_{index}_retry_{retry_index}.md"
                )
                (run_archive_root / repair_prompt_name).write_text(
                    repair_prompt + "\n",
                    encoding="utf-8",
                )

                repaired_answer = prompter.generate_response(
                    repair_prompt,
                    num_answers=1,
                )[0]

                result = evaluator.evaluate(
                    repaired_answer,
                    round_index=round_index,
                    candidate_index=index,
                )
                repair_results.append(result)
                retry_index += 1

            round_results.append(result)

        archive_writer.store(round_index, round_results)

        if repair_results:
            archive_writer.store(f"repairs_{round_index}", repair_results)

        # Keep the best k only
        current_candidates += round_results
        current_candidates.sort()
        current_candidates = current_candidates[:k]

        winner = sorted(current_candidates)[0]
        archive_writer.store(f"winner_{round_index}", [winner])

        follow_up_prompt = build_follow_up_prompt_for_operator(
            current_candidates,
            kernel_cls,
            num_answers=k,
            version=version,
            target=target,
            address_size=address_size,
            ptx_signature=ptx_signature,
        )

        archive_follow_up_path = run_archive_root / f"follow_up_iteration_{round_index}.md"
        archive_follow_up_path.write_text(follow_up_prompt + "\n", encoding="utf-8")

        current_prompt = follow_up_prompt

    return run_archive_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual per-kernel PTX test-time scaling loop."
    )
    parser.add_argument("kernel", help="Kernel class name, for example AddKernel.")
    parser.add_argument(
        "--config",
        type=Path,
        help="OmegaConf YAML configuration path. If omitted, the in-file default config is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    run_test_time_scaling_loop(
        args.kernel,
        config=config,
    )


if __name__ == "__main__":
    main()
