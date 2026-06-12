from __future__ import annotations

import math
from pathlib import Path

from triton_ptx.evaluation import EvaluatedCandidate
from triton_ptx.helpers import get_ptx_constexprs
from triton_ptx.helpers import extract_specification
from triton_ptx.helpers import extract_specification_from_operator
from triton_ptx.prompts import (
    commenting_rules,
    correctness_rules,
    extracted_signature_information,
    follow_up_task,
    output_contract,
    performance_rules,
    ptx_header,
    signature_template,
    triton_kernel_block,
)

BASE_DIR = Path(__file__).resolve().parent
KERNELS_DIR = BASE_DIR.parent / "kernels"


def _format_metric(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _format_constexprs(candidate: EvaluatedCandidate) -> str:
    constexprs = get_ptx_constexprs(candidate.payload)
    if not constexprs:
        return "None"

    return "\n".join(
        f"- {name}: {value!r}"
        for name, value in sorted(constexprs.items())
    )


def _candidate_block(candidate: EvaluatedCandidate) -> str:
    ptx_code = str(candidate.payload.get("ptx", "")).strip() or "<missing PTX payload>"
    diagnostics_block = ""
    if not candidate.correct:
        diagnostics_block = f"""

Compiler stdout:
```text
{candidate.compile_output.strip() or "None"}
```

Compiler stderr:
```text
{candidate.compile_error.strip() or "None"}
```

Timing error:
```text
{candidate.timing_error.strip() or "None"}
```
"""

    return f"""
## Candidate r{candidate.round_index}c{candidate.index}

- Compiles: {"yes" if candidate.compiles else "no"}
- Correct: {"yes" if candidate.correct else "no"}
- Passed: {"yes" if candidate.passed else "no"}
- Message: {candidate.message.strip() or "None"}
- p20: {_format_metric(candidate.p20)}
- p50: {_format_metric(candidate.p50)}
- p80: {_format_metric(candidate.p80)}

Chosen constexpr values:
{_format_constexprs(candidate)}
{diagnostics_block}

PTX:
```ptx
{ptx_code}
```
""".strip()


def candidate_results_block(candidates: list[EvaluatedCandidate]) -> str:
    if not candidates:
        return "\n\n".join(
            [
                "## Candidate Results",
                "No evaluated candidates were provided. Generate strong answers from the Triton kernel specification alone.",
            ]
        )

    sections = [
        "## Candidate Results",
        "Use these results to keep strong ideas from successful candidates and avoid repeating choices that caused compilation, correctness, or performance failures.",
    ]
    sections.extend(_candidate_block(candidate) for candidate in candidates)
    return "\n\n".join(sections)


def follow_up_rules() -> str:
    return """
## Follow-Up Rules

- Study the candidate PTX, compiler feedback, correctness outcome, and runtime percentiles before producing new answers.
- Prefer ideas from candidates that compiled, passed correctness, and achieved lower p50 runtime.
- If a candidate failed, infer the likely cause from the PTX and feedback and avoid repeating that mistake.
- You may combine strong ideas from multiple candidates into a better implementation.
- Improve performance without sacrificing correctness or signature compatibility.
- Return distinct answers, not trivial rewrites of the same PTX.
""".strip()


def prompt_builder(
    spec,
    candidates: list[EvaluatedCandidate],
    *,
    version: str = "8.7",
    target: str = "sm_89",
    address_size: int = 64,
    num_answers: int = 5,
    ptx_signature=None,
) -> str:
    sections = [
        follow_up_task().strip(),
        ptx_header().format(
            version=version,
            target=target,
            address_size=address_size,
        ).strip(),
        extracted_signature_information(spec.parameters),
        signature_template(
            spec.parameters,
            version=version,
            target=target,
            address_size=address_size,
            kernel_name=spec.kernel_name,
            ptx_signature=ptx_signature,
        ),
        correctness_rules(),
        commenting_rules(),
        performance_rules(target, version),
        triton_kernel_block(spec.source),
        candidate_results_block(candidates),
        follow_up_rules(),
        output_contract(),
    ]
    return "\n\n".join(sections)


def discover_kernel_paths():
    return tuple(
        path
        for path in sorted(KERNELS_DIR.glob("*.py"))
        if path.name != "__init__.py"
    )


def build_follow_up_prompt_for_path(
    path,
    candidates: list[EvaluatedCandidate] | None = None,
    *,
    version: str = "8.7",
    target: str = "sm_89",
    address_size: int = 64,
    num_answers: int = 5,
    ptx_signature=None,
):
    spec = extract_specification(path)
    return spec.operator_name, prompt_builder(
        spec,
        candidates or [],
        version=version,
        target=target,
        address_size=address_size,
        num_answers=num_answers,
        ptx_signature=ptx_signature,
    )


def build_follow_up_prompt_for_operator(
    candidates: list[EvaluatedCandidate] | None,
    operator_cls: type,
    *,
    version: str = "8.7",
    target: str = "sm_89",
    address_size: int = 64,
    num_answers: int = 5,
    ptx_signature=None,
):
    spec = extract_specification_from_operator(operator_cls)
    return prompt_builder(
        spec,
        candidates or [],
        version=version,
        target=target,
        address_size=address_size,
        num_answers=num_answers,
        ptx_signature=ptx_signature,
    )


def generate_prompts(
    candidates: list[EvaluatedCandidate] | None = None,
    *,
    version: str = "8.7",
    target: str = "sm_89",
    address_size: int = 64,
    num_answers: int = 5,
):
    out_dir = BASE_DIR / "follow_up"
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    for kernel_path in discover_kernel_paths():
        operator_name, prompt = build_follow_up_prompt_for_path(
            kernel_path,
            candidates,
            version=version,
            target=target,
            address_size=address_size,
            num_answers=num_answers,
        )
        path = out_dir / f"{operator_name}.md"
        path.write_text(prompt + "\n", encoding="utf-8")
        written_files.append(path)

    return written_files


def main():
    generate_prompts()

if __name__ == "__main__":
    main()
