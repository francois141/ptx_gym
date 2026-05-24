from __future__ import annotations

from pathlib import Path

from triton_ptx.helpers import (
    extract_specification,
    extract_specification_from_operator,
)
from triton_ptx.prompts import (
    correctness_rules,
    extracted_signature_information,
    initial_task,
    output_contract,
    performance_rules,
    ptx_header,
    signature_template,
    triton_kernel_block,
)

BASE_DIR = Path(__file__).resolve().parent
KERNELS_DIR = BASE_DIR.parent / "kernels"

# TODO: Dehardcode the target here
def prompt_builder(
    spec,
    version="8.7",
    target="sm_89",
    address_size=64,
    num_answers=5,
):
    sections = [
        initial_task(num_answers=num_answers).strip(),
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
        ),
        correctness_rules(),
        performance_rules(target, version),
        triton_kernel_block(spec.source),
        output_contract(num_answers=num_answers),
    ]
    return "\n\n".join(sections)


def discover_kernel_paths():
    return tuple(
        path
        for path in sorted(KERNELS_DIR.glob("*.py"))
        if path.name != "__init__.py"
    )


def build_prompt_for_path(path, *, num_answers=5):
    spec = extract_specification(path)
    return spec.operator_name, prompt_builder(spec, num_answers=num_answers)


def build_prompt_for_operator(operator, *, num_answers=5):
    spec = extract_specification_from_operator(operator)
    return prompt_builder(spec, num_answers=num_answers)


def generate_prompts(*, num_answers=3):
    out_dir = BASE_DIR / "initial"
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    for kernel_path in discover_kernel_paths():
        operator_name, prompt = build_prompt_for_path(kernel_path, num_answers=num_answers)
        path = out_dir / f"{operator_name}.md"
        path.write_text(prompt + "\n", encoding="utf-8")
        written_files.append(path)

    return written_files

def main():
    for path in generate_prompts():
        print(path)

if __name__ == "__main__":
    main()
