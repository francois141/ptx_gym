from __future__ import annotations

from pathlib import Path

from triton_ptx.helpers.kernels import (
    extract_specification,
    extract_specification_from_operator,
)
from triton_ptx.client.prompts import (
    commenting_rules,
    correctness_rules,
    constexpr_values_block,
    extracted_signature_information,
    initial_task,
    num_warps_block,
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
    *,
    version,
    target,
    address_size,
    num_answers=5,
    ptx_signature=None,
):
    sections = [
        initial_task().strip(),
        ptx_header().format(
            version=version,
            target=target,
            address_size=address_size,
        ).strip(),
        extracted_signature_information(spec.parameters),
        constexpr_values_block(spec),
        num_warps_block(spec),
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
        performance_rules(target, version, spec),
        triton_kernel_block(spec.source),
        output_contract(spec),
    ]
    return "\n\n".join(sections)


def discover_kernel_paths():
    return tuple(
        path
        for path in sorted(KERNELS_DIR.glob("*.py"))
        if path.name != "__init__.py"
    )


def build_prompt_for_path(
    path,
    *,
    version,
    target,
    address_size,
    num_answers=5,
):
    spec = extract_specification(path)
    return spec.operator_name, prompt_builder(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        num_answers=num_answers,
    )


def build_prompt_for_operator(
    operator,
    *,
    version,
    target,
    address_size,
    num_answers=5,
    ptx_signature=None,
):
    spec = extract_specification_from_operator(operator)
    return prompt_builder(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        num_answers=num_answers,
        ptx_signature=ptx_signature,
    )


def generate_prompts(*, version, target, address_size, num_answers=3):
    out_dir = BASE_DIR / "initial"
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    for kernel_path in discover_kernel_paths():
        operator_name, prompt = build_prompt_for_path(
            kernel_path,
            version=version,
            target=target,
            address_size=address_size,
            num_answers=num_answers,
        )
        path = out_dir / f"{operator_name}.md"
        path.write_text(prompt + "\n", encoding="utf-8")
        written_files.append(path)

    return written_files
