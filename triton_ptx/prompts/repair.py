from __future__ import annotations


from triton_ptx.evaluation import EvaluatedCandidate
from triton_ptx.helpers.kernels import extract_specification_from_operator
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.prompts.blocks import (
    commenting_rules,
    correctness_rules,
    constexpr_values_block,
    extracted_signature_information,
    num_warps_block,
    output_contract,
    ptx_header,
    signature_template,
    triton_kernel_block,
)
from triton_ptx.prompts.next import candidate_results_block


def repair_task(retry_index: int, max_retries: int) -> str:
    return f"""
# PTX Isolated Candidate Repair

You are given one failed PTX candidate for a Triton kernel.
Repair only this candidate and return one replacement answer.

This is repair attempt {retry_index} of {max_retries}. Focus on the concrete
compiler and verification feedback below. Do not blend in other candidates or
produce multiple alternatives.
""".strip()


def repair_rules() -> str:
    return """
## Repair Rules

- Fix the smallest part of the PTX needed to address the reported compilation or verification failure.
- Preserve the exact PTX header, kernel entry name, runtime argument order, and launch metadata keys.
- If compilation failed, prioritize valid PTX syntax, declarations, parameter loads, address spaces, and instruction types.
- If correctness verification failed, prioritize matching the Triton semantics, masks, indexing, and stores exactly.
- Return one complete replacement candidate, not a patch or explanation.
""".strip()


def prompt_builder(
    spec,
    failed_candidate: EvaluatedCandidate,
    *,
    retry_index: int,
    max_retries: int,
    version: str,
    target: str,
    address_size: int,
    ptx_signature=None,
) -> str:
    sections = [
        repair_task(retry_index, max_retries),
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
        triton_kernel_block(spec.source),
        candidate_results_block([failed_candidate]),
        repair_rules(),
        output_contract(spec),
    ]
    return "\n\n".join(sections)


def build_repair_prompt_for_operator(
    failed_candidate: EvaluatedCandidate,
    operator_cls: type[TritonPTXKernel],
    *,
    retry_index: int,
    max_retries: int,
    version: str,
    target: str,
    address_size: int,
    ptx_signature=None,
) -> str:
    spec = extract_specification_from_operator(operator_cls)
    return prompt_builder(
        spec,
        failed_candidate,
        retry_index=retry_index,
        max_retries=max_retries,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
    )
