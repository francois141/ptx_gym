from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PTXSignatureParameter:
    name: str
    ptx_type: str


# Matches a PTX kernel/function declaration and captures:
#   - name: kernel/function name
#   - params: contents of the parameter list
#
# Example:
#   .visible .entry my_kernel(
#       .param .u64 input,
#       .param .u32 size
#   )
_PTX_PROTOTYPE_RE = re.compile(
    r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(?P<name>\w+)\s*\((?P<params>.*?)\)",
    re.DOTALL,
)

# Extracts the parameter name from a PTX parameter declaration.
#
# Examples:
#   ".param .u64 input"      -> input
#   ".param .u64 buffer[8]"  -> buffer
_PTX_PARAM_NAME_RE = re.compile(
    r"(?P<name>[A-Za-z_$][\w$]*)(?:\[[^\]]+\])?\s*$"
)

# Matches scalar PTX types such as:
#   .u32, .u64
#   .s32, .s64
#   .f16, .f32, .f64
#   .bf16, .bf16x2
#   .f16x2
#   .b8, .b16, .b32, .b64, .b128
#   .pred
_PTX_TYPE_RE = re.compile(
    r"\.(?:b\d+|u\d+|s\d+|f\d+(?:x\d+)?|bf16(?:x2)?|pred)"
)


def parse_ptx_signature(ptx: str) -> tuple[PTXSignatureParameter, ...]:
    """Parse the ordered parameters from a PTX entry or function signature."""
    match = _PTX_PROTOTYPE_RE.search(ptx)
    if match is None:
        raise ValueError("Could not find a PTX .entry/.func signature.")

    parameters: list[PTXSignatureParameter] = []

    for raw_param in match.group("params").split(","):
        declaration = raw_param.strip()
        if not declaration:
            continue

        name_match = _PTX_PARAM_NAME_RE.search(declaration)
        if name_match is None:
            raise ValueError(
                f"Could not parse PTX parameter name from: {declaration}"
            )

        # Everything before the parameter name contains PTX qualifiers.
        #
        # Example:
        #   ".param .align 8 .b8 input[16]"
        # becomes:
        #   ".param .align 8 .b8"
        prefix = declaration[: name_match.start()].strip()

        # Extract PTX tokens such as:
        #   .param
        #   .align
        #   .ptr
        #   .global
        #   .u64
        #   .b8
        dot_tokens = re.findall(r"\.[A-Za-z_][\w]*", prefix)

        # The parameter type is the last PTX token that is not a
        # storage/alignment/address-space qualifier.
        ptx_type = next(
            (
                token
                for token in reversed(dot_tokens)
                if token
                not in {
                    ".const",
                    ".global",
                    ".local",
                    ".param",
                    ".ptr",
                    ".shared",
                    ".tex",
                    ".align",
                }
            ),
            None,
        )

        assert ptx_type is not None, (
            f"Could not determine PTX type from declaration: {declaration}"
        )

        assert _PTX_TYPE_RE.fullmatch(ptx_type), (
            f"Unexpected PTX type {ptx_type!r} in declaration: {declaration}"
        )

        parameters.append(
            PTXSignatureParameter(
                name=name_match.group("name"),
                ptx_type=ptx_type,
            )
        )

    return tuple(parameters)
