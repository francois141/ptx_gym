from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import orjson

from triton_ptx.evaluation.types import Payload

SingleSanitizerTool = Literal["memcheck", "racecheck", "synccheck", "initcheck"]
SanitizerTool = Literal["all", "memcheck", "racecheck", "synccheck", "initcheck"]
SANITIZER_TOOLS: tuple[SingleSanitizerTool, ...] = (
    "memcheck",
    "racecheck",
    "synccheck",
    "initcheck",
)
SANITIZER_OPTIONS = ("all", *SANITIZER_TOOLS)
ERROR_EXIT_CODE = 99
MAX_REPORTED_ERRORS = 10
SANITIZER_ENV_VAR = "PTX_MEMORY_SANITIZER"
TMP_FILES_DIR = Path(__file__).resolve().parents[2] / "tmp_files"
SANITIZER_PREFIX = "========= "
SUMMARY_PATTERN = re.compile(
    r"(?:ERROR SUMMARY:|RACECHECK SUMMARY:.*?\()\s*(\d+)\s+errors?",
    re.IGNORECASE,
)
LOCATION_PATTERN = re.compile(r"candidate\.ptx:(\d+)(?::\d+)?")
FILE_DIRECTIVE_PATTERN = re.compile(r"^\s*\.file\s+(\d+)\b", re.MULTILINE)


def _sanitizer_input_kwargs(kernel_name: str) -> list[dict[str, int]]:
    """Return the kernel's fixed input configuration."""
    del kernel_name
    return [{}]


def _add_ptx_line_information(ptx: str) -> str:
    """Map executable PTX instructions to their original source lines.

    Args:
        ptx: Candidate PTX source.

    Returns:
        PTX containing ``.file`` and ``.loc`` directives for Compute Sanitizer.
    """
    file_ids = (int(match) for match in FILE_DIRECTIVE_PATTERN.findall(ptx))
    file_id = max(file_ids, default=0) + 1
    annotated_lines: list[str] = []
    file_directive = f'.file {file_id} "candidate.ptx"'
    file_directive_added = False
    brace_depth = 0

    for line_number, line in enumerate(ptx.splitlines(), start=1):
        stripped = line.strip()
        if (
            not file_directive_added
            and not stripped.startswith("//")
            and re.search(r"\.(?:entry|func)\b", stripped)
        ):
            annotated_lines.append(file_directive)
            file_directive_added = True
        is_instruction = (
            brace_depth > 0
            and stripped.endswith(";")
            and not stripped.startswith((".", "//"))
        )
        if is_instruction:
            indentation = line[: len(line) - len(line.lstrip())]
            annotated_lines.append(f"{indentation}.loc {file_id} {line_number} 0")
        annotated_lines.append(line)
        code = line.split("//", maxsplit=1)[0]
        brace_depth += code.count("{") - code.count("}")

    if not file_directive_added:
        annotated_lines.append(file_directive)

    if ptx.endswith("\n"):
        return "\n".join(annotated_lines) + "\n"
    return "\n".join(annotated_lines)


def _reported_ptx_locations(
    diagnostics: list[str], ptx: str
) -> list[dict[str, object]]:
    """Extract unique candidate PTX locations named in sanitizer diagnostics.

    Args:
        diagnostics: Normalized Compute Sanitizer diagnostic lines.
        ptx: Original, unannotated PTX source.

    Returns:
        Locations containing original one-based line numbers and source text.
    """
    source_lines = ptx.splitlines()
    line_numbers = {
        int(match.group(1))
        for diagnostic in diagnostics
        for match in LOCATION_PATTERN.finditer(diagnostic)
    }
    return [
        {
            "line": line_number,
            "source": source_lines[line_number - 1].strip(),
        }
        for line_number in sorted(line_numbers)
        if line_number <= len(source_lines)
    ]


def _parse_sanitizer_output(output: str) -> dict[str, object]:
    lines = [
        line
        for raw_line in output.splitlines()
        if (line := raw_line.strip().removeprefix(SANITIZER_PREFIX).strip())
    ]
    summary = next(filter(None, map(SUMMARY_PATTERN.search, reversed(lines))), None)
    return {
        "error_count": int(summary.group(1)) if summary else None,
        "diagnostics": [
            line
            for line in lines
            if not line.lower().startswith(
                (
                    "compute-sanitizer",
                    "copyright",
                    "error summary:",
                    "racecheck summary:",
                )
            )
        ],
    }


def _run_sanitizer_tool(
    sanitizer_path: str,
    sanitizer_tool: SingleSanitizerTool,
    request_path: Path,
    timeout_seconds: int,
    ptx: str,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                sanitizer_path,
                "--tool",
                sanitizer_tool,
                "--target-processes",
                "application-only",
                "--show-backtrace",
                "device",
                "--print-limit",
                str(MAX_REPORTED_ERRORS),
                "--error-exitcode",
                str(ERROR_EXIT_CODE),
                sys.executable,
                "-m",
                "triton_ptx.evaluation.sanitizer",
                "--run-request",
                str(request_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        timed_out = isinstance(error, subprocess.TimeoutExpired)
        return {
            "available": True,
            "clean": False,
            "tool": sanitizer_tool,
            "error": (
                f"Compute Sanitizer timed out after {timeout_seconds} seconds."
                if timed_out
                else f"Failed to start Compute Sanitizer: {error}"
            ),
            **({"diagnostics": [str(error)]} if timed_out else {}),
        }

    parsed = _parse_sanitizer_output(f"{completed.stdout}\n{completed.stderr}")
    diagnostics = cast(list[str], parsed["diagnostics"])
    ptx_locations = _reported_ptx_locations(diagnostics, ptx)
    location_error = (
        "PTX memory error at "
        + ", ".join(
            f"line {location['line']}: {location['source']}"
            for location in ptx_locations
        )
        if ptx_locations
        else None
    )
    return {
        "available": True,
        "clean": completed.returncode == 0 and parsed["error_count"] == 0,
        "tool": sanitizer_tool,
        "returncode": completed.returncode,
        **parsed,
        "ptx_locations": ptx_locations,
        **({"error": location_error} if location_error else {}),
    }


def diagnose_ptx(
    kernel_name: str,
    candidate: dict[str, object],
    sanitizer_tool: str = "all",
    timeout_seconds: int = 120,
) -> dict[str, object]:
    """Run a PTX candidate under NVIDIA Compute Sanitizer.

    Args:
        kernel_name: Registered kernel class used to construct inputs and the grid.
        candidate: PTX text and launch dimensions accepted by ``Payload``.
        sanitizer_tool: Analysis to perform, or ``all`` to run every analysis.
        timeout_seconds: Maximum runtime in seconds for each analysis.

    Returns:
        Structured availability, execution, and diagnostic results.

    Raises:
        ValueError: If an option or candidate payload is invalid.
    """
    invalid = (
        (
            sanitizer_tool not in SANITIZER_OPTIONS,
            f"sanitizer_tool must be one of: {', '.join(SANITIZER_OPTIONS)}.",
        ),
        (timeout_seconds <= 0, "timeout_seconds must be positive."),
    )
    if message := next((message for failed, message in invalid if failed), None):
        raise ValueError(message)

    normalized_candidate = Payload.from_input(candidate).to_launch_dict()
    sanitizer_path = os.environ.get(SANITIZER_ENV_VAR)
    if not sanitizer_path:
        return {
            "available": False,
            "clean": False,
            "tool": sanitizer_tool,
            "error": (
                f"{SANITIZER_ENV_VAR} is not set to the Compute Sanitizer executable path."
            ),
        }

    input_kwargs_list = _sanitizer_input_kwargs(kernel_name)
    request = {
        "kernel_name": kernel_name,
        "candidate": {
            **normalized_candidate,
            "ptx": _add_ptx_line_information(cast(str, normalized_candidate["ptx"])),
        },
        "input_kwargs": [{}],
    }
    request_dir = TMP_FILES_DIR / uuid4().hex
    request_dir.mkdir(parents=True)
    request_path = request_dir / "request.json"
    try:
        tools = (
            SANITIZER_TOOLS
            if sanitizer_tool == "all"
            else (cast(SingleSanitizerTool, sanitizer_tool),)
        )
        reports: list[dict[str, object]] = []
        for tool in tools:
            for input_kwargs in input_kwargs_list:
                request["input_kwargs"] = [input_kwargs]
                request_path.write_bytes(orjson.dumps(request))
                report = _run_sanitizer_tool(
                    sanitizer_path,
                    tool,
                    request_path,
                    timeout_seconds,
                    cast(str, normalized_candidate["ptx"]),
                )
                if report["clean"] is not True:
                    return {**report, "input_kwargs": input_kwargs}
                reports.append(report)

        if sanitizer_tool != "all":
            return reports[0]
        return {
            "available": all(report["available"] is True for report in reports),
            "clean": all(report["clean"] is True for report in reports),
            "tool": "all",
            "reports": reports,
        }
    finally:
        shutil.rmtree(request_dir)


def _run_request(request_path: Path) -> None:
    """Load and execute one sanitizer child-process request.

    Args:
        request_path: Path to the serialized launch request. Requests may include
            ``input_kwargs`` as a list of integer keyword mappings.

    Raises:
        TypeError: If the serialized request has an invalid shape.
    """
    from triton_ptx.evaluation.run_candidate import run_candidate

    request = orjson.loads(request_path.read_bytes())
    if not isinstance(request, dict):
        raise TypeError("Sanitizer request must be a JSON object.")

    kernel_name = request.get("kernel_name")
    candidate = request.get("candidate")
    if not isinstance(kernel_name, str):
        raise TypeError("Sanitizer request kernel_name must be a string.")
    if not isinstance(candidate, dict):
        raise TypeError("Sanitizer request candidate must be a JSON object.")

    input_kwargs = request.get("input_kwargs", [{}])
    if not isinstance(input_kwargs, list):
        raise TypeError("Sanitizer request input_kwargs must be a list.")

    for kwargs in input_kwargs:
        if not isinstance(kwargs, Mapping) or any(
            not isinstance(name, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for name, value in kwargs.items()
        ):
            raise TypeError(
                "Sanitizer request input_kwargs entries must map strings to positive integers."
            )
        run_candidate(kernel_name, candidate, kwargs)


def _main() -> None:
    """Parse and execute a sanitizer child-process request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-request", type=Path, required=True)
    arguments = parser.parse_args()
    _run_request(arguments.run_request)


if __name__ == "__main__":
    _main()
