from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
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
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                sanitizer_path,
                "--tool",
                sanitizer_tool,
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
    return {
        "available": True,
        "clean": completed.returncode == 0 and parsed["error_count"] == 0,
        "tool": sanitizer_tool,
        "returncode": completed.returncode,
        **parsed,
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

    request = {
        "kernel_name": kernel_name,
        "candidate": normalized_candidate,
    }
    request_dir = TMP_FILES_DIR / uuid4().hex
    request_dir.mkdir(parents=True)
    request_path = request_dir / "request.json"
    try:
        request_path.write_bytes(orjson.dumps(request))
        tools = (
            SANITIZER_TOOLS
            if sanitizer_tool == "all"
            else (cast(SingleSanitizerTool, sanitizer_tool),)
        )
        reports: list[dict[str, object]] = []
        for tool in tools:
            report = _run_sanitizer_tool(
                sanitizer_path, tool, request_path, timeout_seconds
            )
            if report["clean"] is not True:
                return report
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
        request_path: Path to the serialized launch request.

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

    run_candidate(kernel_name, candidate)


def _main() -> None:
    """Parse and execute a sanitizer child-process request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-request", type=Path, required=True)
    arguments = parser.parse_args()
    _run_request(arguments.run_request)


if __name__ == "__main__":
    _main()
