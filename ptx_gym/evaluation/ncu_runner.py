from __future__ import annotations

import argparse
from pathlib import Path

import orjson

from ptx_gym.evaluation.run_candidate import run_candidate


def _run_request(request_path: Path) -> None:
    """Execute one serialized Nsight Compute launch request.

    Args:
        request_path: JSON request containing the kernel and candidate.

    Raises:
        TypeError: If the request does not contain the expected value types.
    """
    request = orjson.loads(request_path.read_bytes())
    if not isinstance(request, dict):
        raise TypeError("NCU request must be a JSON object.")

    kernel_name = request.get("kernel_name")
    candidate = request.get("candidate")
    if not isinstance(kernel_name, str):
        raise TypeError("NCU request kernel_name must be a string.")
    if not isinstance(candidate, dict):
        raise TypeError("NCU request candidate must be a JSON object.")

    run_candidate(kernel_name, candidate)


def _main() -> None:
    """Parse and execute an Nsight Compute child-process request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("request_path", type=Path)
    arguments = parser.parse_args()
    _run_request(arguments.request_path)


if __name__ == "__main__":
    _main()
