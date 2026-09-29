from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ptx_gym.helpers.triton import get_kernel_shared_memory_bytes
from ptx_gym.kernels.base import TritonPTXKernel

VOLTA_BIN_ENV_VAR = "VOLTA_BIN"
VOLTA_DIR = Path(__file__).resolve().parents[2] / "volta"
DEFAULT_VOLTA_BIN = VOLTA_DIR / "target" / "release" / "volta"
VOLTA_TIMEOUT_SECONDS = 600
MAX_REPORTED_OUTPUT_CHARS = 4000
EQUIVALENT_PATTERN = re.compile(r"^EQUIVALENT", re.MULTILINE)
# Every candidate declares the template's ``.extern .shared`` window, and Volta
# rejects an extern window with no size, so kernels whose launcher allocates no
# dynamic shared memory still get this small window.
MIN_DYN_SHARED_BYTES = 4
# Output elements compared with the spec, taken in ascending index order from
# CTA 0's footprint; symbolic execution of the CTA dominates the run time.
VOLTA_SAMPLE = 4


def volta_binary() -> Path:
    """Return the Volta CLI path from ``VOLTA_BIN`` or the bundled checkout."""
    path = Path(os.environ.get(VOLTA_BIN_ENV_VAR, DEFAULT_VOLTA_BIN))
    if not path.is_file():
        raise FileNotFoundError(
            f"Volta binary not found at {path}. Build it with "
            f"`cargo build -p volta_cli --release` in {VOLTA_DIR} or set "
            f"{VOLTA_BIN_ENV_VAR}."
        )
    return path


def verify_ptx_with_volta(
    operator: TritonPTXKernel, launch_payload: dict[str, Any]
) -> dict[str, Any]:
    """Prove candidate PTX equivalent to the operator's Volta spec.

    Args:
        operator: Kernel whose ``volta_arguments`` describe the spec and launch.
        launch_payload: Candidate PTX and thread dimensions.

    Returns:
        Report with the verdict, the Volta exit code, and the tail of its output.
    """
    spec_path, arguments = operator.volta_arguments()
    if "--dyn-shared" not in arguments:
        dyn_shared = max(get_kernel_shared_memory_bytes(operator), MIN_DYN_SHARED_BYTES)
        arguments = [*arguments, "--dyn-shared", str(dyn_shared)]
    block = ",".join(
        str(launch_payload.get(key) or 1)
        for key in ("num_threads_x", "num_threads_y", "num_threads_z")
    )
    with tempfile.TemporaryDirectory(prefix="volta_") as directory:
        ptx_path = Path(directory) / "candidate.ptx"
        ptx_path.write_text(launch_payload["ptx"], encoding="utf-8")
        completed = subprocess.run(
            [
                str(volta_binary()),
                "verify",
                str(ptx_path),
                str(spec_path),
                "--no-log-file",
                "--no-profile",
                "-b",
                block,
                *arguments,
                "--sample",
                str(VOLTA_SAMPLE),
                "--verify-numeric",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=VOLTA_TIMEOUT_SECONDS,
        )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    return {
        "equivalent": EQUIVALENT_PATTERN.search(output) is not None,
        "return_code": completed.returncode,
        "output": output[-MAX_REPORTED_OUTPUT_CHARS:],
    }
