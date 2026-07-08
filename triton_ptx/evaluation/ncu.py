from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import orjson

from triton_ptx.evaluation.types import Payload

NCU_ENV_VAR = "NCU_PATH"
TMP_FILES_DIR = Path(__file__).resolve().parents[2] / "tmp_files"


@dataclass(frozen=True)
class NCUResult:
    """Result of profiling one candidate with NVIDIA Nsight Compute.

    Attributes:
        output: Detailed Nsight Compute CSV output.
        error: Nsight Compute diagnostics written to standard error.
        return_code: Process exit status, or ``None`` when profiling did not run.
    """

    output: str
    error: str
    return_code: int | None

    def to_dict(self) -> dict[str, str | int | None | bool]:
        """Return a JSON-serializable profiling report.

        Returns:
            Availability, process status, and complete profiler output.
        """
        return {
            "available": self.return_code is not None,
            **asdict(self),
        }

def profile_ptx_with_ncu(
    kernel_name: str,
    payload: Payload,
    *,
    input_size: int = 128,
    timeout_seconds: int = 300,
) -> NCUResult:
    """Collect detailed hardware metrics for one PTX kernel launch.

    Args:
        kernel_name: Registered kernel class used to launch the candidate.
        payload: Compiled PTX and launch dimensions.
        input_size: Representative size for configurable random inputs.
        timeout_seconds: Maximum Nsight Compute runtime in seconds.

    Returns:
        Nsight Compute CSV output, diagnostics, and process status. A missing
        executable or launch failure is represented in the result instead of
        failing an otherwise valid evaluation.

    Raises:
        ValueError: If ``input_size`` or ``timeout_seconds`` is not positive.
    """
    if input_size <= 0:
        raise ValueError("input_size must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    ncu_path = os.environ.get(NCU_ENV_VAR)
    if ncu_path is None:
        return NCUResult(
            output="",
            error=f"ncu not found; set {NCU_ENV_VAR} to the executable path.",
            return_code=None,
        )

    request_dir = TMP_FILES_DIR / uuid4().hex
    request_dir.mkdir(parents=True)
    request_path = request_dir / "request.json"
    try:
        request_path.write_bytes(
            orjson.dumps(
                {
                    "kernel_name": kernel_name,
                    "candidate": payload.to_launch_dict(),
                    "input_size": input_size,
                }
            )
        )
        try:
            completed = subprocess.run(
                [
                    ncu_path,
                    "--target-processes",
                    "application-only",
                    "--set",
                    "full",
                    "--page",
                    "raw",
                    "--csv",
                    sys.executable,
                    "-m",
                    "triton_ptx.evaluation.ncu_runner",
                    str(request_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_output = exc.stdout
            return NCUResult(
                output=(
                    timeout_output.decode(errors="replace")
                    if isinstance(timeout_output, bytes)
                    else timeout_output or ""
                ),
                error=f"Nsight Compute timed out after {timeout_seconds} seconds.",
                return_code=None,
            )
        except OSError as exc:
            return NCUResult(
                output="",
                error=f"Failed to start Nsight Compute: {exc}",
                return_code=None,
            )

        return NCUResult(
            output=completed.stdout,
            error=completed.stderr,
            return_code=completed.returncode,
        )
    finally:
        shutil.rmtree(request_dir)
