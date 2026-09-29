from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import tempfile

from ptx_gym.helpers.environment import (
    get_ptx_system_config,
    get_ptxas_path,
)
from ptx_gym.evaluation.types import Payload


@dataclass(frozen=True)
class CompilationResult:
    compiles: bool
    sm: str
    command: list[str]
    flags: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    output: str
    error: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _format_output(
    command: list[str],
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> str:
    command_text = " ".join(shlex.quote(str(part)) for part in command)
    return (
        f"Command: {command_text}\n\n"
        f"Return code: {returncode}\n\n"
        f"STDOUT:\n{stdout}\n\n"
        f"STDERR:\n{stderr}"
    )


def compile_ptx(payload: Payload) -> CompilationResult:
    if not isinstance(payload, Payload):
        raise TypeError("payload must be a Payload instance.")

    ptx = payload.ptx
    _, target, _ = get_ptx_system_config()
    ptxas_path = get_ptxas_path(target)
    flags = [f"-arch={target}", "-v", "--warning-as-error", "-o"]

    def error_result(
        command: list[str],
        returncode: int | None,
        stdout: str,
        stderr: str,
    ) -> CompilationResult:
        return CompilationResult(
            compiles=False,
            sm=target,
            command=command,
            flags=flags,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            output=_format_output(command, returncode, stdout, stderr),
            error=stderr,
        )

    with tempfile.TemporaryDirectory(prefix="triton-ptxas-") as temp_dir:
        temp_path = Path(temp_dir)
        ptx_path = temp_path / "kernel.ptx"
        cubin_path = temp_path / "kernel.cubin"
        ptx_path.write_text(ptx, encoding="utf-8")

        command = [
            str(ptxas_path),
            *flags[:3],
            str(ptx_path),
            flags[3],
            str(cubin_path),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            return error_result(
                command,
                None,
                "",
                f"ptxas not found: {exc}",
            )
        except OSError as exc:
            return error_result(
                command,
                None,
                "",
                f"Failed to run ptxas: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            return error_result(
                command,
                None,
                "",
                f"Failed to compile PTX: {type(exc).__name__}: {exc}",
            )

    output = _format_output(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    success = completed.returncode == 0

    return CompilationResult(
        compiles=success,
        sm=target,
        command=command,
        flags=flags,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output=output,
        error=""
        if success
        else completed.stderr.strip()
        or f"ptxas failed with code {completed.returncode}",
    )


def run_ptx_compilation(kernel, payload: Payload) -> CompilationResult:
    return compile_ptx(payload)
