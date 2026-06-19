import shlex
from pathlib import Path
import subprocess
import tempfile

from triton_ptx.helpers.environment import (
    get_ptx_system_config,
    get_ptxas_path,
)
from triton_ptx.helpers.kernels import get_ptx_code


def compile_ptx(ptx_code):
    ptx = get_ptx_code(ptx_code)
    if ptx is None:
        return {
            "success": False,
            "output": "",
            "error": "Candidate payload does not contain PTX code.",
        }

    _, target, _ = get_ptx_system_config()
    ptxas_path = get_ptxas_path(target)

    def format_output(command, returncode, stdout, stderr):
        command_text = " ".join(shlex.quote(str(part)) for part in command)
        return (
            f"Command: {command_text}\n\n"
            f"Return code: {returncode}\n\n"
            f"STDOUT:\n{stdout}\n\n"
            f"STDERR:\n{stderr}"
        )

    def error_result(command, returncode, stdout, stderr):
        return {
            "success": False,
            "sm": target,
            "output": format_output(
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            ),
            "error": stderr,
        }

    with tempfile.TemporaryDirectory(prefix="triton-ptxas-") as temp_dir:
        temp_path = Path(temp_dir)
        ptx_path = temp_path / "kernel.ptx"
        cubin_path = temp_path / "kernel.cubin"
        ptx_path.write_text(ptx, encoding="utf-8")

        command = [
            str(ptxas_path),
            f"-arch={target}",
            str(ptx_path),
            "-o",
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

    output = format_output(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    success = completed.returncode == 0

    return {
        "success": success,
        "sm": target,
        "output": output,
        "error": ""
        if success
        else completed.stderr.strip()
        or f"ptxas failed with code {completed.returncode}",
    }


def run_ptx_compilation(kernel, ptx_code):
    return compile_ptx(ptx_code)
