import shlex
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import tempfile

from triton_ptx.helpers.environment import (
    get_ptx_system_config,
    get_ptxas_path,
)
from triton_ptx.helpers.kernels import get_ptx_code


class CompilationRunnerBase(ABC):
    def __init__(self, kernel, ptx_code):
        if ptx_code is None:
            raise ValueError("ptx_code is required and cannot be None")
        self.kernel = kernel
        self.ptx_code = ptx_code

    @abstractmethod
    def run(self):
        pass


class PTXCompilationRunner(CompilationRunnerBase):
    def run(self):
        ptx = get_ptx_code(self.ptx_code)
        if ptx is None:
            return {
                "success": False,
                "output": "",
                "error": "Candidate payload does not contain PTX code.",
            }

        _, target, _ = get_ptx_system_config()
        ptxas_path = get_ptxas_path(target)

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
                return self._error_result(
                    command,
                    None,
                    "",
                    f"ptxas not found: {exc}",
                )
            except OSError as exc:
                return self._error_result(
                    command,
                    None,
                    "",
                    f"Failed to run ptxas: {type(exc).__name__}: {exc}",
                )
            except Exception as exc:
                return self._error_result(
                    command,
                    None,
                    "",
                    f"Failed to compile PTX: {type(exc).__name__}: {exc}",
                )

        output = self._format_output(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

        success = completed.returncode == 0

        return {
            "success": success,
            "output": output,
            "error": ""
            if success
            else completed.stderr.strip()
            or f"ptxas failed with code {completed.returncode}",
        }

    @staticmethod
    def _error_result(command, returncode, stdout, stderr):
        return {
            "success": False,
            "output": PTXCompilationRunner._format_output(
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            ),
            "error": stderr,
        }

    @staticmethod
    def _format_output(command, returncode, stdout, stderr):
        command_text = " ".join(shlex.quote(str(part)) for part in command)
        return (
            f"Command: {command_text}\n\n"
            f"Return code: {returncode}\n\n"
            f"STDOUT:\n{stdout}\n\n"
            f"STDERR:\n{stderr}"
        )


def run_ptx_compilation(kernel, ptx_code):
    return PTXCompilationRunner(kernel, ptx_code).run()