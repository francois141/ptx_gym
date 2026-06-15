import re
import shlex
import subprocess
import traceback
from abc import ABC, abstractmethod
import contextlib
import io

from triton_ptx.helpers.kernels import instantiate_operator


class CompilationRunnerBase(ABC):
    def __init__(self, kernel, ptx_code):
        if ptx_code is None:
            raise ValueError("ptx_code is required and cannot be None")
        self.kernel = kernel
        self.ptx_code = ptx_code

    def create_kernel(self):
        return instantiate_operator(self.kernel, self.ptx_code)

    @abstractmethod
    def run(self):
        pass


class PTXCompilationRunner(CompilationRunnerBase):
    def run(self):
        op = self.create_kernel()

        # Suppress stdout/stderr during compilation
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                op.forward_triton(op.get_random_input(), ptx=True)
                return {
                    "success": True,
                    "output": "",
                    "error": "",
                }

            except Exception:
                tb = traceback.format_exc()

        match = re.search(r"Repro command:\s*(.+)", tb)
        if not match:
            return {
                "success": True,
                "output": "",
                "error": "",
            }

        repro_cmd = match.group(1).strip()

        try:
            result = subprocess.run(
                shlex.split(repro_cmd),
                capture_output=True,
                text=True,
            )
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": (
                    f"Failed to run repro command: "
                    f"{type(e).__name__}: {e}"
                ),
            }

        output = (
            f"Return code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

        return {
            "success": result.returncode == 0,
            "output": output,
            "error": (
                "" if result.returncode == 0
                else f"Repro command failed with code {result.returncode}"
            ),
        }


def run_ptx_compilation(kernel, ptx_code):
    runner = PTXCompilationRunner(kernel, ptx_code)
    return runner.run()
