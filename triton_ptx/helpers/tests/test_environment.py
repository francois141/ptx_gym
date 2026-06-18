from pathlib import Path

import pytest

from triton_ptx.evaluation.sandbox import compilation
from triton_ptx.evaluation.sandbox.compilation import compile_ptx
from triton_ptx.helpers import environment
from triton_ptx.helpers.environment import (
    get_ptx_system_config,
    get_ptxas_path,
    is_gpu_available,
)


def test_guess_ptx_version_from_ptxas_output(monkeypatch):
    mock_output = """
ptxas: NVIDIA (R) Ptx optimizing assembler
Cuda compilation tools, release 12.8, V12.8.93
""".strip()

    monkeypatch.setattr(
        environment.subprocess,
        "check_output",
        lambda *args, **kwargs: mock_output.encode("utf-8"),
    )

    assert environment._guess_ptx_version_from_ptxas() == "8.7"

def test_get_ptxas_path_prefers_blackwell_binary(monkeypatch, tmp_path):
    blackwell_path = tmp_path / "ptxas-blackwell"
    blackwell_path.touch()
    monkeypatch.setattr(environment, "_PTXAS_BLACKWELL_PATH", blackwell_path)

    assert get_ptxas_path("sm_100") == blackwell_path

def test_compile_ptx(monkeypatch):
    calls = {}

    class Completed:
        returncode = 0
        stdout = "compiled"
        stderr = ""

    def fake_run(command, capture_output, text):
        calls["command"] = command
        return Completed()

    monkeypatch.setattr(compilation, "get_ptx_system_config", lambda: ("8.7", "sm_90", 64))
    monkeypatch.setattr(compilation, "get_ptxas_path", lambda target=None: Path("/fake/ptxas"))
    monkeypatch.setattr(compilation.subprocess, "run", fake_run)

    result = compile_ptx(".version 8.0\n")

    assert result["success"] is True
    assert result["sm"] == "sm_90"
    assert calls["command"][1] == "-arch=sm_90"


@pytest.mark.skipif(
    not is_gpu_available(),
    reason="CUDA is not available on this system.",
)
def test_get_ptx_system_config_returns_non_empty_values():
    version, target, address_size = get_ptx_system_config()

    assert version
    assert target
    assert address_size
