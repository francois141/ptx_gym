from pathlib import Path

import pytest

from triton_ptx.evaluation import compilation
from triton_ptx.evaluation.compilation import compile_ptx
from triton_ptx.evaluation.types import Payload
from triton_ptx.helpers import environment
from triton_ptx.helpers.environment import (
    get_available_video_memory_bytes,
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

# memory.total [MiB]
# 46068 MiB
def test_get_available_video_memory_bytes(monkeypatch):

    def fake_check_output(command, text, stderr):
        return """
        memory.total [MiB]
        46068 MiB
        """.strip()

    monkeypatch.setattr(environment.subprocess, "check_output", fake_check_output)

    assert get_available_video_memory_bytes() == 46068 * 1024 * 1024

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

    result = compile_ptx(Payload(ptx=".version 8.0\n", threads_x=128))

    assert result.compiles is True
    assert result.sm == "sm_90"
    assert result.flags == ["-arch=sm_90", "-v", "--warning-as-error", "-o"]
    assert calls["command"][1] == "-arch=sm_90"
    assert "-v" in calls["command"]
    assert "--warning-as-error" in calls["command"]


@pytest.mark.skipif(
    not is_gpu_available(),
    reason="CUDA is not available on this system.",
)
def test_get_ptx_system_config_returns_non_empty_values():
    version, target, address_size = get_ptx_system_config()

    assert version
    assert target
    assert address_size
