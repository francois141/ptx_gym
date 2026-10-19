from __future__ import annotations

from pathlib import Path

import pytest

from ptx_gym.evaluation import compilation
from ptx_gym.evaluation.compilation import compile_ptx
from ptx_gym.evaluation.types import Payload
from ptx_gym.helpers import environment
from ptx_gym.helpers.environment import (
    get_available_video_memory_bytes,
    get_ptx_system_config,
    get_ptxas_path,
    is_gpu_available,
)


def test_guess_ptx_version_from_ptxas_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map the CUDA release reported by ptxas to the expected PTX version."""
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
def test_get_available_video_memory_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert the available-memory nvidia-smi response from MiB to bytes."""

    def fake_check_output(
        command: list[str],
        text: bool,
        stderr: int,
    ) -> str:
        """Return a representative nvidia-smi memory response."""
        return """
        memory.total [MiB]
        46068 MiB
        """.strip()

    monkeypatch.setattr(environment.subprocess, "check_output", fake_check_output)

    assert get_available_video_memory_bytes() == 46068 * 1024 * 1024


def test_get_ptxas_path_prefers_blackwell_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prefer the dedicated Blackwell assembler for Blackwell targets."""
    blackwell_path = tmp_path / "ptxas-blackwell"
    blackwell_path.touch()
    monkeypatch.setattr(environment, "_PTXAS_BLACKWELL_PATH", blackwell_path)

    assert get_ptxas_path("sm_100") == blackwell_path
    assert get_ptxas_path("sm_121a") == blackwell_path


def test_blackwell_system_config_uses_selected_assembler(monkeypatch, tmp_path):
    blackwell_path = tmp_path / "ptxas-blackwell"
    blackwell_path.touch()
    monkeypatch.setattr(environment, "_PTXAS_BLACKWELL_PATH", blackwell_path)
    monkeypatch.setattr(environment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(environment, "_get_ptx_target", lambda: "sm_121a")

    def version_output(command, **kwargs):
        assert command == [str(blackwell_path), "--version"]
        return b"Cuda compilation tools, release 12.9, V12.9.86"

    monkeypatch.setattr(environment.subprocess, "check_output", version_output)
    assert get_ptx_system_config() == ("8.8", "sm_121a", 64)


@pytest.mark.parametrize(
    "release,version", [("12.8", "8.7"), ("12.9", "8.8"), ("13.0", "9.0")]
)
def test_cuda_release_mapping(release, version):
    assert environment._cuda_release_to_ptx_version(release) == version


@pytest.mark.parametrize(
    "output", ["[N/A]", "Not Supported", "", "GPU 0: Not Supported"]
)
def test_memory_query_falls_back_to_cuda(monkeypatch, output):
    monkeypatch.setattr(environment.subprocess, "check_output", lambda *a, **k: output)
    calls = []

    def cuda_memory(device):
        calls.append(device)
        return 123456789, 987654321

    monkeypatch.setattr(environment.torch.cuda, "mem_get_info", cuda_memory)
    assert get_available_video_memory_bytes(2) == 123456789
    assert calls == [2]


def test_missing_nvidia_smi_falls_back_to_cuda(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(environment.subprocess, "check_output", missing)
    monkeypatch.setattr(
        environment.torch.cuda, "mem_get_info", lambda device: (1024, 2048)
    )
    assert get_available_video_memory_bytes() == 1024


def test_memory_query_reports_both_failures(monkeypatch):
    monkeypatch.setattr(environment.subprocess, "check_output", lambda *a, **k: "N/A")

    def unavailable(device):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(environment.torch.cuda, "mem_get_info", unavailable)
    with pytest.raises(RuntimeError, match="nvidia-smi.*CUDA unavailable"):
        get_available_video_memory_bytes()


def test_compile_ptx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the detected target and strict flags to the PTX assembler."""
    calls: dict[str, list[str]] = {}

    class Completed:
        returncode = 0
        stdout = "compiled"
        stderr = ""

    def fake_run(
        command: list[str],
        capture_output: bool,
        text: bool,
    ) -> Completed:
        """Record an assembler command and return a successful result."""
        calls["command"] = command
        return Completed()

    monkeypatch.setattr(
        compilation, "get_ptx_system_config", lambda: ("8.7", "sm_90", 64)
    )
    monkeypatch.setattr(
        compilation, "get_ptxas_path", lambda target=None: Path("/fake/ptxas")
    )
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
def test_get_ptx_system_config_returns_non_empty_values() -> None:
    """Return a complete PTX configuration when CUDA is available."""
    version, target, address_size = get_ptx_system_config()

    assert version
    assert target
    assert address_size
