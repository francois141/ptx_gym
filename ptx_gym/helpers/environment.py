from __future__ import annotations

import ctypes
from pathlib import Path
import re
import subprocess

import torch


_PTXAS_BIN_DIR = (
    Path(__file__).resolve().parents[2]
    / "triton_src"
    / "third_party"
    / "nvidia"
    / "backend"
    / "bin"
)
_PTXAS_PATH = _PTXAS_BIN_DIR / "ptxas"
_PTXAS_BLACKWELL_PATH = _PTXAS_BIN_DIR / "ptxas-blackwell"


def _cuda_release_to_ptx_version(cuda_version: str) -> str:
    """Map a CUDA toolkit release string like ``12.8`` to a PTX ISA version."""
    try:
        major, minor = map(int, cuda_version.split(".")[:2])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid CUDA version format: {cuda_version!r}"
        ) from exc

    ptx_mapping: dict[tuple[int, int], str] = {
        (11, 0): "7.0",
        (11, 1): "7.1",
        (11, 2): "7.2",
        (11, 3): "7.4",
        (11, 4): "7.4",
        (11, 5): "7.5",
        (11, 6): "7.6",
        (11, 7): "7.7",
        (11, 8): "7.8",
        (12, 0): "8.0",
        (12, 1): "8.1",
        (12, 2): "8.2",
        (12, 3): "8.3",
        (12, 4): "8.4",
        (12, 5): "8.5",
        (12, 6): "8.5",
        (12, 7): "8.6",
        (12, 8): "8.7",
        (13, 0): "9.0",
        (13, 1): "9.1",
        (13, 2): "9.2",
        (13, 3): "9.3",
    }

    key = (major, minor)
    if key not in ptx_mapping:
        supported = ", ".join(
            f"{maj}.{min_}" for maj, min_ in sorted(ptx_mapping)
        )
        raise ValueError(
            f"Unsupported CUDA version: {cuda_version}. "
            f"Supported versions: {supported}"
        )

    return ptx_mapping[key]


def is_gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is available, otherwise False."""
    return torch.cuda.is_available()


def get_available_video_memory_bytes(device_index: int | None = None) -> int:
    """
    Return currently available NVIDIA GPU memory in bytes.

    The value is queried with ``nvidia-smi`` in MiB and converted to bytes.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.insert(1, f"--id={device_index}")

    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to query available video memory with nvidia-smi.") from exc

    memory_mib = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"\d+", line)
        if match is not None:
            memory_mib.append(int(match.group(0)))

    if not memory_mib:
        raise RuntimeError(f"Unable to parse available video memory from nvidia-smi output: {output!r}")

    return min(memory_mib) * 1024 * 1024


def resolve_git_commit_hash() -> str:
    """Return the current Git commit hash, or ``"unknown"`` outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def get_ptxas_path(target: str | int | None) -> Path:
    """
    Return the bundled ``ptxas`` path for the requested target.

    Blackwell targets use the dedicated assembler when it is available.
    """
    if target is None:
        return _PTXAS_PATH

    if isinstance(target, int):
        target = f"sm_{target}"
    elif not str(target).startswith("sm_"):
        target = f"sm_{target}"

    capability_match = re.match(r"sm_(\d+)", target)
    assert capability_match is not None, f"Unexpected target: {target}"
    capability = int(capability_match.group(1))

    if capability >= 100 and _PTXAS_BLACKWELL_PATH.exists():
        return _PTXAS_BLACKWELL_PATH

    return _PTXAS_PATH


def get_ptx_system_config() -> tuple[str, str, int]:
    """
    Returns:
        version: PTX ISA version, e.g. "9.0"
        target: CUDA target architecture, e.g. "sm_89"
        address_size: pointer size in bits, usually 64
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    major, minor = torch.cuda.get_device_capability()
    target = f"sm_{major}{minor}"
    if target == "sm_90":
        # Temporary hack until a better target-version mapping is available.
        target = "sm_90a"

    if target == "sm_100":
        # Temporary hack until a better target-version mapping is available.
        target = "sm_100a"

    address_size = ctypes.sizeof(ctypes.c_void_p) * 8

    try:
        output = subprocess.check_output(
            [str(_PTXAS_PATH), "--version"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Unable to determine PTX version from {_PTXAS_PATH}."
        ) from exc

    match = re.search(r"release\s+(\d+\.\d+)", output)
    if match is None:
        raise RuntimeError(
            "Unable to parse CUDA release from ptxas --version output."
        )

    detected_cuda_version = match.group(1)
    version = _cuda_release_to_ptx_version(detected_cuda_version)

    return version, target, address_size
