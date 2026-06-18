from __future__ import annotations

import ctypes
from pathlib import Path
import re
import subprocess

import torch


_PTXAS_BIN_DIR = (
    Path(__file__).resolve().parents[2]
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


def get_ptxas_path(target: str) -> Path:
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

    address_size = ctypes.sizeof(ctypes.c_void_p) * 8

    version = _guess_ptx_version_from_ptxas()

    return version, target, address_size


def _guess_ptx_version_from_ptxas() -> str:
    """
    Run the bundled ``ptxas --version`` command and map its CUDA release
    to a PTX ISA version.

    Raises:
        RuntimeError: If ``ptxas`` cannot be executed or its version cannot be parsed.
        ValueError: If the detected CUDA release is not explicitly supported.
    """
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
    return _cuda_release_to_ptx_version(detected_cuda_version)
