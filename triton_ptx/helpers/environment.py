from __future__ import annotations

import ctypes
import torch


def is_gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is available, otherwise False."""
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


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

    version = _guess_ptx_version_from_cuda(torch.version.cuda)

    return version, target, address_size


def _guess_ptx_version_from_cuda(cuda_version: str | None) -> str:
    """
    Map CUDA toolkit versions to PTX ISA versions.

    Raises:
        RuntimeError: If CUDA version is unavailable.
        ValueError: If CUDA version is not explicitly supported.
    """
    if cuda_version is None:
        raise RuntimeError(
            "Unable to determine CUDA version from torch.version.cuda."
        )

    try:
        major, minor = map(int, cuda_version.split(".")[:2])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid CUDA version format: {cuda_version!r}"
        ) from exc

    # CUDA Toolkit -> PTX ISA mapping
    ptx_mapping: dict[tuple[int, int], str] = {
        # CUDA 11.x → PTX 7.x
        (11, 0): "7.0",
        (11, 1): "7.1",
        (11, 2): "7.2",
        (11, 3): "7.4",
        (11, 4): "7.4",
        (11, 5): "7.5",
        (11, 6): "7.6",
        (11, 7): "7.7",
        (11, 8): "7.8",

        # CUDA 12.x → PTX 8.x
        (12, 0): "8.0",
        (12, 1): "8.1",
        (12, 2): "8.2",
        (12, 3): "8.3",
        (12, 4): "8.4",
        (12, 5): "8.5",
        (12, 6): "8.5",
        (12, 7): "8.6",
        (12, 8): "8.7",

        # CUDA 13.x → PTX 9.x
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


if __name__ == "__main__":
    version, target, address_size = get_ptx_system_config()

    print(version)
    print(target)
    print(address_size)