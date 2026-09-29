"""Compatibility shim for local imports from the repository root."""

from pathlib import Path


# Make the source package available as ``ptx_gym`` when running directly
# from the repository root, before resolving its lazy public exports.
__path__.append(str(Path(__file__).with_name("ptx_gym")))

from . import ptx_gym as _implementation

__all__ = _implementation.__all__


def __getattr__(name: str):
    value = getattr(_implementation, name)
    globals()[name] = value
    return value
