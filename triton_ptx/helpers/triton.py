"""Utilities for Triton compilation, inspection, and local cache management."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
import os

import triton
import triton.language as tl
import torch
from triton.compiler.compiler import CompiledKernel


def clear_triton_cache() -> None:
    """Remove Triton's local compilation cache if it exists."""
    cache = Path(
        os.environ.get(
            "TRITON_CACHE_DIR",
            Path.home() / ".triton" / "cache",
        )
    )

    if not cache.exists():
        return

    try:
        shutil.rmtree(cache)
    except OSError as exc:
        print(f"Warning: failed to clear Triton cache at {cache}: {exc}")


def extract_ptx(
    compiled_kernel: CompiledKernel,
    *,
    print_ttir: bool = False,
    print_ttgir: bool = False,
    print_llir: bool = False,
) -> str | None:
    """Return PTX from a compiled Triton kernel and optionally print IR stages.

    Args:
        compiled_kernel: Triton's compiled kernel object with an ``asm`` mapping.
        print_ttir: Whether to print the TTIR snippet.
        print_ttgir: Whether to print the TTGIR snippet.
        print_llir: Whether to print the LLVM IR snippet.

    Returns:
        The compiled PTX source when present, otherwise ``None``.
    """
    if print_ttir:
        ttir = compiled_kernel.asm.get("ttir")
        print(f"ttir: {ttir[:2000] if ttir is not None else None}")
        print("-" * 100 + "\n")

    if print_ttgir:
        ttgir = compiled_kernel.asm.get("ttgir")
        print(f"ttgir: {ttgir[:1000] if ttgir is not None else None}")
        print("-" * 100 + "\n")

    if print_llir:
        llir = compiled_kernel.asm.get("llir")
        print(f"llir: {llir[:1000] if llir is not None else None}")
        print("-" * 100 + "\n")

    ptx = compiled_kernel.asm.get("ptx")
    return ptx if isinstance(ptx, str) else None


def dump_kernel_ptx(kernel, inputs=None):
    """Compile a kernel with randomized inputs and return its generated PTX.

    Args:
        kernel: Kernel wrapper exposing ``get_random_input`` and
            ``forward_triton`` helpers.

    Returns:
        The PTX emitted by Triton for the compiled kernel, if available.
    """
    inputs = kernel.get_random_input() if inputs is None else inputs
    with torch.no_grad():
        _, compiled_kernel = kernel.forward_triton(inputs)
    return extract_ptx(compiled_kernel)


def jit_fixed_parameters(fn=None, **triton_kwargs):
    """Wrap ``triton.jit`` while preserving non-``tl.constexpr`` parameters.

    This decorator prevents specialization of scalar runtime parameters while
    preserving pointer specialization. Pointer specialization retains alignment
    facts that enable vectorized memory operations and asynchronous copies;
    pointers remain runtime PTX parameters.

    Args:
        fn: Optional function passed when the decorator is used without
            parentheses.
        **triton_kwargs: Extra keyword arguments forwarded to ``triton.jit``.

    Returns:
        A Triton-jitted function or a decorator that produces one.
    """

    def decorator(func):
        """JIT-compile one function with inferred specialization settings."""
        params = inspect.signature(func).parameters
        annotations = inspect.get_annotations(func, eval_str=True)
        do_not_specialize_list = []

        for name, param in params.items():
            annotation = annotations.get(name, param.annotation)
            if annotation is not tl.constexpr and not name.endswith("_ptr"):
                do_not_specialize_list.append(name)

        return triton.jit(
            fn=func,
            do_not_specialize=do_not_specialize_list,
            **triton_kwargs,
        )

    if fn is None:
        return decorator

    return decorator(fn)
