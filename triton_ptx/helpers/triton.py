"""Utilities for Triton compilation, inspection, and local cache management."""

from __future__ import annotations

import errno
import inspect
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import triton
import triton.language as tl

def clear_triton_cache() -> None:
    """Remove Triton's local compilation cache if it exists.

    The cache directory is first renamed when possible so concurrent readers are
    less likely to observe a partially deleted tree. Deletion is retried a few
    times to tolerate transient filesystem races.
    """
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return

    delete_target = cache
    renamed_cache = cache.with_name(f"{cache.name}.deleting.{uuid.uuid4().hex}")
    try:
        cache.replace(renamed_cache)
        delete_target = renamed_cache
    except FileNotFoundError:
        return
    except OSError:
        # If the rename races with another process, fall back to deleting the
        # live cache path directly.
        delete_target = cache

    for attempt in range(3):
        try:
            shutil.rmtree(delete_target)
            print(f"Cache cleared: {cache}")
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY, errno.EPERM) or attempt == 2:
                print(f"Warning: failed to fully clear Triton cache at {cache}: {exc}")
                return
            time.sleep(0.1 * (attempt + 1))


def extract_ptx(
    compiled_kernel: Any,
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
        print(f"ttir: {compiled_kernel.asm.get('ttir', None)[:2000]}")
        print("-" * 100 + "\n")

    if print_ttgir:
        print(f"ttgir: {compiled_kernel.asm.get('ttgir', None)[:1000]}")
        print("-" * 100 + "\n")

    if print_llir:
        print(f"llir: {compiled_kernel.asm.get('llir', None)[:1000]}")
        print("-" * 100 + "\n")

    return compiled_kernel.asm.get("ptx", None)


def dump_kernel_ptx(kernel: Any) -> str | None:
    """Compile a kernel with randomized inputs and return its generated PTX.

    Args:
        kernel: Kernel wrapper exposing ``get_random_input`` and
            ``forward_triton`` helpers.

    Returns:
        The PTX emitted by Triton for the compiled kernel, if available.
    """
    inputs = kernel.get_random_input()
    _, compiled_kernel = kernel.forward_triton(inputs)
    return extract_ptx(compiled_kernel)


def jit_fixed_parameters(
    fn: Callable[..., Any] | None = None,
    **triton_kwargs: Any,
) -> Callable[..., Any]:
    """Wrap ``triton.jit`` while preserving non-``tl.constexpr`` parameters.

    This decorator auto-populates ``do_not_specialize`` with every function
    parameter that is not annotated as ``tl.constexpr``. That keeps regular
    runtime arguments in the generated PTX signature while still allowing
    compile-time constants to be specialized away.

    Args:
        fn: Optional function passed when the decorator is used without
            parentheses.
        **triton_kwargs: Extra keyword arguments forwarded to ``triton.jit``.

    Returns:
        A Triton-jitted function or a decorator that produces one.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        params = inspect.signature(func).parameters
        annotations = inspect.get_annotations(func, eval_str=True)
        do_not_specialize_list = []

        for name, param in params.items():
            annotation = annotations.get(name, param.annotation)
            if annotation is not tl.constexpr:
                do_not_specialize_list.append(name)

        return triton.jit(
            fn=func,
            do_not_specialize=do_not_specialize_list,
            **triton_kwargs,
        )

    if fn is None:
        return decorator

    return decorator(fn)
