from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import triton

from triton_ptx.helpers import get_ptx_code, has_ptx_code


class TritonPTXKernel(ABC):
    """Shared lifecycle helpers for Triton operators with optional PTX overrides."""

    ptx: Any
    compiled_kernel: Any
    compiled_kernel_ptx: Any | None

    def init_compiled_kernels(self, *, ptx: Any, jit=triton.jit) -> None:
        self.ptx = ptx
        self.compiled_kernel = jit(self.kernel)
        self.compiled_kernel_ptx = None
        if has_ptx_code(ptx):
            self.compiled_kernel_ptx = jit(self.kernel, ptx=get_ptx_code(ptx))

    def require_compiled_ptx(self):
        if not has_ptx_code(getattr(self, "ptx", None)):
            raise RuntimeError(f"{type(self).__name__} was called with ptx=True but has no PTX code.")
        if getattr(self, "compiled_kernel_ptx", None) is None:
            raise RuntimeError(f"{type(self).__name__} has PTX code but no compiled PTX kernel.")
        return self.compiled_kernel_ptx

    @staticmethod
    @abstractmethod
    def kernel(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_random_input(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_triton(self, inputs, ptx: bool = False):
        raise NotImplementedError

    @abstractmethod
    def forward_torch(self, inputs):
        raise NotImplementedError
