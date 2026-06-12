from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from triton_ptx.helpers.kernels import get_ptx_code, has_ptx_code
from triton_ptx.helpers.triton import jit_fixed_parameters


class TritonPTXKernel(ABC):
    """Shared lifecycle helpers for Triton operators with optional PTX overrides."""

    ptx: Any
    compiled_kernel: Any
    compiled_kernel_ptx: Any | None

    def init_compiled_kernels(self, *, ptx: Any) -> None:
        self.ptx = ptx
        self.compiled_kernel = jit_fixed_parameters(self.kernel)
        self.compiled_kernel_ptx = None
        if has_ptx_code(ptx):
            self.compiled_kernel_ptx = jit_fixed_parameters(self.kernel, ptx=get_ptx_code(ptx))

    def require_compiled_ptx(self):
        if not has_ptx_code(getattr(self, "ptx", None)):
            raise RuntimeError(f"{type(self).__name__} was called with ptx=True but has no PTX code.")
        if getattr(self, "compiled_kernel_ptx", None) is None:
            raise RuntimeError(f"{type(self).__name__} has PTX code but no compiled PTX kernel.")
        return self.compiled_kernel_ptx

    def ptx_launch_kwargs(self, **defaults: Any) -> dict[str, Any]:
        launch_kwargs = dict(defaults)
        if not isinstance(getattr(self, "ptx", None), dict):
            return launch_kwargs
        launch_kwargs.update({
            key: self.ptx[key]
            for key in ("num_warps", "num_ctas", "num_threads_x", "num_threads_y", "num_threads_z", "BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK_SIZE")
            if self.ptx.get(key) is not None
        })
        return launch_kwargs

    @staticmethod
    @abstractmethod
    def kernel(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _rand_1d(size, *, dtype=torch.float32):
        return torch.rand(size, device="cuda", dtype=dtype)

    @abstractmethod
    def get_random_input(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_triton(self, inputs, ptx: bool = False):
        raise NotImplementedError

    @abstractmethod
    def forward_torch(self, inputs):
        raise NotImplementedError
