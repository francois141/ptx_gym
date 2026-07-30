from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
            self.compiled_kernel_ptx = jit_fixed_parameters(
                self.kernel, ptx=get_ptx_code(ptx)
            )

    def ptx_launch_kwargs(self, **kwargs) -> dict[str, Any]:
        launch_kwargs = dict(kwargs)
        if not isinstance(getattr(self, "ptx", None), dict):
            return launch_kwargs
        launch_kwargs.update(
            {
                key: self.ptx[key]
                for key in ("num_threads_x", "num_threads_y", "num_threads_z")
                if self.ptx.get(key) is not None
            }
        )
        return launch_kwargs

    @staticmethod
    @abstractmethod
    def kernel(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_random_input(self):
        raise NotImplementedError

    @abstractmethod
    def get_shape_information(self) -> str:
        """Describe the dtype and shape of every pointer kernel argument."""
        raise NotImplementedError

    @abstractmethod
    def forward_triton(self, inputs, ptx: bool = False):
        raise NotImplementedError

    @abstractmethod
    def forward_torch(self, inputs):
        raise NotImplementedError
