from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch
import triton
from ptx_gym.helpers.kernels import get_ptx_code, has_ptx_code
from ptx_gym.helpers.triton import ensure_triton_allocator, jit_fixed_parameters
from tqdm.auto import tqdm


@dataclass(frozen=True)
class KernelTuningResult:
    """The fastest verified launch configuration found for an operator."""

    parameters: dict[str, int]
    latency_ms: float
    evaluated_configurations: int
    rejected_configurations: int


class TritonPTXKernel(ABC):
    """Shared lifecycle helpers for Triton operators with optional PTX overrides."""

    tuning_options: Mapping[str, tuple[int, ...]]
    ptx: Any
    compiled_kernel: Any
    compiled_kernel_ptx: Any | None

    def init_compiled_kernels(
        self,
        *,
        ptx: Any,
        autotune: bool = True,
    ) -> KernelTuningResult | None:
        """Compile kernels and optionally select the fastest verified config.

        Each kernel declares ``tuning_options``, mapping instance attributes
        (for example ``block_m`` or ``num_warps``) to values to search. The
        selected configuration is stored in ``best_config`` and returned to
        the caller.
        """
        ensure_triton_allocator()
        self.ptx = ptx
        has_fixed_tuning_config = False
        if isinstance(ptx, Mapping):
            tuning_config = ptx.get("tuning_config")
            if tuning_config is not None:
                if not isinstance(tuning_config, Mapping):
                    raise TypeError("PTX tuning_config must be a mapping.")
                self._apply_tuning_config(tuning_config)
                has_fixed_tuning_config = True
        self.compiled_kernel = jit_fixed_parameters(self.kernel)
        self.compiled_kernel_ptx = None
        if has_ptx_code(ptx):
            self.compiled_kernel_ptx = jit_fixed_parameters(
                self.kernel, ptx=get_ptx_code(ptx)
            )
        if (
            not autotune
            or has_fixed_tuning_config
            or has_ptx_code(ptx)
            or not torch.cuda.is_available()
        ):
            self.best_config = self._current_tuning_config()
            self.tuning_result = None
            return None

        return self.autotune()

    def set_ptx(self, ptx: Any) -> None:
        """Replace the injected PTX launcher without retuning Triton."""
        self.ptx = ptx
        self.compiled_kernel_ptx = (
            jit_fixed_parameters(self.kernel, ptx=get_ptx_code(ptx))
            if has_ptx_code(ptx)
            else None
        )

    def autotune(self) -> KernelTuningResult:
        """Benchmark verified Triton configurations with fixed, tileable inputs."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to autotune a Triton kernel.")

        original_config = self._current_tuning_config()
        candidates = self._tuning_candidates()
        inputs = self.get_random_input(fixed=True)
        reference, _ = self.forward_triton(inputs)
        reference = self._clone_output(reference)
        best_config: dict[str, int] | None = None
        best_latency_ms = float("inf")
        rejected_configurations = 0

        try:
            for config in tqdm(
                candidates,
                desc=f"Tuning {type(self).__name__}",
                unit="configuration",
            ):
                self._apply_tuning_config(config)
                try:
                    output, _ = self.forward_triton(inputs)
                    tolerance = getattr(self, "autotune_tolerance", 1e-3)
                    torch.testing.assert_close(
                        output,
                        reference,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    latency_ms = triton.testing.do_bench(
                        lambda inputs=inputs: self.forward_triton(inputs),
                        warmup=100,
                        rep=200,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(exc)
                    rejected_configurations += 1
                    continue

                if latency_ms < best_latency_ms:
                    best_latency_ms = latency_ms
                    best_config = self._current_tuning_config()
        finally:
            self._apply_tuning_config(original_config)

        if best_config is None:
            raise RuntimeError(
                f"No valid tuning configuration found for {type(self).__name__}."
            )

        self._apply_tuning_config(best_config)
        result = KernelTuningResult(
            parameters=best_config,
            latency_ms=best_latency_ms,
            evaluated_configurations=len(candidates),
            rejected_configurations=rejected_configurations,
        )
        self.best_config = dict(best_config)
        self.tuning_result = result
        print(
            f"{type(self).__name__} best tuning config: {result.parameters} "
            f"({result.latency_ms:.4f} ms median; "
            f"{result.rejected_configurations}/{result.evaluated_configurations} "
            "rejected)"
        )
        return result

    def _tuning_candidates(self) -> list[dict[str, int]]:
        try:
            options = {
                name: tuple(values) for name, values in self.tuning_options.items()
            }
        except AttributeError as exc:
            raise TypeError(
                f"{type(self).__name__} must define tuning_options."
            ) from exc
        if not options:
            return [self._current_tuning_config()]

        names = tuple(options)
        candidates = []
        for values in product(*(options[name] for name in names)):
            config = self._current_tuning_config()
            config.update(dict(zip(names, values, strict=True)))
            candidates.append(config)
        return candidates

    @staticmethod
    def _clone_output(output):
        if isinstance(output, torch.Tensor):
            return output.clone()
        if isinstance(output, tuple):
            return tuple(TritonPTXKernel._clone_output(item) for item in output)
        if isinstance(output, list):
            return [TritonPTXKernel._clone_output(item) for item in output]
        raise TypeError(
            "Triton kernel outputs must be tensors or nested tuples/lists of tensors."
        )

    def _current_tuning_config(self) -> dict[str, int]:
        config = {
            attribute: value
            for attribute in self.tuning_options
            if isinstance(value := getattr(self, attribute, None), int)
        }
        for attribute in ("num_warps", "num_stages", "maxnreg"):
            value = getattr(self, attribute, None)
            if isinstance(value, int) or (
                attribute == "maxnreg" and hasattr(self, attribute)
            ):
                config[attribute] = value
        return config

    def _apply_tuning_config(self, config: Mapping[str, int]) -> None:
        for attribute, value in config.items():
            setattr(self, attribute, value)
        constexpr_values = getattr(self, "constexpr_values", None)
        if isinstance(constexpr_values, dict):
            for name in constexpr_values:
                attribute = name.lower()
                if attribute in config:
                    constexpr_values[name] = config[attribute]

    def ptx_launch_kwargs(self, **kwargs) -> dict[str, Any]:
        launch_kwargs = dict(kwargs)
        ptx = getattr(self, "ptx", None)
        if not isinstance(ptx, dict):
            launch_kwargs.setdefault("num_warps", self.num_warps)
            return launch_kwargs
        launch_kwargs.update(
            {
                key: ptx[key]
                for key in ("num_threads_x", "num_threads_y", "num_threads_z")
                if ptx.get(key) is not None
            }
        )
        thread_dimensions = tuple(
            launch_kwargs.get(key, 1)
            for key in ("num_threads_x", "num_threads_y", "num_threads_z")
        )
        thread_count = (
            thread_dimensions[0] * thread_dimensions[1] * thread_dimensions[2]
        )
        if thread_count % 32 == 0:
            launch_kwargs.setdefault("num_warps", thread_count // 32)
        else:
            launch_kwargs.setdefault("num_warps", self.num_warps)
        return launch_kwargs

    def volta_arguments(self) -> tuple[Path, list[str]]:
        """Return the Volta spec path and ``volta verify`` launch arguments.

        The arguments describe the grid, arrays, parameters, and dimensions;
        the evaluator supplies the block size and the sample size.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support Volta verification."
        )

    @staticmethod
    @abstractmethod
    def kernel(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_random_input(self, fixed: bool = False):
        """Create random inputs, optionally using a fixed tile-compatible shape."""
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
