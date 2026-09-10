from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any

import torch
from tqdm.auto import tqdm
from triton.runtime.errors import OutOfResources, PTXASError
from triton_ptx.helpers.kernels import get_ptx_code, has_ptx_code
from triton_ptx.helpers.triton import jit_fixed_parameters


@dataclass(frozen=True)
class KernelTuningResult:
    """The fastest verified launch configuration found for an operator."""

    parameters: dict[str, int]
    latency_ms: float
    evaluated_configurations: int
    rejected_configurations: int


class TritonPTXKernel(ABC):
    """Shared lifecycle helpers for Triton operators with optional PTX overrides."""

    ptx: Any
    compiled_kernel: Any
    compiled_kernel_ptx: Any | None

    def init_compiled_kernels(
        self,
        *,
        ptx: Any,
        autotune: bool = True,
        tuning_options: Mapping[str, tuple[int, ...]] | None = None,
    ) -> KernelTuningResult | None:
        """Compile kernels and optionally select the fastest verified config.

        ``tuning_options`` maps instance attributes (for example ``block_m`` or
        ``num_warps``) to values to search.  Omitting it uses conservative
        safe power-of-two divisors for every ``BLOCK_*`` constexpr and standard
        Triton warp/stage counts.  The selected configuration is stored in
        ``best_config`` and returned to the caller.
        """
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

        return self.autotune(tuning_options=tuning_options)

    def set_ptx(self, ptx: Any) -> None:
        """Replace the injected PTX launcher without retuning Triton."""
        self.ptx = ptx
        self.compiled_kernel_ptx = (
            jit_fixed_parameters(self.kernel, ptx=get_ptx_code(ptx))
            if has_ptx_code(ptx)
            else None
        )

    def autotune(
        self,
        *,
        tuning_options: Mapping[str, tuple[int, ...]] | None = None,
    ) -> KernelTuningResult:
        """Benchmark verified Triton configurations with fixed, tileable inputs."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to autotune a Triton kernel.")

        original_config = self._current_tuning_config()
        candidates = self._tuning_candidates(tuning_options)
        inputs = self.get_random_input(fixed=True)
        reference, _ = self.forward_triton(inputs)
        reference = reference.clone()
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
                    latency_ms = self._benchmark_current_configuration(inputs)
                except (
                    AssertionError,
                    OutOfResources,
                    PTXASError,
                    RuntimeError,
                ) as exc:
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


    def _benchmark_current_configuration(self, inputs: Any) -> float:
        """Return average launch latency, benchmarking for at least 25 ms."""
        self.forward_triton(inputs)
        torch.cuda.synchronize()

        min_duration_s = 0.025
        iterations = 0

        start = perf_counter()
        while True:
            self.forward_triton(inputs)
            iterations += 1

            # Avoid synchronizing after every launch.
            if iterations % 10 == 0:
                torch.cuda.synchronize()
                elapsed = perf_counter() - start
                if elapsed >= min_duration_s:
                    break

        torch.cuda.synchronize()
        elapsed = perf_counter() - start

        return (elapsed / iterations) * 1000.0  # ms per launch



    def _tuning_candidates(
        self, tuning_options: Mapping[str, tuple[int, ...]] | None
    ) -> list[dict[str, int]]:
        options = (
            {name: tuple(values) for name, values in tuning_options.items()}
            if tuning_options is not None
            else self._default_tuning_options()
        )
        if not options:
            return [self._current_tuning_config()]

        names = tuple(options)
        candidates = []
        for values in product(*(options[name] for name in names)):
            config = self._current_tuning_config()
            config.update(dict(zip(names, values, strict=True)))
            candidates.append(config)
        return candidates

    def _default_tuning_options(self) -> dict[str, tuple[int, ...]]:
        options: dict[str, tuple[int, ...]] = {}
        for constexpr_name in getattr(self, "constexpr_values", {}):
            attribute = constexpr_name.lower()
            value = getattr(self, attribute, None)
            if constexpr_name.startswith("BLOCK_") and isinstance(value, int):
                options[attribute] = (32, 64, 128, 256)
        if isinstance(getattr(self, "num_warps", None), int):
            options["num_warps"] = (4, 8, 16)
        if isinstance(getattr(self, "num_stages", None), int):
            options["num_stages"] = (2, 3)
        return options

    def _current_tuning_config(self) -> dict[str, int]:
        config = {}
        for constexpr_name in getattr(self, "constexpr_values", {}):
            attribute = constexpr_name.lower()
            value = getattr(self, attribute, None)
            if constexpr_name.startswith("BLOCK_") and isinstance(value, int):
                config[attribute] = value
        for attribute in ("num_warps", "num_stages"):
            value = getattr(self, attribute, None)
            if isinstance(value, int):
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
