from abc import ABC, abstractmethod
import inspect
import torch
from triton_ptx.helpers import has_ptx_code

class BaseVerifier(ABC):
    """Abstract parent class for verifiers."""

    @abstractmethod
    def verify(self, op) -> bool:
        """Run verification logic."""
        pass


class OutputVerifier(BaseVerifier):
    """Verifies Triton/PTX outputs against Torch output."""

    MAX_RANDOM_INPUT_VALUES = 2 * 10**7

    def __init__(
        self,
        num_samples: int = 20,
        growth_factor: float = 1.5,
        runs_per_phase: int = 40,
    ):
        self.num_samples = max(1, int(num_samples))
        self.growth_factor = max(1.0, float(growth_factor))
        self.runs_per_phase = max(1, int(runs_per_phase))
        self.last_report = {}

    def check_similarity(self, actual, expected, rtol=1e-2, atol=1e-2) -> bool:
        return actual.shape == expected.shape and torch.allclose(
            actual,
            expected.to(actual.dtype),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )

    def _should_scale_arg(self, name: str) -> bool:
        name = name.lower()
        if any(token in name for token in ("kernel", "stride", "block", "tile")):
            return False
        return any(token in name for token in ("size", "m", "n", "k", "h", "w", "len"))

    def _count_tensor_values(self, value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.numel())
        if isinstance(value, (list, tuple)):
            return sum(self._count_tensor_values(v) for v in value)
        if isinstance(value, dict):
            return sum(self._count_tensor_values(v) for v in value.values())
        return 0

    def _phase_for_sample(self, sample_index: int, scalable_args):
        if not scalable_args:
            return None, 0, 0, 0

        phase_index = sample_index // self.runs_per_phase
        active_dimension_index = phase_index % len(scalable_args)
        phase_round = phase_index // len(scalable_args)
        return (
            scalable_args[active_dimension_index],
            active_dimension_index,
            phase_index,
            phase_round,
        )

    def _scaled_random_input(self, op, sample_index: int):
        signature = inspect.signature(op.get_random_input)
        kwargs = {}
        scalable_args = []

        for name, param in signature.parameters.items():
            if param.default is inspect._empty:
                continue
            default = 4
            if isinstance(default, int) and self._should_scale_arg(name):
                kwargs[name] = default
                scalable_args.append(name)
            else:
                kwargs[name] = default

        active_dimension, active_dimension_index, phase_index, phase_round = (
            self._phase_for_sample(sample_index, scalable_args)
        )
        dimension_scale_steps = {}
        for dimension_index, name in enumerate(scalable_args):
            scale_step = phase_round + int(dimension_index <= active_dimension_index)
            dimension_scale_steps[name] = scale_step
            kwargs[name] = max(1, int(4 * (self.growth_factor ** scale_step)))

        while True:
            inputs = op.get_random_input(**kwargs)
            num_values = self._count_tensor_values(inputs)
            if num_values <= self.MAX_RANDOM_INPUT_VALUES:
                kwargs["max_allowed_values"] = self.MAX_RANDOM_INPUT_VALUES
                kwargs["num_values"] = num_values
                kwargs["runs_per_phase"] = self.runs_per_phase
                kwargs["phase_index"] = phase_index
                kwargs["active_dimension"] = active_dimension
                kwargs["active_dimension_index"] = active_dimension_index
                kwargs["phase_round"] = phase_round
                kwargs["dimension_scale_steps"] = dimension_scale_steps
                return inputs, kwargs

            if not scalable_args:
                raise ValueError(
                    f"Random input contains {num_values} values, above the "
                    f"max allowed threshold of {self.MAX_RANDOM_INPUT_VALUES}, "
                    "and no scalable get_random_input arguments were found."
                )

            shrink_factor = (self.MAX_RANDOM_INPUT_VALUES / num_values) ** (
                1 / len(scalable_args)
            )
            previous_kwargs = kwargs.copy()
            for name in scalable_args:
                kwargs[name] = max(1, int(kwargs[name] * shrink_factor))

            if all(kwargs[name] == previous_kwargs[name] for name in scalable_args):
                if all(kwargs[name] == 1 for name in scalable_args):
                    raise ValueError(
                        f"Random input contains {num_values} values, above the "
                        f"max allowed threshold of {self.MAX_RANDOM_INPUT_VALUES}, "
                        "even with all scalable get_random_input arguments set to 1."
                    )
                largest_arg = max(scalable_args, key=lambda name: kwargs[name])
                kwargs[largest_arg] = max(1, kwargs[largest_arg] - 1)

    def _serialize_value(self, value):
        if isinstance(value, torch.Tensor):
            info = {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": int(value.numel()),
            }
            if value.numel() > 0:
                cpu = value.detach().flatten().to("cpu")
                finite = cpu[torch.isfinite(cpu)]
                if finite.numel() > 0:
                    info["min"] = float(finite.min().item())
                    info["max"] = float(finite.max().item())
                    info["mean"] = float(finite.mean().item())
                preview = cpu[: min(64, cpu.numel())].tolist()
                info["preview"] = preview
            return info

        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        return repr(value)

    def verify(self, op) -> bool:
        assert has_ptx_code(getattr(op, "ptx", None))

        last_success = None
        for sample_index in range(self.num_samples):
            inputs, scaled_kwargs = self._scaled_random_input(op, sample_index)
            triton_out, _ = op.forward_triton(inputs)
            ptx_out, _ = op.forward_triton(inputs, ptx=True)

            sample = {
                "sample_index": sample_index,
                "scaled_get_random_input_kwargs": scaled_kwargs,
                "inputs": self._serialize_value(inputs),
                "triton_output": self._serialize_value(triton_out),
                "ptx_output": self._serialize_value(ptx_out),
            }

            if not self.check_similarity(ptx_out, triton_out):
                self.last_report = {
                    "status": "failed",
                    "num_samples": self.num_samples,
                    "growth_factor": self.growth_factor,
                    "runs_per_phase": self.runs_per_phase,
                    "last_success": last_success,
                    "failure": sample,
                }
                return False

            last_success = sample

        self.last_report = {
            "status": "passed",
            "num_samples": self.num_samples,
            "growth_factor": self.growth_factor,
            "runs_per_phase": self.runs_per_phase,
            "last_success": last_success,
            "failure": None,
        }
        return True
