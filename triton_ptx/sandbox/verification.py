from abc import ABC, abstractmethod
import inspect
import random

import torch
from triton_ptx.helpers import has_ptx_code
from triton_ptx.kernels import test_kernel


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, op) -> bool:
        pass


class OutputVerifier(BaseVerifier):
    MAX_VALUES = 2 * 10**7

    def __init__(
        self,
        sizes=(4, 8, 16, 32, 64, 128),
        iters_per_size=100,
        seed=42,
        rtol=1e-2,
        atol=1e-2,
        max_print=32,
    ):
        self.sizes = sizes
        self.iters_per_size = iters_per_size
        self.seed = seed
        self.rtol = rtol
        self.atol = atol
        self.max_print = max_print
        self.last_report = {}

    def _is_size_arg(self, name):
        name = name.lower()
        return (
            any(x in name for x in ("size", "m", "n", "k", "h", "w", "len"))
            and not any(x in name for x in ("kernel", "stride", "block", "tile"))
        )

    def _kwargs(self, op, size):
        return {
            name: size if self._is_size_arg(name) else 4
            for name, param in inspect.signature(op.get_random_input).parameters.items()
            if param.default is not inspect._empty
        }

    def _numel(self, x):
        if isinstance(x, torch.Tensor):
            return x.numel()
        if isinstance(x, (list, tuple)):
            return sum(self._numel(v) for v in x)
        if isinstance(x, dict):
            return sum(self._numel(v) for v in x.values())
        return 0

    def _same(self, actual, expected):
        return (
            actual.shape == expected.shape
            and torch.allclose(
                actual,
                expected.to(actual.dtype),
                rtol=self.rtol,
                atol=self.atol,
                equal_nan=True,
            )
        )

    def _bad_indices(self, actual, expected):
        expected = expected.to(actual.dtype)

        ok = torch.isclose(
            actual,
            expected,
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=True,
        )

        return (~ok).nonzero(as_tuple=False)[: self.max_print]

    def _dump(self, x):
        if isinstance(x, torch.Tensor):
            return {
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "values": x.detach().cpu(),
            }

        if isinstance(x, (list, tuple)):
            return [self._dump(v) for v in x]

        if isinstance(x, dict):
            return {k: self._dump(v) for k, v in x.items()}

        return x

    def _failure_report(
        self,
        size,
        iteration,
        kwargs,
        inputs,
        triton_out,
        expected_out,
        expected_name="ptx",
    ):
        expected_output_key = f"{expected_name}_output"
        report = {
            "status": "failed",
            "seed": self.seed,
            "size": size,
            "iteration": iteration,
            "kwargs": kwargs,
            "inputs": self._dump(inputs),
            "triton_output": self._dump(triton_out),
            expected_output_key: self._dump(expected_out),
        }

        if not isinstance(expected_out, torch.Tensor) or not isinstance(triton_out, torch.Tensor):
            return report

        if expected_out.shape != triton_out.shape:
            report["error"] = {
                "type": "shape_mismatch",
                f"{expected_name}_shape": tuple(expected_out.shape),
                "triton_shape": tuple(triton_out.shape),
            }
            return report

        bad = self._bad_indices(expected_out, triton_out)
        report["wrong_indices"] = bad.cpu()
        report["wrong_values"] = [
            {
                "index": tuple(idx.tolist()),
                expected_name: expected_out[tuple(idx)].detach().cpu().item(),
                "triton": triton_out[tuple(idx)].detach().cpu().item(),
            }
            for idx in bad
        ]

        return report

    def verify(self, op) -> bool:
        assert has_ptx_code(getattr(op, "ptx", None))

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        for size in self.sizes:
            for iteration in range(self.iters_per_size):
                kwargs = self._kwargs(op, size)
                inputs = op.get_random_input(**kwargs)

                num_values = self._numel(inputs)
                if num_values > self.MAX_VALUES:
                    raise ValueError(
                        f"size={size} generated {num_values} tensor values, "
                        f"above limit {self.MAX_VALUES}"
                    )

                triton_out, _ = op.forward_triton(inputs)
                ptx_out, _ = op.forward_triton(inputs, ptx=True)

                if not self._same(ptx_out, triton_out):
                    self.last_report = self._failure_report(
                        size,
                        iteration,
                        kwargs,
                        inputs,
                        triton_out,
                        ptx_out,
                    )
                    return False

                self.last_report = {
                    "status": "passed",
                    "seed": self.seed,
                    "size": size,
                    "iteration": iteration,
                    "kwargs": kwargs,
                    "num_values": int(num_values),
                }

        return True

    def verify_triton_vs_torch(self, op) -> bool:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        for size in self.sizes:
            for iteration in range(self.iters_per_size):
                kwargs = self._kwargs(op, size)
                inputs = op.get_random_input(**kwargs)

                num_values = self._numel(inputs)
                if num_values > self.MAX_VALUES:
                    raise ValueError(
                        f"size={size} generated {num_values} tensor values, "
                        f"above limit {self.MAX_VALUES}"
                    )

                triton_out, _ = op.forward_triton(inputs)
                torch_out = op.forward_torch(inputs)

                if not self._same(triton_out, torch_out):
                    self.last_report = self._failure_report(
                        size,
                        iteration,
                        kwargs,
                        inputs,
                        triton_out,
                        torch_out,
                        expected_name="torch",
                    )
                    return False

                self.last_report = {
                    "status": "passed",
                    "seed": self.seed,
                    "size": size,
                    "iteration": iteration,
                    "kwargs": kwargs,
                    "num_values": int(num_values),
                }

        return True


def main() -> bool:
    verifier = OutputVerifier()
    passed = True

    kernel_cls = test_kernel
    op = kernel_cls()

    if not has_ptx_code(getattr(op, "ptx", None)):
        return passed

    ok = verifier.verify(op)
    print(f"[{'PASSED' if ok else 'FAILED'}] {kernel_cls.__name__}")

    if not ok:
        print(verifier.last_report)
        passed = False

    return passed


if __name__ == "__main__":
    main()
