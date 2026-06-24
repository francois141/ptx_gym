import inspect
import random

import torch
from triton_ptx.evaluation.base import BaseVerifier
from triton_ptx.helpers.kernels import has_ptx_code
from triton_ptx.helpers.serialization import dump_nested, tensor_summary


class OutputVerifier(BaseVerifier):
    MAX_VALUES = 5 * 10**7

    def __init__(
        self,
        sizes=(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072),
        iters_per_size=250,
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
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
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

        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            return len(actual) == len(expected) and all(
                self._same(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )

        if isinstance(actual, dict) and isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(
                self._same(actual[key], expected[key])
                for key in actual
            )

        return actual == expected

    def _bad_mask(self, actual, expected):
        expected = expected.to(actual.dtype)

        ok = torch.isclose(
            actual,
            expected,
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=True,
        )

        return ~ok

    def _bad_indices(self, actual, expected, limit=None):
        bad = self._bad_mask(actual, expected).nonzero(as_tuple=False)
        if limit is None:
            return bad
        return bad[:limit]

    def _error_stats(self, actual, expected, mask=None):
        expected = expected.to(actual.dtype)
        actual64 = actual.detach().to(torch.float64)
        expected64 = expected.detach().to(torch.float64)
        abs_error = torch.abs(actual64 - expected64)
        rel_denom = torch.clamp(
            torch.abs(expected64),
            min=torch.finfo(torch.float64).eps,
        )
        rel_error = abs_error / rel_denom

        if mask is not None:
            abs_error = abs_error[mask]
            rel_error = rel_error[mask]
        if abs_error.numel() == 0:
            return {
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "max_relative_error": 0.0,
                "mean_relative_error": 0.0,
            }

        return {
            "max_abs_error": abs_error.max().detach().cpu().item(),
            "mean_abs_error": abs_error.mean().detach().cpu().item(),
            "max_relative_error": rel_error.max().detach().cpu().item(),
            "mean_relative_error": rel_error.mean().detach().cpu().item(),
        }

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
        report = {
            "status": "failed",
            "seed": self.seed,
            "size": size,
            "iteration": iteration,
            "kwargs": kwargs,
        }

        if not isinstance(expected_out, torch.Tensor) or not isinstance(triton_out, torch.Tensor):
            report[f"{expected_name}_output"] = dump_nested(expected_out)
            report["triton_output"] = dump_nested(triton_out)
            return report

        report[f"{expected_name}_output"] = tensor_summary(expected_out)
        report["triton_output"] = tensor_summary(triton_out)

        if expected_out.shape != triton_out.shape:
            report["error"] = {
                "type": "shape_mismatch",
                f"{expected_name}_shape": tuple(expected_out.shape),
                "triton_shape": tuple(triton_out.shape),
            }
            return report

        bad_mask = self._bad_mask(expected_out, triton_out)
        bad = self._bad_indices(expected_out, triton_out, limit=self.max_print)
        report["wrong_indices"] = bad.cpu()
        report["num_wrong"] = int(bad_mask.sum().detach().cpu().item())
        report["error_stats"] = {
            "all_values": self._error_stats(triton_out, expected_out),
            "wrong_values": self._error_stats(triton_out, expected_out, bad_mask),
        }
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
