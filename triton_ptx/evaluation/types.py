from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final


@dataclass(frozen=True)
class Payload:
    """Compiled PTX payload and launch dimensions."""

    ptx: str
    threads_x: int
    threads_y: int | None = None
    threads_z: int | None = None

    def __post_init__(self) -> None:
        if not self.ptx.strip():
            raise ValueError('Candidate payload is missing non-empty "ptx" code.')

        for name in ("threads_x", "threads_y", "threads_z"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f'Candidate payload field "{name}" must be a positive integer.')

    @classmethod
    def from_input(cls, payload: Payload | dict[str, Any]) -> Payload:
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("Candidate payload must be a dictionary.")

        return cls(
            ptx=payload.get("ptx", ""),
            threads_x=payload.get("threads_x", payload.get("num_threads_x")),
            threads_y=payload.get("threads_y", payload.get("num_threads_y")),
            threads_z=payload.get("threads_z", payload.get("num_threads_z")),
        )

    def to_launch_dict(self) -> dict[str, Any]:
        """Return the payload in evaluator launch configuration format."""
        payload: dict[str, Any] = {
            "ptx": self.ptx,
            "num_threads_x": self.threads_x,
        }
        if self.threads_y is not None:
            payload["num_threads_y"] = self.threads_y
        if self.threads_z is not None:
            payload["num_threads_z"] = self.threads_z
        return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key if isinstance(key, (str, int, float, bool)) or key is None else str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return _json_safe(value.detach().cpu().tolist())

    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


@dataclass(frozen=True)
class Timing:
    p20: float
    p50: float
    p80: float
    p90: float
    p95: float
    p99: float


@dataclass(order=True)
class EvaluatedCandidate:
    sort_index: tuple[int, float] = field(init=False, repr=False)

    kernel_name: str = field(compare=False)
    git_commit_hash: str = field(compare=False)

    payload: dict[str, Any] = field(compare=False)
    compiles: bool = field(compare=False)
    correct: bool = field(compare=False)
    message: str = field(compare=False)
    triton_p20: float = field(compare=False)
    triton_p50: float = field(compare=False)
    triton_p80: float = field(compare=False)
    triton_p90: float = field(compare=False)
    triton_p95: float = field(compare=False)
    triton_p99: float = field(compare=False)
    p20: float = field(compare=False)
    p50: float = field(compare=False)
    p80: float = field(compare=False)
    p90: float = field(compare=False)
    p95: float = field(compare=False)
    p99: float = field(compare=False)
    speedup_vs_triton: float = field(compare=False)
    compile_output: str = field(default="", compare=False)
    compile_error: str = field(default="", compare=False)
    timing_error: str = field(default="", compare=False)
    verifier_report: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        self.sort_index = self._sort_key()

    @classmethod
    def failed(
        cls,
        *,
        kernel_name: str,
        git_commit_hash: str,
        payload: dict[str, Any],
        compiles: bool,
        correct: bool,
        message: str,
        compile_output: str = "",
        compile_error: str = "",
        timing_error: str = "",
        verifier_report: dict[str, Any] | None = None,
    ) -> EvaluatedCandidate:
        return cls(
            kernel_name=kernel_name,
            git_commit_hash=git_commit_hash,
            payload=payload,
            compiles=compiles,
            correct=correct,
            message=message,
            triton_p20=math.inf,
            triton_p50=math.inf,
            triton_p80=math.inf,
            triton_p90=math.inf,
            triton_p95=math.inf,
            triton_p99=math.inf,
            p20=math.inf,
            p50=math.inf,
            p80=math.inf,
            p90=math.inf,
            p95=math.inf,
            p99=math.inf,
            speedup_vs_triton=0,
            compile_output=compile_output,
            compile_error=compile_error,
            timing_error=timing_error,
            verifier_report=verifier_report or {},
        )

    def _sort_key(self) -> tuple[int, float]:
        message = (self.message or "").lower()
        compile_error = (self.compile_error or "").lower()

        if "syntax" in message or "syntax" in compile_error:
            return (2, math.inf)

        if not self.compiles:
            return (1, math.inf)

        return (0, self.p50)

    @property
    def passed(self) -> bool:
        return self.compiles and self.correct

    def selection_record(self) -> dict[str, Any]:
        return {
            "compiles": self.compiles,
            "correct": self.correct,
            "triton_p20": self.triton_p20,
            "triton_p50": self.triton_p50,
            "triton_p80": self.triton_p80,
            "triton_p90": self.triton_p90,
            "triton_p95": self.triton_p95,
            "triton_p99": self.triton_p99,
            "p20": self.p20,
            "p50": self.p50,
            "p80": self.p80,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "execution_time": self.p50,
            "runtime": self.p50,
            "speedup_vs_triton": self.speedup_vs_triton,
        }

    def artifact_summary(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("sort_index", None)
        data["passed"] = self.passed
        return {
            "kernel_name": data.pop("kernel_name", ""),
            "git_commit_hash": data.pop("git_commit_hash", ""),
            **data,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            _json_safe(self.to_dict()),
            indent=indent,
            ensure_ascii=False,
        )
