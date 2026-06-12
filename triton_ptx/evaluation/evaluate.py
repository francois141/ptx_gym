from __future__ import annotations

import math
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from triton_ptx.helpers.triton import clear_triton_cache
from triton_ptx.evaluation.sandbox import (
    OutputVerifier,
    PTXBenchmarkRunner,
    run_ptx_compilation,
)


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


def benchmark_operator(operator):
    """
    Backward-compatible single-operator benchmarking entrypoint.

    Returns the same metrics mapping as PTXBenchmarkRunner.evaluate(), including
    a "ptx" Timing object used by triton_ptx.evaluation.
    """
    runner = PTXBenchmarkRunner()
    inputs = operator.get_random_input()
    return runner.evaluate(operator, inputs)


@dataclass(frozen=True)
class PromptCandidate:
    code: str
    compiles: bool
    message: str
    p20: float
    p50: float
    p80: float


@dataclass(order=True)
class EvaluatedCandidate:
    sort_index: tuple[int, float] = field(init=False, repr=False)

    kernel_name: str = field(compare=False)
    git_commit_hash: str = field(compare=False)

    round_index: int = field(compare=False)
    index: int = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    compiles: bool = field(compare=False)
    correct: bool = field(compare=False)
    message: str = field(compare=False)
    triton_p20: float = field(compare=False)
    triton_p50: float = field(compare=False)
    triton_p80: float = field(compare=False)
    p20: float = field(compare=False)
    p50: float = field(compare=False)
    p80: float = field(compare=False)
    speedup_vs_triton: float = field(compare=False)
    compile_output: str = field(default="", compare=False)
    compile_error: str = field(default="", compare=False)
    timing_error: str = field(default="", compare=False)
    verifier_report: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        self.sort_index = self._sort_key()

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
            "round_index": self.round_index,
            "index": self.index,
            "compiles": self.compiles,
            "correct": self.correct,
            "triton_p20": self.triton_p20,
            "triton_p50": self.triton_p50,
            "triton_p80": self.triton_p80,
            "p20": self.p20,
            "p50": self.p50,
            "p80": self.p80,
            "execution_time": self.p50,
            "runtime": self.p50,
            "speedup_vs_triton": self.speedup_vs_triton,
        }

    def prompt_candidate(self) -> PromptCandidate:
        return PromptCandidate(
            code=str(self.payload.get("ptx", "")),
            compiles=self.compiles,
            message=self.message,
            p20=self.p20,
            p50=self.p50,
            p80=self.p80,
        )

    def artifact_summary(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("sort_index", None)
        data["passed"] = self.passed
        # Keep metadata first in serialized artifacts for easier indexing.
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


class BaseCandidateEvaluator(ABC):
    """
    Abstract base class for candidate evaluators.
    """

    def __init__(
        self,
        operator_cls: type,
        *,
        clear_cache: bool = False,
    ) -> None:
        self.operator_cls = operator_cls
        self.clear_cache = clear_cache
        self.kernel_name = operator_cls.__name__
        self.git_commit_hash = self._resolve_git_commit_hash()

    @staticmethod
    def _resolve_git_commit_hash() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "unknown"

    @abstractmethod
    def evaluate(
        self,
        payload: dict[str, Any],
        *,
        round_index: int,
        candidate_index: int,
    ) -> EvaluatedCandidate:
        """
        Evaluate one candidate payload.
        """
        raise NotImplementedError


class TritonPTXCandidateEvaluator(BaseCandidateEvaluator):
    """
    Evaluates PTX candidates by compiling, checking correctness, and benchmarking.
    """

    def evaluate(
        self,
        payload: dict[str, Any],
        *,
        round_index: int,
        candidate_index: int,
    ) -> EvaluatedCandidate:
        if self.clear_cache:
            clear_triton_cache()

        compile_result = run_ptx_compilation(self.operator_cls, ptx_code=payload)
        compile_output = str(compile_result.get("output", "")).strip()
        compile_error = str(compile_result.get("error", "")).strip()
        compiles = bool(compile_result.get("success", False))

        if not compiles:
            return self._failed(
                payload,
                round_index=round_index,
                candidate_index=candidate_index,
                compiles=False,
                correct=False,
                message=compile_error or "Compilation failed",
                compile_output=compile_output,
                compile_error=compile_error,
            )

        try:
            operator = self.operator_cls(ptx=payload)
            verifier = OutputVerifier()
            correct = verifier.verify(operator)
            verifier_report = getattr(verifier, "last_report", {})
            
        except Exception as exc:
            return self._failed(
                payload,
                round_index=round_index,
                candidate_index=candidate_index,
                compiles=True,
                correct=False,
                message=f"Correctness check crashed: {type(exc).__name__}: {exc}",
                compile_output=compile_output,
                compile_error=compile_error,
                verifier_report={},
            )

        if not correct:
            return self._failed(
                payload,
                round_index=round_index,
                candidate_index=candidate_index,
                compiles=True,
                correct=False,
                message="Correctness check failed",
                compile_output=compile_output,
                compile_error=compile_error,
                verifier_report=verifier_report,
            )

        try:
            metrics = benchmark_operator(operator)
            ptx_timing = metrics["ptx"]
            triton_timing = metrics["triton"]

            if ptx_timing is None:
                raise RuntimeError("PTX timing metrics were not produced.")

            return EvaluatedCandidate(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                round_index=round_index,
                index=candidate_index,
                payload=payload,
                compiles=True,
                correct=True,
                message="Compiled, passed correctness, benchmarked successfully",
                triton_p20=float(triton_timing.p20),
                triton_p50=float(triton_timing.p50),
                triton_p80=float(triton_timing.p80),
                p20=float(ptx_timing.p20),
                p50=float(ptx_timing.p50),
                p80=float(ptx_timing.p80),
                speedup_vs_triton=(
                    float(triton_timing.p50 / ptx_timing.p50)
                    if ptx_timing.p50 > 0
                    else None
                ),
                compile_output=compile_output,
                compile_error=compile_error,
                verifier_report=verifier_report,
            )

        except Exception as exc:
            return self._failed(
                payload,
                round_index=round_index,
                candidate_index=candidate_index,
                compiles=True,
                correct=False,
                message=f"Benchmark failed: {type(exc).__name__}: {exc}",
                compile_output=compile_output,
                compile_error=compile_error,
                timing_error=str(exc),
                verifier_report=verifier_report,
            )

    def _failed(
        self,
        payload: dict[str, Any],
        *,
        round_index: int,
        candidate_index: int,
        compiles: bool,
        correct: bool,
        message: str,
        compile_output: str = "",
        compile_error: str = "",
        timing_error: str = "",
        verifier_report: dict[str, Any] | None = None,
    ) -> EvaluatedCandidate:
        return EvaluatedCandidate(
            kernel_name=self.kernel_name,
            git_commit_hash=self.git_commit_hash,
            round_index=round_index,
            index=candidate_index,
            payload=payload,
            compiles=compiles,
            correct=correct,
            message=message,
            triton_p20=math.inf,
            triton_p50=math.inf,
            triton_p80=math.inf,
            p20=math.inf,
            p50=math.inf,
            p80=math.inf,
            speedup_vs_triton=0,
            compile_output=compile_output,
            compile_error=compile_error,
            timing_error=timing_error,
            verifier_report=verifier_report or {},
        )
