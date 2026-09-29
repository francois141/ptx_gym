from __future__ import annotations

import subprocess

from ptx_gym.evaluation.base import BaseCandidateEvaluator
from ptx_gym.evaluation.compilation import compile_ptx
from ptx_gym.evaluation.ncu import profile_ptx_with_ncu
from ptx_gym.evaluation.performance import evaluate_ptx_performance
from ptx_gym.evaluation.sanitizer import diagnose_ptx
from ptx_gym.evaluation.types import EvaluatedCandidate, Payload
from ptx_gym.evaluation.verification import OutputVerifier
from ptx_gym.evaluation.volta import verify_ptx_with_volta, volta_binary
from ptx_gym.helpers.triton import clear_triton_cache


def benchmark_operator(operator):
    """
    Backward-compatible single-operator benchmarking entrypoint.

    Returns the same metrics mapping as evaluate_ptx_performance(), including
    a "ptx" Timing object used by ptx_gym.evaluation.
    """
    inputs = operator.get_random_input()
    return evaluate_ptx_performance(operator, inputs)


class TritonPTXCandidateEvaluator(BaseCandidateEvaluator):
    """
    Evaluates PTX candidates by compiling, checking correctness, and benchmarking.
    """

    def __init__(
        self,
        operator_cls,
        *,
        operator=None,
        enable_ncu_report=True,
        enable_sanitizer=True,
        enable_volta=False,
    ):
        super().__init__(operator_cls, operator=operator)
        self.enable_ncu_report = enable_ncu_report
        self.enable_sanitizer = enable_sanitizer
        self.enable_volta = enable_volta
        if enable_volta:
            volta_binary()
            self.operator.volta_arguments()

    def evaluate(
        self,
        payload: Payload,
    ) -> EvaluatedCandidate:
        clear_triton_cache()

        if not isinstance(payload, Payload):
            raise TypeError("payload must be a Payload instance.")

        launch_payload = payload.to_launch_dict()

        compile_result = compile_ptx(payload)
        compile_output = compile_result.output.strip()
        compile_error = compile_result.error.strip()
        compiles = compile_result.compiles

        if not compiles:
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                compiles=False,
                correct=False,
                message=compile_error or "Compilation failed",
                compile_output=compile_output,
                compile_error=compile_error,
            )
        sanitizer_report = {}
        if self.enable_sanitizer:
            try:
                sanitizer_report = diagnose_ptx(
                    self.kernel_name,
                    launch_payload,
                    sanitizer_tool="all",
                    tuning_config=self.operator.best_config,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                return EvaluatedCandidate.failed(
                    kernel_name=self.kernel_name,
                    git_commit_hash=self.git_commit_hash,
                    compiles=True,
                    correct=False,
                    message=(f"Sanitizer check crashed: {type(exc).__name__}: {exc}"),
                    compile_output=compile_output,
                    compile_error=compile_error,
                )

        if self.enable_sanitizer and sanitizer_report.get("clean") is not True:
            sanitizer_error = sanitizer_report.get("error")
            input_kwargs = sanitizer_report.get("input_kwargs")
            message = "Sanitizer check failed"
            if isinstance(input_kwargs, dict):
                dimensions = ", ".join(
                    f"{name}={value}" for name, value in sorted(input_kwargs.items())
                )
                message = f"{message} (input: {dimensions or 'default'})"
            if isinstance(sanitizer_error, str) and sanitizer_error:
                message = f"{message}: {sanitizer_error}"
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                compiles=True,
                correct=False,
                message=message,
                compile_output=compile_output,
                compile_error=compile_error,
                sanitizer_report=sanitizer_report,
            )

        volta_report = {}
        if self.enable_volta:
            try:
                volta_report = verify_ptx_with_volta(self.operator, launch_payload)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return EvaluatedCandidate.failed(
                    kernel_name=self.kernel_name,
                    git_commit_hash=self.git_commit_hash,
                    compiles=True,
                    correct=False,
                    message=f"Volta verification crashed: {type(exc).__name__}: {exc}",
                    compile_output=compile_output,
                    compile_error=compile_error,
                    sanitizer_report=sanitizer_report,
                )
            if not volta_report["equivalent"]:
                return EvaluatedCandidate.failed(
                    kernel_name=self.kernel_name,
                    git_commit_hash=self.git_commit_hash,
                    compiles=True,
                    correct=False,
                    message="Volta verification failed: PTX is not equivalent to "
                    "the kernel spec",
                    compile_output=compile_output,
                    compile_error=compile_error,
                    sanitizer_report=sanitizer_report,
                    volta_report=volta_report,
                )

        try:
            self.operator.set_ptx(launch_payload)
            operator = self.operator
            tolerance = getattr(operator, "verification_tolerance", 1e-3)
            verifier = OutputVerifier(rtol=tolerance, atol=tolerance)
            correct = verifier.verify(operator)
            verifier_report = getattr(verifier, "last_report", {})

        except Exception as exc:
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                compiles=True,
                correct=False,
                message=f"Correctness check crashed: {type(exc).__name__}: {exc}",
                compile_output=compile_output,
                compile_error=compile_error,
                sanitizer_report=sanitizer_report,
                volta_report=volta_report,
                verifier_report={},
            )

        if not correct:
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                compiles=True,
                correct=False,
                message="Correctness check failed",
                compile_output=compile_output,
                compile_error=compile_error,
                sanitizer_report=sanitizer_report,
                volta_report=volta_report,
                verifier_report=verifier_report,
            )

        try:
            metrics = benchmark_operator(operator)
            ptx_timing = metrics["ptx"]
            triton_timing = metrics["triton"]

            if ptx_timing is None:
                raise RuntimeError("PTX timing metrics were not produced.")

            ncu_report = {}
            if self.enable_ncu_report:
                ncu_report = profile_ptx_with_ncu(
                    self.kernel_name,
                    payload,
                    tuning_config=self.operator.best_config,
                ).to_dict()

            return EvaluatedCandidate(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                compiles=True,
                correct=True,
                message="Compiled, passed correctness, benchmarked successfully",
                triton_p20=float(triton_timing.p20),
                triton_p50=float(triton_timing.p50),
                triton_p80=float(triton_timing.p80),
                triton_p90=float(triton_timing.p90),
                triton_p95=float(triton_timing.p95),
                triton_p99=float(triton_timing.p99),
                p20=float(ptx_timing.p20),
                p50=float(ptx_timing.p50),
                p80=float(ptx_timing.p80),
                p90=float(ptx_timing.p90),
                p95=float(ptx_timing.p95),
                p99=float(ptx_timing.p99),
                speedup_vs_triton=(
                    float(triton_timing.p50 / ptx_timing.p50)
                    if ptx_timing.p50 > 0
                    else None
                ),
                compile_output=compile_output,
                compile_error=compile_error,
                ncu_report=ncu_report,
                sanitizer_report=sanitizer_report,
                volta_report=volta_report,
                verifier_report=verifier_report,
            )

        except Exception as exc:
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                compiles=True,
                correct=False,
                message=f"Benchmark failed: {type(exc).__name__}: {exc}",
                compile_output=compile_output,
                compile_error=compile_error,
                timing_error=str(exc),
                sanitizer_report=sanitizer_report,
                volta_report=volta_report,
                verifier_report=verifier_report,
            )
