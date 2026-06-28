from __future__ import annotations

from triton_ptx.evaluation.base import BaseCandidateEvaluator
from triton_ptx.evaluation.compilation import compile_ptx
from triton_ptx.evaluation.performance import evaluate_ptx_performance
from triton_ptx.evaluation.types import EvaluatedCandidate, Payload
from triton_ptx.evaluation.verification import OutputVerifier
from triton_ptx.helpers.triton import clear_triton_cache


def benchmark_operator(operator):
    """
    Backward-compatible single-operator benchmarking entrypoint.

    Returns the same metrics mapping as evaluate_ptx_performance(), including
    a "ptx" Timing object used by triton_ptx.evaluation.
    """
    inputs = operator.get_random_input()
    return evaluate_ptx_performance(operator, inputs)


class TritonPTXCandidateEvaluator(BaseCandidateEvaluator):
    """
    Evaluates PTX candidates by compiling, checking correctness, and benchmarking.
    """

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
                payload=launch_payload,
                compiles=False,
                correct=False,
                message=compile_error or "Compilation failed",
                compile_output=compile_output,
                compile_error=compile_error,
            )

        try:
            operator = self.operator_cls(ptx=launch_payload)
            verifier = OutputVerifier()
            correct = verifier.verify(operator)
            verifier_report = getattr(verifier, "last_report", {})
            
        except Exception as exc:
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                payload=launch_payload,
                compiles=True,
                correct=False,
                message=f"Correctness check crashed: {type(exc).__name__}: {exc}",
                compile_output=compile_output,
                compile_error=compile_error,
                verifier_report={},
            )

        if not correct:
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                payload=launch_payload,
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
                payload=launch_payload,
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
                verifier_report=verifier_report,
            )

        except Exception as exc:
            return EvaluatedCandidate.failed(
                kernel_name=self.kernel_name,
                git_commit_hash=self.git_commit_hash,
                payload=launch_payload,
                compiles=True,
                correct=False,
                message=f"Benchmark failed: {type(exc).__name__}: {exc}",
                compile_output=compile_output,
                compile_error=compile_error,
                timing_error=str(exc),
                verifier_report=verifier_report,
            )
