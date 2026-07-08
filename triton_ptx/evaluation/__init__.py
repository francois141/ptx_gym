"""Evaluation API for Triton PTX candidates."""

from importlib import import_module

_EXPORTS = {
    "BaseCandidateEvaluator": "triton_ptx.evaluation.base",
    "BaseVerifier": "triton_ptx.evaluation.base",
    "CompilationResult": "triton_ptx.evaluation.compilation",
    "EvaluatedCandidate": "triton_ptx.evaluation.types",
    "OutputVerifier": "triton_ptx.evaluation.verification",
    "NCUResult": "triton_ptx.evaluation.ncu",
    "Payload": "triton_ptx.evaluation.types",
    "Timing": "triton_ptx.evaluation.types",
    "TritonPTXCandidateEvaluator": "triton_ptx.evaluation.evaluate",
    "benchmark_operator": "triton_ptx.evaluation.evaluate",
    "compile_ptx": "triton_ptx.evaluation.compilation",
    "diagnose_ptx": "triton_ptx.evaluation.sanitizer",
    "profile_ptx_with_ncu": "triton_ptx.evaluation.ncu",
    "run_candidate": "triton_ptx.evaluation.run_candidate",
    "evaluate_ptx_performance": "triton_ptx.evaluation.performance",
    "run_ptx_compilation": "triton_ptx.evaluation.compilation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
