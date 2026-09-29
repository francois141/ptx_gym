"""Evaluation API for Triton PTX candidates."""

from importlib import import_module

_EXPORTS = {
    "BaseCandidateEvaluator": "ptx_gym.evaluation.base",
    "BaseVerifier": "ptx_gym.evaluation.base",
    "CompilationResult": "ptx_gym.evaluation.compilation",
    "EvaluatedCandidate": "ptx_gym.evaluation.types",
    "OutputVerifier": "ptx_gym.evaluation.verification",
    "NCUResult": "ptx_gym.evaluation.ncu",
    "Payload": "ptx_gym.evaluation.types",
    "Timing": "ptx_gym.evaluation.types",
    "TritonPTXCandidateEvaluator": "ptx_gym.evaluation.evaluate",
    "benchmark_operator": "ptx_gym.evaluation.evaluate",
    "compile_ptx": "ptx_gym.evaluation.compilation",
    "diagnose_ptx": "ptx_gym.evaluation.sanitizer",
    "profile_ptx_with_ncu": "ptx_gym.evaluation.ncu",
    "run_candidate": "ptx_gym.evaluation.run_candidate",
    "evaluate_ptx_performance": "ptx_gym.evaluation.performance",
    "run_ptx_compilation": "ptx_gym.evaluation.compilation",
    "verify_ptx_with_volta": "ptx_gym.evaluation.volta",
    "volta_binary": "ptx_gym.evaluation.volta",
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
