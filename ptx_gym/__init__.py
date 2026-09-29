"""Public API for Triton PTX utilities."""

import os
from importlib import import_module

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

_EXPORTS = {
    "CompilationResult": "ptx_gym.evaluation.compilation",
    "EvaluatedCandidate": "ptx_gym.evaluation.types",
    "KernelTuningResult": "ptx_gym.kernels.base",
    "OutputVerifier": "ptx_gym.evaluation.verification",
    "PTXSignatureParameter": "ptx_gym.helpers.ptx",
    "Payload": "ptx_gym.evaluation.types",
    "Timing": "ptx_gym.evaluation.types",
    "TritonPTXCandidateEvaluator": "ptx_gym.evaluation.evaluate",
    "TritonPTXKernel": "ptx_gym.kernels.base",
    "available_kernels": "ptx_gym.kernels",
    "benchmark_operator": "ptx_gym.evaluation.evaluate",
    "clear_triton_cache": "ptx_gym.helpers.triton",
    "compile_ptx": "ptx_gym.evaluation.compilation",
    "dump_kernel_ptx": "ptx_gym.api",
    "evaluate_candidate": "ptx_gym.api",
    "evaluate_ptx_performance": "ptx_gym.evaluation.performance",
    "extract_ptx": "ptx_gym.helpers.triton",
    "get_kernel_data": "ptx_gym.api",
    "get_ptx_system_config": "ptx_gym.helpers.environment",
    "get_ptxas_path": "ptx_gym.helpers.environment",
    "is_gpu_available": "ptx_gym.helpers.environment",
    "jit_fixed_parameters": "ptx_gym.helpers.triton",
    "kernel_list": "ptx_gym.kernels",
    "list_kernels": "ptx_gym.api",
    "parse_ptx_signature": "ptx_gym.helpers.ptx",
    "resolve_kernel": "ptx_gym.kernels",
    "run_ptx_compilation": "ptx_gym.evaluation.compilation",
}

__all__ = [name for name in sorted(_EXPORTS)]


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
