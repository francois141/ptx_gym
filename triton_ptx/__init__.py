"""Public API for Triton PTX utilities."""

from importlib import import_module

_EXPORTS = {
    "CompilationResult": "triton_ptx.evaluation.compilation",
    "EvaluatedCandidate": "triton_ptx.evaluation.types",
    "OutputVerifier": "triton_ptx.evaluation.verification",
    "PTXSignatureParameter": "triton_ptx.helpers.ptx",
    "Payload": "triton_ptx.evaluation.types",
    "Timing": "triton_ptx.evaluation.types",
    "TritonPTXCandidateEvaluator": "triton_ptx.evaluation.evaluate",
    "TritonPTXKernel": "triton_ptx.kernels.base",
    "available_kernels": "triton_ptx.kernels",
    "benchmark_operator": "triton_ptx.evaluation.evaluate",
    "clear_triton_cache": "triton_ptx.helpers.triton",
    "compile_ptx": "triton_ptx.evaluation.compilation",
    "dump_kernel_ptx": "triton_ptx.helpers.triton",
    "evaluate_ptx_performance": "triton_ptx.evaluation.performance",
    "extract_ptx": "triton_ptx.helpers.triton",
    "get_ptx_system_config": "triton_ptx.helpers.environment",
    "get_ptxas_path": "triton_ptx.helpers.environment",
    "is_gpu_available": "triton_ptx.helpers.environment",
    "jit_fixed_parameters": "triton_ptx.helpers.triton",
    "kernel_list": "triton_ptx.kernels",
    "parse_ptx_signature": "triton_ptx.helpers.ptx",
    "resolve_kernel": "triton_ptx.kernels",
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
