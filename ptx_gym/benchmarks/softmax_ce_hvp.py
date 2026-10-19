"""Run HVP smoke checks, full self-PTX evaluation, or baseline shape sweeps.

    python -m ptx_gym.benchmarks.softmax_ce_hvp --mode smoke

The CLI and --help work without Triton. Actual runs require NVIDIA CUDA and
the patched Triton built from this checkout. No mode generates optimized PTX.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KERNEL_IDS = tuple(
    f"SoftmaxCrossEntropyHVP{variant}{precision}Kernel"
    for variant in ("", "Cached")
    for precision in ("Float32", "Float16", "BFloat16")
)
SWEEP_SHAPES = (
    (1, 10),
    (32, 256),
    (256, 1000),
    (4096, 1000),
    (1024, 4096),
    (256, 8192),
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_shape(value: str) -> tuple[int, int]:
    try:
        batch, classes = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a shape such as 4096x1000.") from exc
    if batch <= 0 or not 1 <= classes <= 8192:
        raise argparse.ArgumentTypeError("Require batch > 0 and 1 <= classes <= 8192.")
    return batch, classes


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("smoke", "full", "sweep"), default="smoke")
    result.add_argument(
        "--kernel",
        choices=KERNEL_IDS,
        action="append",
        help="Repeat to select tasks; default: all six.",
    )
    result.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        help="Sweep-only BxC shape; repeat to replace the default sweep.",
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/softmax_ce_hvp"),
        help="Parent for a fresh timestamped run directory.",
    )
    return result


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _command_description(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def _preflight(environment: dict, mode: str) -> None:
    import torch

    environment.update(
        torch_version=torch.__version__, torch_cuda_version=torch.version.cuda
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this benchmark on NVIDIA Linux with the patched Triton installed."
        )
    capability = torch.cuda.get_device_capability()
    environment.update(
        device_name=torch.cuda.get_device_name(),
        compute_capability=list(capability),
        driver_version=_command_description(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
    )

    import triton
    from triton.backends.nvidia.compiler import get_ptxas

    from ptx_gym.helpers.environment import get_ptx_system_config, get_ptxas_path

    if "ptx" not in inspect.signature(triton.jit).parameters:
        raise RuntimeError(
            "This Triton lacks PTX injection. Build the patched Triton from the repository root."
        )
    version, target, address_size = get_ptx_system_config()
    assembler = get_ptxas_path(target).resolve()
    triton_assembler = Path(
        get_ptxas(capability[0] * 10 + capability[1]).path
    ).resolve()
    if assembler != triton_assembler:
        raise RuntimeError(
            f"Gym assembler {assembler} differs from Triton's {triton_assembler}; align the build/overrides first."
        )
    cache_dir = (
        Path(
            os.environ.setdefault(
                "TRITON_CACHE_DIR",
                str(REPO_ROOT / ".cache" / "softmax_ce_hvp" / target),
            )
        )
        .expanduser()
        .resolve()
    )
    # Normalize a user-supplied relative cache path before child processes run.
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    environment.update(
        triton_version=triton.__version__,
        triton_module=str(Path(triton.__file__).resolve()),
        device_name=torch.cuda.get_device_name(),
        compute_capability=list(capability),
        ptx_version=version,
        target=target,
        address_size=address_size,
        assembler=str(assembler),
        assembler_version=subprocess.check_output(
            [str(assembler), "--version"], text=True
        ).strip(),
        triton_cache_dir=str(cache_dir),
        bf16_supported=torch.cuda.is_bf16_supported(),
    )
    if not environment["bf16_supported"]:
        raise RuntimeError("This suite requires a GPU with BF16 support.")
    for variable in ("PTX_MEMORY_SANITIZER", "NCU_PATH"):
        executable = shutil.which(os.environ.get(variable, ""))
        environment[variable] = executable
        if executable:
            environment[f"{variable}_version"] = _command_description(
                [executable, "--version"]
            )
        if mode == "full" and not executable:
            raise RuntimeError(
                f"Full mode requires {variable} set to an executable path."
            )


def _task_metadata(kernel_cls, shape, tuning, compiled=None) -> dict:
    vector_dtype = str(kernel_cls.vector_dtype).removeprefix("torch.")
    result = {
        "kernel_id": kernel_cls.__name__,
        "shape": list(shape),
        "variant": "cached" if kernel_cls.cached_probabilities else "fused",
        "input_dtype": "float32" if kernel_cls.cached_probabilities else vector_dtype,
        "vector_dtype": vector_dtype,
        "output_dtype": "float32",
        "loss_reduction": "sum",
        "tuning_config": dict(tuning),
        "num_stages": 1,
        "grid": [shape[0]],
        "constexpr_values": {"BLOCK_SIZE": 1 << (shape[1] - 1).bit_length()},
        "baseline_threads": [tuning["num_warps"] * 32, 1, 1],
    }
    if compiled is not None:
        result["resources"] = {
            "registers_per_thread": compiled.n_regs,
            "spills": compiled.n_spills,
            "shared_memory_bytes": compiled.metadata.shared,
        }
    return result


def _check_output(actual, expected) -> float:
    import torch

    if actual.dtype != torch.float32 or not torch.isfinite(actual).all():
        raise AssertionError("HVP output must be finite FP32.")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    return float((actual - expected).abs().max().item())


def _smoke(kernel_id: str, directory: Path) -> dict:
    import torch

    from ptx_gym import Payload, resolve_kernel
    from ptx_gym.evaluation.compilation import compile_ptx
    from ptx_gym.helpers.kernels import extract_specification_from_operator
    from ptx_gym.helpers.ptx import parse_ptx_signature
    from ptx_gym.helpers.triton import extract_ptx

    op = resolve_kernel(kernel_id)(autotune=False)
    errors = {"triton_vs_torch": 0.0, "ptx_vs_torch": 0.0}
    with torch.inference_mode():
        for iteration in range(3):
            inputs = op.get_random_input()
            originals = [x.clone() for x in inputs]
            expected = op.forward_torch(inputs)
            actual, compiled = op.forward_triton(inputs)
            errors["triton_vs_torch"] = max(
                errors["triton_vs_torch"], _check_output(actual, expected)
            )
            if iteration == 0:
                ptx = extract_ptx(compiled)
                if not ptx:
                    raise RuntimeError("The compiled reference contains no PTX.")
                (directory / "reference.ptx").write_text(ptx, encoding="utf-8")
                spec = extract_specification_from_operator(op)
                metadata = _task_metadata(
                    type(op), inputs[0].shape, op.best_config, compiled
                )
                metadata.update(
                    source=spec.source,
                    shape_information=spec.shape_information,
                    constexpr_values=spec.constexpr_values,
                    ptx_signature=[
                        {"name": p.name, "ptx_type": p.ptx_type}
                        for p in parse_ptx_signature(ptx)
                    ],
                )
                _write_json(directory / "metadata.json", metadata)
                payload = Payload(ptx=ptx, threads_x=op.num_warps * 32)
                compile_report = compile_ptx(payload).to_dict()
                _write_json(directory / "compilation.json", compile_report)
                if not compile_report["compiles"]:
                    raise RuntimeError(compile_report["error"])
                op.set_ptx(payload.to_launch_dict())
            injected, _ = op.forward_triton(inputs, ptx=True)
            errors["ptx_vs_torch"] = max(
                errors["ptx_vs_torch"], _check_output(injected, expected)
            )
            for value, original in zip(inputs, originals):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
    return {
        "passed": True,
        "iterations": 3,
        "max_abs_errors": errors,
        "autotuned": False,
        "sanitizer_run": False,
        "ncu_run": False,
        "candidate_kind": "triton_generated_self_ptx",
    }


def _full_report_complete(result: dict) -> bool:
    # Gym's `passed` only covers compilation/correctness. Full mode also
    # requires an actual successful profile, not merely an attempted one.
    ncu = result.get("ncu_report", {})
    return bool(
        result.get("passed")
        and result.get("sanitizer_report", {}).get("clean") is True
        and ncu.get("available") is True
        and ncu.get("return_code") == 0
        and ncu.get("metrics")
        and result.get("p50") is not None
    )


def _full(kernel_id: str, directory: Path) -> dict:
    from ptx_gym import (
        dump_kernel_ptx,
        evaluate_candidate,
        get_kernel_data,
        resolve_kernel,
    )
    from ptx_gym.kernels.common.softmax_ce_hvp import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_NUM_CLASSES,
    )

    # These three public API calls share one cached, tuned operator.
    data = get_kernel_data(kernel_id)
    ptx = dump_kernel_ptx(kernel_id)
    metadata = _task_metadata(
        resolve_kernel(kernel_id),
        (DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLASSES),
        {"num_warps": data["num_warps"], "num_stages": 1},
    )
    metadata["kernel_data"] = data
    _write_json(directory / "metadata.json", metadata)
    (directory / "reference.ptx").write_text(ptx, encoding="utf-8")
    result = evaluate_candidate(
        kernel_id, {"ptx": ptx, "num_threads_x": data["num_warps"] * 32}
    )
    _write_json(directory / "gym_result.json", result)
    return {
        "passed": _full_report_complete(result),
        "gym_passed": result["passed"],
        "message": (
            result["message"]
            if _full_report_complete(result) or not result["passed"]
            else "Gym correctness passed, but full timing/tool evidence is incomplete; see gym_result.json."
        ),
        "candidate_kind": "triton_generated_self_ptx",
        "speedup_vs_triton": result.get("speedup_vs_triton"),
        "full_checks_complete": _full_report_complete(result),
    }


def _sweep(kernel_id: str, directory: Path, shape: tuple[int, int]) -> dict:
    import torch
    import triton

    from ptx_gym import resolve_kernel
    from ptx_gym.helpers.triton import extract_ptx

    op = resolve_kernel(kernel_id)(batch_size=shape[0], num_classes=shape[1])
    with torch.inference_mode():
        inputs = op.get_random_input()
        expected = op.forward_torch(inputs)
        actual, compiled = op.forward_triton(inputs)
        max_error = _check_output(actual, expected)
        ptx = extract_ptx(compiled)
        if not ptx:
            raise RuntimeError("The compiled baseline contains no PTX.")
        (directory / "reference.ptx").write_text(ptx, encoding="utf-8")
        _write_json(
            directory / "metadata.json",
            _task_metadata(type(op), shape, op.best_config, compiled),
        )
        timings = {}
        for name, fn in (
            ("triton", lambda: op.forward_triton(inputs)),
            ("torch", lambda: op.forward_torch(inputs)),
        ):
            quantiles = triton.testing.do_bench(
                fn, warmup=100, rep=200, quantiles=[0.2, 0.5, 0.8]
            )
            timings[name] = dict(
                zip(("p20_ms", "p50_ms", "p80_ms"), map(float, quantiles))
            )
    return {
        "passed": True,
        "baseline_only": True,
        "max_abs_error": max_error,
        "timings": timings,
        "sanitizer_run": False,
        "ncu_run": False,
    }


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.shape and args.mode != "sweep":
        cli.error(
            "--shape is only supported in sweep mode; Gym evaluation uses canonical tasks."
        )
    now = datetime.now(timezone.utc)
    run_directory = args.output_dir / f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{args.mode}"
    run_directory.mkdir(parents=True, exist_ok=False)
    environment = {
        "timestamp_utc": now.isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain", "--untracked-files=no")),
    }
    summary = {"mode": args.mode, "seed": args.seed, "passed": False, "tasks": []}
    print(f"Artifacts: {run_directory.resolve()}", flush=True)
    try:
        _preflight(environment, args.mode)
        import torch

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        kernels = tuple(dict.fromkeys(args.kernel or KERNEL_IDS))
        shapes = (
            tuple(dict.fromkeys(args.shape or SWEEP_SHAPES))
            if args.mode == "sweep"
            else (None,)
        )
        for kernel_id in kernels:
            for shape in shapes:
                task_name = (
                    kernel_id if shape is None else f"{kernel_id}-{shape[0]}x{shape[1]}"
                )
                directory = run_directory / task_name
                directory.mkdir()
                print(f"Running {task_name}", flush=True)
                try:
                    if args.mode == "smoke":
                        report = _smoke(kernel_id, directory)
                    elif args.mode == "full":
                        report = _full(kernel_id, directory)
                    else:
                        report = _sweep(kernel_id, directory, shape)
                except Exception as exc:  # noqa: BLE001 -- retain per-task failure artifacts
                    report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                _write_json(directory / "report.json", report)
                summary["tasks"].append({"task": task_name, **report})
                print(
                    f"{'PASS' if report['passed'] else 'FAIL'} {task_name}", flush=True
                )
                if not report["passed"]:
                    print(
                        report.get("error") or report.get("message", "See report.json"),
                        flush=True,
                    )
        summary["passed"] = all(task["passed"] for task in summary["tasks"])
    except Exception as exc:  # noqa: BLE001 -- report setup failures without inventing results
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(summary["error"], file=sys.stderr)
    _write_json(run_directory / "environment.json", environment)
    _write_json(run_directory / "summary.json", summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
