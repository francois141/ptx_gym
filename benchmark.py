import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

from kernels import operator_list, AddOperator, FancyFusedOperator
from helpers import *
from ptx_values import *

def benchmark_operators(operator_list):
    """
    Tests a list of operators for correctness and speedup.
    Benchmarks plain Triton, Triton PTX, and torch.compile().
    Triton is treated as the baseline for speedup calculations.
    Uses established check_similarity for correctness.
    """
    cache_path = Path.home() / ".triton" / "cache"
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"Cache cleared: {cache_path}")

    results = []

    for OperatorClass in operator_list:
        try:
            op = OperatorClass(ptx=ptx_values.get(OperatorClass.__name__, None))

            if getattr(op, "ptx", None) is None:
                continue

            values = op.get_random_input()

            # Correctness checks against the eager PyTorch reference.
            triton_res, _ = op.forward_triton(values)
            ptx_res = None
            if getattr(op, "ptx", None) is not None:
                ptx_res, _ = op.forward_triton(values, ptx=True)
            torch_res_eager = op.forward_torch(values)
            is_correct = check_similarity(triton_res, torch_res_eager)
            if ptx_res is not None:
                is_correct = is_correct and check_similarity(ptx_res, torch_res_eager)

            if not is_correct:
                print(f"Result: {OperatorClass.__name__} FAILED (Correctness check failed)")
                results.append({
                    "Operator Name": OperatorClass.__name__,
                    "Status": "FAILED",
                    "Triton (ms)": 0.0,
                    "Triton PTX (ms)": 0.0,
                    "Torch (compiled) (ms)": 0.0,
                    "PTX vs Triton Speedup": 0.0,
                    "Torch vs Triton Speedup": 0.0,
                })
                continue

            # Setup torch.compile baseline
            compiled_torch_forward = torch.compile(op.forward_torch)

            # Warmup
            for _ in range(10):
                op.forward_triton(values)
            if getattr(op, "ptx", None) is not None:
                for _ in range(10):
                    op.forward_triton(values, ptx=True)
            for _ in range(10):
                compiled_torch_forward(values)
            torch.cuda.synchronize()

            def benchmark_run(fn):
                total_time = 0.0
                runs = 0
                start_time = time.time()
                while (time.time() - start_time < 1.0) or (runs < 10):
                    t0 = time.time()
                    fn()
                    torch.cuda.synchronize()
                    total_time += (time.time() - t0)
                    runs += 1
                return (total_time / runs) * 1000

            avg_triton_ms = benchmark_run(lambda: op.forward_triton(values))
            avg_triton_ptx_ms = None
            if getattr(op, "ptx", None) is not None:
                avg_triton_ptx_ms = benchmark_run(lambda: op.forward_triton(values, ptx=True))
            avg_torch_ms = benchmark_run(lambda: compiled_torch_forward(values))

            ptx_speedup = None
            if avg_triton_ptx_ms is not None and avg_triton_ptx_ms > 0:
                ptx_speedup = avg_triton_ms / avg_triton_ptx_ms

            torch_speedup = avg_triton_ms / avg_torch_ms if avg_torch_ms > 0 else None

            if avg_triton_ptx_ms is not None:
                print(
                    f"Result: {OperatorClass.__name__} PASSED "
                    f"(Triton: {avg_triton_ms:.4f} ms, "
                    f"Triton PTX: {avg_triton_ptx_ms:.4f} ms "
                    f"({ptx_speedup:.2f}x vs Triton), "
                    f"Torch: {avg_torch_ms:.4f} ms "
                    f"({torch_speedup:.2f}x vs Triton))"
                )
            else:
                print(
                    f"Result: {OperatorClass.__name__} PASSED "
                    f"(Triton: {avg_triton_ms:.4f} ms, "
                    f"Torch: {avg_torch_ms:.4f} ms "
                    f"({torch_speedup:.2f}x vs Triton))"
                )

            results.append({
                "Operator Name": OperatorClass.__name__,
                "Status": "PASSED",
                "Triton (ms)": round(avg_triton_ms, 4),
                "Triton PTX (ms)": round(avg_triton_ptx_ms, 4) if avg_triton_ptx_ms is not None else None,
                "Torch (compiled) (ms)": round(avg_torch_ms, 4),
                "PTX vs Triton Speedup": round(ptx_speedup, 2) if ptx_speedup is not None else None,
                "Torch vs Triton Speedup": round(torch_speedup, 2) if torch_speedup is not None else None,
            })

        except Exception as e:
            print(f"Result: {OperatorClass.__name__} ERROR ({str(e)})")
            results.append({
                "Operator Name": OperatorClass.__name__,
                "Status": "ERROR",
                "Triton (ms)": 0.0,
                "Triton PTX (ms)": 0.0,
                "Torch (compiled) (ms)": 0.0,
                "PTX vs Triton Speedup": 0.0,
                "Torch vs Triton Speedup": 0.0,
            })

    df_results = pd.DataFrame(results)
    display(df_results)

    passed_ops = df_results[df_results['Status'] == 'PASSED']
    if not passed_ops.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_cols = ['Triton (ms)', 'Torch (compiled) (ms)']
        if 'Triton PTX (ms)' in passed_ops.columns and passed_ops['Triton PTX (ms)'].notna().any():
            plot_cols.insert(1, 'Triton PTX (ms)')
        passed_ops.plot(x='Operator Name', y=plot_cols, kind='bar', ax=ax)
        plt.title('Execution Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    return df_results

benchmark_operators(operator_list)
