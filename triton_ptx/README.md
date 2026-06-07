# Triton PTX

Utilities for prompt generation, PTX extraction, evaluation, verification, and test-time scaling.

## Layout

- `kernels/`: Triton kernel implementations and embedded PTX payloads.
- `generator/`: LLM client wrappers used during candidate generation.
- `evaluation/`: Candidate evaluation logic plus `evaluation/sandbox/` helpers for compilation, verification, and benchmarking.
- `helpers/`: Shared utilities for environment detection, storage, and Triton integration.
- `policy/`: Candidate selection logic.
- `prompts/`: Prompt builders for the test-time scaling workflow.
- `triton_ptx/*.py`: User-facing entrypoints.

## Common Commands

Inspect the current GPU environment:

```bash
python3 -m triton_ptx.helpers.environment
```

Run test-time scaling for one kernel:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel
```

Write test-time scaling artifacts to a custom directory:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --output-dir output
```

Generate prompt payloads:

```bash
python3 -m triton_ptx.prompts.initial
python3 -m triton_ptx.prompts.next
```

Extract embedded PTX from Triton kernels:

```bash
python3 -m triton_ptx.extract_ptx
```

Summarize archived winners from `database/`:

```bash
python3 -m triton_ptx.measure_ptx
python3 -m triton_ptx.measure_ptx --database-dir /path/to/database
```

Verify Triton kernels against PyTorch:

```bash
python3 -m triton_ptx.verify_triton_kernels
```

Verify PTX overrides against Triton kernels:

```bash
python3 -m triton_ptx.verify_generated_ptx
```

Lint the package:

```bash
python3 -m ruff check triton_ptx
python3 -m ruff check --fix triton_ptx
```

## Notes

- Test-time scaling writes the latest run to `output/` and archives runs under `database/<timestamp>/`.
- `measure_ptx` reads archived `output_winner_*.json` files and reports the best valid `speedup_vs_triton` found for each kernel.
