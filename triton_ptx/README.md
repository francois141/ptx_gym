# Triton PTX

Utilities for prompt generation, PTX extraction, compilation, verification, and test-time scaling loops.

## Project Layout

- `kernels/`: Triton kernel implementations and embedded PTX payloads.
- `generator/`: Endpoints/connectors used to call LLMs.
- `evaluation/`: Candidate evaluation logic and serialized result schema.
- `helpers/`: Shared utilities for environment detection, storage, Triton helpers, and prompt extraction.
- `policy/`: Policy logic for test-time scaling selection.
- `prompts/`: Prompt builders for the test-time scaling workflow.
- `sandbox/`: Compilation, verification, and benchmarking helpers used by evaluation.
- `triton_ptx/*.py`: Top-level launchers and entrypoints.

## Environment

To see which GPU you are currently running on:

```bash
python3 -m triton_ptx.helpers.environment
```

## Commands

### Test-Time Scaling

Run the loop for a specific operator:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel
```

Write artifacts to a custom output directory:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --output-dir output
```

### Prompt Generation

Generate initial prompts for all kernels:

```bash
python3 -m triton_ptx.prompts.initial
```

Generate follow-up prompt templates:

```bash
python3 -m triton_ptx.prompts.next
```

### PTX Extraction and Compilation

Extract PTX values from Triton kernels:

```bash
python3 -m triton_ptx.extract_ptx
```

Verify compilation process:

```bash
python3 -m triton_ptx.sandbox.compilation
```

Verify verification process:

```bash
python3 -m triton_ptx.sandbox.verification
```

Summarize the best valid archived PTX winner per kernel from `database/`:

```bash
python3 -m triton_ptx.measure_ptx
```

Summarize a different archive root:

```bash
python3 -m triton_ptx.measure_ptx --database-dir /path/to/database
```

### Verification

Verify generated Triton kernels:

```bash
python3 -m triton_ptx.verify_triton_kernels
```

Verify generated PTX kernels:

```bash
python3 -m triton_ptx.verify_generated_ptx
```

## Linting `triton_ptx/`

Run Ruff directly on this folder:

```bash
python3 -m ruff check triton_ptx
```

Auto-fix lint issues when possible:

```bash
python3 -m ruff check --fix triton_ptx
```

## Test before commit

```bash
python3 -m ruff check --fix triton_ptx && \
python3 -m triton_ptx.prompts.initial && \
python3 -m triton_ptx.prompts.next && \
python3 -m triton_ptx.extract_ptx && \
python3 -m triton_ptx.sandbox.compilation && \
python3 -m triton_ptx.sandbox.verification && \
python3 -m triton_ptx.measure_ptx && \
python3 -m triton_ptx.verify_triton_kernels && \
python3 -m triton_ptx.verify_generated_ptx && \
printf '\033[32mSUCCESS\033[0m\n' || \
printf '\033[31mFAILURE\033[0m\n'
```

## Notes

- Test-time scaling writes the latest run to `output/` and also archives each run under `database/<timestamp>/`.
- `measure_ptx` reads archived `output_winner_*.json` files and reports the best valid `speedup_vs_triton` found for each kernel.
