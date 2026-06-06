# Triton PTX

Utilities for prompt generation, PTX extraction, compilation, verification, and test-time scaling loops.

## Project Layout

- `kernel/`: Triton kernel implementations.
- `generator/`: Endpoints/connectors used to call LLMs.
- `evaluation/`: Main evaluation loop.
- `policy/`: Policy logic for test-time scaling selection.
- `prompts/`: Prompt builders for the test-time scaling workflow.
- `sandbox/`: Isolated environment utilities (for example, compilation).
- `triton_ptx/*.py`: Top-level launchers and entrypoints.

## Environment

To see which gpu you are currently running on

```bash
python3 -m triton_ptx.helpers.environment
```

## Commands

### Test-Time Scaling

Run the loop for a specific operator:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel
```

Run with default configuration:

```bash
python3 -m triton_ptx.test_time_scaling_loop
```

### Prompt Generation

Generate the initial prompt:

```bash
python3 -m triton_ptx.prompts.initial
```

Generate the next prompt:

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

Verify evaluation process:

```bash
python3 -m triton_ptx.sandbox.evaluation
```

Measure PTX performance:

```bash
python3 -m triton_ptx.measure_ptx
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
