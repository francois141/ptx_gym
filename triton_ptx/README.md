# Triton PTX

Utilities for prompt generation, PTX extraction, evaluation, verification, and test-time scaling.

## Setup

Create a virtual environment from the repo root and install the dependencies used by `triton_ptx`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Common Commands

Run all tests and verification checks:

```bash
python3 -m pytest triton_ptx
```

Run test-time scaling for one kernel:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel
```

Run an OpenAI tool-calling agent loop where the model can directly call local
compile, correctness, and benchmark tools:

```bash
python3 -m triton_ptx.openai_agent_tools MatrixMultiplicationKernel
```

Run test-time scaling against Anthropic Claude Opus 4.8:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --anthropic
```

Run test-time scaling against Gemini 2.5 Pro:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --gemini
```

Write test-time scaling artifacts to a custom database directory:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --database-dir database
```

Extract embedded PTX from Triton kernels:

```bash
python3 -m triton_ptx.extract_ptx
```

Summarize archived winners from `database/`:

```bash
python3 -m triton_ptx.extract_ptx
python3 -m triton_ptx.measure_ptx --database-dir /path/to/database
```

Lint the package:

```bash
python3 -m ruff check triton_ptx
python3 -m ruff check --fix triton_ptx
```

## Notes

- Test-time scaling archives each run under `database/<timestamp>_<kernel>/`.
- `measure_ptx` reads archived `output_winner_*.json` files and reports the best valid `speedup_vs_triton` found for each kernel.
