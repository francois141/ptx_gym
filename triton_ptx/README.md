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
python3 -m triton_ptx.test_time_scaling_loop AddKernel --config triton_ptx/configs/test_time_scaling_openai.yaml
```

Run the deterministic fake AddKernel test-time-scaling loop:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --config triton_ptx/configs/test_time_scaling_fake_add.yaml
```

Run an OpenAI tool-calling agent loop where the model can directly call local
compile, correctness, and benchmark tools:

```bash
python3 -m triton_ptx.openai_agent_tools MatrixMultiplicationKernel
```

Run test-time scaling against Anthropic Claude Opus 4.8:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --config triton_ptx/configs/test_time_scaling_anthropic.yaml
```

Run test-time scaling against Gemini 2.5 Pro:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --config triton_ptx/configs/test_time_scaling_gemini.yaml
```

Write test-time scaling artifacts to a custom database directory:

```bash
python3 -m triton_ptx.test_time_scaling_loop AddKernel --config path/to/config.yaml
```

The test-time scaling command expects the kernel class name first and an
optional OmegaConf YAML path via `--config`. If `--config` is omitted, the
in-file default config in `triton_ptx/test_time_scaling_loop.py` is used.

The config controls loop parameters, generator selection, model settings, and
the archive directory:

```yaml
loop:
  rounds: 10
  k: 2
  max_retries: 3

generator:
  provider: openai
  model: gpt-5
  options:
    reasoning_effort: medium

storage:
  database_dir: database
```

Three ready-to-edit presets live in `triton_ptx/configs/`:

- `test_time_scaling_openai.yaml`
- `test_time_scaling_anthropic.yaml`
- `test_time_scaling_gemini.yaml`

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
