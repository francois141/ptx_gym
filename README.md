<p align="center">
  <img alt="PTX Gym logo" src="logo.png" width="200">
</p>

<p align="center">
  <strong>An environment for testing whether LLMs can compile Triton kernels (or build compilers that compile kernels) to PTX directly, without the Triton compiler's lowering pipeline.</strong>
</p>

<p align="center">
  <img alt="Status: research preview" src="https://img.shields.io/badge/status-research%20preview-orange">
  <img alt="Task: Triton to PTX" src="https://img.shields.io/badge/task-Triton%20%E2%86%92%20PTX-blue">
  <img alt="Hardware: NVIDIA GPU" src="https://img.shields.io/badge/hardware-NVIDIA%20GPU-76B900">
</p>

<p align="center">
  <a href="#quickrun">Quickrun</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#kernel-suite">Kernel suite</a>
</p>

## Quickrun

```bash
git clone --recursive https://github.com/francois141/ptx_gym.git
cd ptx_gym

uv venv .ptx_gym_env
source .ptx_gym_env/bin/activate
uv pip install torch
MAX_JOBS=64 uv pip install -e . -v     # builds Triton, installs triton + ptx_gym

export PTX_MEMORY_SANITIZER=$(which compute-sanitizer)
export NCU_PATH=$(which ncu)

python quick_start.py 
```

## Overview

A normal Triton compile lowers a kernel step by step: TTIR, then TTGIR, then
LLVM/NVVM IR, then PTX, with a fixed set of hand-written passes. **AI
lowering** removes that pipeline. A model writes the PTX for a Triton kernel
directly, and the kernel keeps the same interface it would have had after a
normal compile.

PTX Gym checks whether such PTX is valid and fast. For a given kernel it:

1. **Fixes the compilation contract** `(c, I, h)`:
   - `c`: compile-time values (`tl.constexpr` values, shapes, strides),
   - `I`: runtime launch interface (grid, threads per CTA, argument ABI),
   - `h`: target GPU (for example `sm_89`, `sm_90a`, `sm_100a`).

   Each baseline is autotuned on the target GPU first. The best configuration
   found is then used for every comparison.
2. **Exposes** the Triton source, the constexpr values, and the PTX entry
   signature Triton would produce. With these, a model can write a drop-in
   PTX body.
3. **Accepts** a candidate PTX program plus its CTA dimensions.
4. **Assembles, sanitizes, tests, benchmarks, and profiles** the candidate.
   It uses the same Triton launcher, so the grid, the argument passing, and
   the launch path are identical to the Triton reference.
5. **Returns** a structured, JSON-serializable result. An agent can use this
   result to improve its next candidate.

A candidate **passes** when it assembles for the target, passes the memory
and race checks, and gives the same outputs as the Triton reference within
tolerance on randomized inputs. Its score is `speedup_vs_triton`, the Triton
p50 latency divided by the candidate's p50 latency.

PTX Gym does not depend on any model. It evaluates PTX and does not care how
that PTX was produced. Feel free to use the environment in many creative ways!

### Environment variables

| Variable | Needed for |
| --- | --- |
| `PTX_MEMORY_SANITIZER` | Path to the `compute-sanitizer` executable. Required when the sanitizer is on, which is the default. |
| `NCU_PATH` | Path to the `ncu` executable. Required when NCU profiling is on, which is the default. |

## Quick start

The script below walks through the whole workflow: pick a kernel, read what a
model needs to write PTX for it, submit a candidate, and read the verdict. Run
it after setting the [environment variables](#environment-variables).

```python
from ptx_gym import (
    Payload,
    TritonPTXCandidateEvaluator,
    dump_kernel_ptx,
    evaluate_candidate,
    get_kernel_data,
    list_kernels,
    resolve_kernel,
)
from ptx_gym.helpers.triton import dump_kernel_ptx as dump_operator_ptx

KERNEL_ID = "ReLUFloat16Kernel"

# 1. List the kernels. Any of these names can be used as a kernel_id.
print(list_kernels())

# 2. Get everything a model needs to write PTX for the kernel. The first call
#    autotunes the Triton baseline and caches it for the rest of the process,
#    so get_kernel_data, dump_kernel_ptx, and evaluate_candidate all agree on
#    the same launch configuration.
data = get_kernel_data(KERNEL_ID)
print(data["source"])            # Triton source of the kernel
print(data["system"])            # {"target": "sm_..", "version": .., "address_size": 64}
print(data["num_warps"])         # CTA size is num_warps * 32 threads
print(data["constexpr_values"])  # constexpr values baked into the kernel
print(data["ptx_signature"])     # [{"name": .., "ptx_type": ..}, ...] the entry must match

# 3. Submit a candidate. Here it is Triton's own PTX, which should pass with a
#    speedup close to 1.0 (a quick check that the setup works). Put your
#    model's PTX here instead.
candidate_ptx = dump_kernel_ptx(KERNEL_ID)
result = evaluate_candidate(
    KERNEL_ID,
    {"ptx": candidate_ptx, "num_threads_x": data["num_warps"] * 32},
)

# 4. Read the verdict. The result is a plain JSON dict.
print(result["passed"], result["message"])
if result["passed"]:
    print(f"{result['p50']:.4f} ms vs {result['triton_p50']:.4f} ms Triton (p50), "
          f"{result['speedup_vs_triton']:.3f}x speedup")

# 5. Optional: build the evaluator yourself to choose which checks run.
#    It autotunes its own baseline, whose configuration can differ from the
#    one above, so take the reference PTX and CTA size from evaluator.operator.
evaluator = TritonPTXCandidateEvaluator(
    resolve_kernel(KERNEL_ID),
    enable_sanitizer=True,    # compute-sanitizer memcheck/racecheck/...
    enable_ncu_report=True,   # Nsight Compute profile for passing candidates
    enable_volta=False,       # symbolic equivalence proof (see below)
)
operator = evaluator.operator
result = evaluator.evaluate(
    Payload(ptx=dump_operator_ptx(operator), threads_x=operator.num_warps * 32)
)
print(result.passed, result.speedup_vs_triton, result.message)
print(result.to_json())
```

A few rules apply to every candidate:

- `ptx` is the full PTX module. Its entry signature must match
  `ptx_signature`.
- `threads_x`, `threads_y`, and `threads_z` set the CTA dimensions, at most
  1024 threads in total. If the PTX declares `.reqntid` or `.maxntid`, these
  values must agree with it. In dicts, `num_threads_x`, `num_threads_y`, and
  `num_threads_z` are accepted as aliases.
- Autotuning picks among configurations that often tie, so a new process may
  choose a different one. Take the kernel data, the reference PTX, and the
  evaluation from the same process.


### Formal verification with Volta

Releasing more details about this section soon!

### Triton AI Compiler

Releasing more details about this section soon!
## Kernel suite

Each kernel is a subclass of
[`TritonPTXKernel`](ptx_gym/kernels/base.py). A kernel instance fixes the
problem size and declares the autotuning space. Pass its class name as the
`kernel_id`.

**Common kernels** ([`kernels/common`](ptx_gym/kernels/common)). Every one
comes in `Float16` and `Float8` (E4M3) versions:

| Family | Classes |
| --- | --- |
| GEMM-like | `MatrixMultiplicationFloat{16,8}`, `MatrixVectorMultiplicationFloat*Kernel`, `Convolution2DFloat*Kernel`, `FusedGEMMAddSiLUFloat*Kernel`, `FusedGEMMAddGELUFloat*Kernel` |
| Reductions and normalization | `SoftmaxFloat*Kernel`, `RMSNormFloat*Kernel`, `ReductionSumFloat*Kernel`, `RoPEFloat*Kernel` |
| Elementwise | `ReLUFloat*Kernel`, `SiLUFloat*Kernel`, `SigmoidFloat*Kernel`, `GELUFloat*Kernel`, `SwiGLUFloat*Kernel`, `AddFloat*Kernel` |

**Kernels from recent papers** ([`kernels/papers`](ptx_gym/kernels/papers)).
These are the authors' official Triton implementations. We only removed masks
that the fixed problem sizes make unnecessary.

| Paper | Classes |
| --- | --- |
| FlashAttention (NeurIPS 2022) | `FlashAttentionNeurIPS2022Forward`, `...Backward` |
| FlashSinkhorn (ICML 2026) | `FlashSinkhornFusedSchurMatvec` |
| Forgetting Attention (ICLR 2025) | `ForgettingAttentionICLR2025Forward`, `...Backward` |
| SageAttention (ICLR 2025) | `SageAttentionICLR2025` |
| BitDelta (NeurIPS 2024) | `BitDeltaNeurIPS2024Matmul`, `BitDeltaNeurIPS2024BatchedMatmul` |
| Lion (NeurIPS 2023) | `LionNeurIPS2023Optimizer` |
| Dion | `Dion2TritonPostOrthogonalize` |
| Mamba-2 (ICML 2024) | `Mamba2ChunkScanForward`, `Mamba2ChunkStateForward` |
| Others | `MambaICLR2026*`, `ChanMixICLR2026*`, `HouseholderDiagonalizedLinearAttentionICLR2026*` |

**Semantic probes.** These check whether a model follows what the source
actually computes:

- [`kernels/random`](ptx_gym/kernels/random): `RandomKernel1`-`5`. Each is
  about 150 lines of generated integer code that implements no known
  algorithm. They use bitwise operations, hashing, `tl.sum`, `tl.max`,
  `tl.static_range`, `tl.where`, and hints.
- [`kernels/adversial`](ptx_gym/kernels/adversial): kernels whose names,
  comments, or familiar structure point to the wrong computation. Examples: a
  matmul that subtracts instead of adding, a FlashAttention variant with a
  modified scale, a "ReLU" kernel whose comments ask for GELU, and an FP32
  matmul (`MatrixMultiplicationFloat32`) whose comment claims tensor cores are
  fine despite `input_precision="ieee"`.

  ## Citation

```bibtex
@misc{costa2026aicompilercompilingtriton,
      title={AI as a Compiler: Compiling Triton kernels without the Triton compiler}, 
      author={François Costa and Charly Castes and Thomas Bourgeat and Azalia Mirhoseini},
      year={2026},
      eprint={2609.36800},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.36800}, 
}
```
