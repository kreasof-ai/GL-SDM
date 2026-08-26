# Triton backend preparation

## First vertical slice

The first Triton lowering implements routed weighted reduction:

```text
indices [Q, K]       int32 or int64
weights [Q, K]       fp16, bf16, or fp32
values  [S, D]       fp16, bf16, or fp32

output[q, d] = sum(k, weights[q, k] * values[indices[q, k], d])
```

Products and reductions accumulate in fp32. Output and input gradients are cast
to the corresponding input dtype. Indices are precomputed; scoring, Top-k,
threshold compaction, capacity management, and expert computation are separate
operators.

This boundary is useful for MoE expert combination, sparse sequence or memory
gathers, and parameter-token reduction. It does not claim to replace the main
FlashAttention, grouped-GEMM, recurrent-scan, or SDM update kernels.

## Version 1 support

| Property | Supported |
| --- | --- |
| Device | CUDA through PyTorch and Triton |
| Input ranks | `indices/weights: [Q,K]`, `values: [S,D]` |
| Layout | Row-major contiguous |
| Route width | `1 <= K <= 64` and `K <= S` |
| Index dtype | int32, int64 |
| Floating dtype | fp16, bf16, fp32; weights and values may differ |
| Accumulation | fp32 |
| Value dimension | Positive, including non-powers of two |
| Duplicate indices | Forward and backward supported |
| Autograd | Weights and values |
| Index bounds | Optional synchronized validation before launch |

The forward kernel assigns a program to each `(query, value-dimension tile)`
and reduces the route axis in registers. The weight-gradient kernel computes one
dot product per route. The value-gradient kernel uses fp32 atomic additions so
duplicate routes are correct.

Atomic value gradients are not bitwise deterministic. A deterministic backward
would require sorting/segmentation or another conflict-free schedule and must be
introduced as a separate capability rather than silently changing semantics.

## Environment

Use a Linux CUDA development host for GPU execution. From `projects/urm`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,gpu,dev]'
PYTHONPATH=src python benchmarks/triton_preflight.py --require-ready
```

The preflight command reports exact Python, PyTorch, Triton, CUDA, GPU, compute
capability, and memory information. It exits non-zero with `--require-ready` if
the backend cannot run.

The validated stack is recorded in `benchmarks/validated-environment.json`
(A10G, PyTorch 2.8.0+cu129, Triton 3.4.0). The `gpu` extra deliberately pins
`triton>=3.4,<3.5`: the steady-state launch fast path uses Triton 3.4 internals
and capability-gates them at runtime, falling back to the standard launcher
(verified bitwise-identical) on any other version. Ruff is pinned in the `dev`
extra so lint/format results are reproducible.

CPU-only development remains supported. Triton modules are imported lazily, so
IR, contract, and NumPy tests do not require GPU dependencies.

## Correctness workflow

```bash
python -m pytest
python -m pytest tests/test_triton_routed_reduction.py -v
ruff check src tests benchmarks
```

The GPU test matrix covers:

- fp16, bf16, and fp32;
- int32 and int64 routes;
- `K` equal to 1, 3, 8, and 32;
- value dimensions smaller than, equal to, and larger than a tile;
- non-power-of-two shapes;
- duplicate indices and their weight/value gradients.

Before trusting benchmark results, also add the exact target GPU to CI or run
the GPU tests on every pinned benchmark environment. Import and compile checks
on a CPU host cannot validate generated PTX or numerical behavior.

## Benchmark workflow

List the committed cases:

```bash
PYTHONPATH=src python benchmarks/routed_reduce.py --list-cases
```

Run the smoke comparison and retain the JSON result:

```bash
PYTHONPATH=src python benchmarks/routed_reduce.py \
  --case smoke --backend both \
  --output results/routed-reduce-smoke.json
```

Run forward-plus-backward separately:

```bash
PYTHONPATH=src python benchmarks/routed_reduce.py \
  --case smoke --backend both --mode backward \
  --output results/routed-reduce-smoke-backward.json
```

The harness records cold compilation/first-call time separately from CUDA-event
steady-state timings. It emits median, p95, min/max, useful FLOP/s, peak
allocated memory, kernel launch parameters, correctness error, source revision,
and the complete runtime/GPU identity. Output follows
`benchmarks/result-schema.json`.

Index validation reads min/max back to the host and therefore synchronizes.
Normal API calls enable it for safety; the benchmark disables it after inputs
are generated from a trusted in-range source.

## Profiling and comparison workflow

After optimization work (see
[the optimization report](triton-optimization-report.md) for findings):

```bash
# Measured denominators: sustainable HBM bandwidth, FP32 CUDA-core peak,
# BF16 tensor-core peak. Never use vendor marketing numbers as MFU/MBU bases.
PYTHONPATH=src python benchmarks/measure_device_limits.py \
  --output results/device-limits.json

# Roofline profile of the complete committed case matrix (forward and backward):
# wall vs GPU kernel time vs host dispatch, MFU against the measured FP32
# CUDA-core peak, static traffic bounds with MBU labeled static_estimate,
# route statistics, atomic contention indicator, per-kernel breakdown.
# Counter-derived fields are explicit not_available when Nsight Compute
# lacks permissions on the host; nothing is fabricated.
PYTHONPATH=src python benchmarks/profile_roofline.py --case all --mode both

# Compare two result directories under correctness/p95/memory constraints:
python benchmarks/compare_results.py results/baseline results/final
```

The first production comparator is dense causal attention at four levels
(oracle, SDPA math, pinned FlashAttention upstream direct, and the same call
behind a URM adapter); see `benchmarks/dense_attention.py` and
`results/attention/dense-causal.json`.

## Acceptance gate before optimization

1. GPU forward tests pass for every supported dtype and committed shape class.
2. Weight and value gradients match the PyTorch baseline, including collisions.
3. Sanitizer or compute-sanitizer runs show no invalid accesses.
4. Benchmark JSON validates against the committed schema.
5. Cold compilation and steady-state measurements are reported separately.
6. Performance comparisons use identical input layouts, dtypes, accumulation,
   and index distributions.

These gates were passed and the first optimization campaign completed on
A10G; the full history of accepted and rejected candidates, roofline findings,
and remaining bottlenecks is in
[the optimization report](triton-optimization-report.md). Any future kernel
change must re-run the same gates plus the roofline matrix, and retain
before/after JSON under `results/`. A deterministic (sorted/segmented) backward
remains a separate future capability: it must not silently replace the atomic
path.

## Planned follow-on lowerings

1. Route compaction and stable Top-k/threshold selection.
2. Route grouping and expert dispatch metadata.
3. Grouped expert GEMM adapters rather than a replacement for every upstream
   expert kernel.
4. Token- and block-indexed sparse-attention gather paths.
5. Ordered recurrent scans for Mamba/GDN/KDA families.
6. Original SDM adapter, followed by page-local memory reads and transactional
   write merge/commit.

Each follow-on operation needs its own frozen tensor contract and oracle. The
routed-reduction signature must not grow into an untyped universal call.

## Triton references

- [Official vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
  for the JIT, launch-grid, mask, validation, and benchmark model.
- [`triton.language.atomic_add`](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)
  for collision-safe value-gradient accumulation.
- [`triton.language.range`](https://triton-lang.org/main/python-api/generated/triton.language.range.html)
  for the route and value-tile loops used by the kernels.
