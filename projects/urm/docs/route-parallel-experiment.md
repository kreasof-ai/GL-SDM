# Ordered route parallelism and indexed state residency

This is an isolated training/prefill experiment from `ae95829` and `bdae3d7`.
No production backend, compiler schedule, model, optimizer, data, frozen gate, or
decode dispatch is changed. Triangular chunk integration remains rejected.

## Mechanism and references

Read [M²RNN v2](https://arxiv.org/html/2603.14360v2), especially §4.1–4.2, and its
[pinned implementation](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844/xma/layers/m2rnn/triton_implementation).
The code traverses tokens in order, retaining a local state tile and transition
weights; backward uses full-state histories. Only the within-timestep parallelism
and state reuse motivate this experiment. Dense transitions, tensor-core claims,
and its history scheme are not URM mechanisms. At URM's P=12,T=1024,S=4096,D=64,
full BF16 histories would be 6 GiB versus 192 MiB for selected write/read histories.

Also read ATMA at `fcefd2d75f9db73c16850343e20b57b3b729b3ea`:
[gated-delta decode](https://github.com/kreasof-ai/atma/blob/fcefd2d75f9db73c16850343e20b57b3b729b3ea/kernel/gated_delta_triton.py),
[inference fusions](https://github.com/kreasof-ai/atma/blob/fcefd2d75f9db73c16850343e20b57b3b729b3ea/kernel/inference_ops_triton.py), and
[kernel benchmark documentation](https://github.com/kreasof-ai/atma/blob/fcefd2d75f9db73c16850343e20b57b3b729b3ea/docs/kernel.md).
Its dense FP32 decode state is not URM's sparse BF16 training recurrence. Relevant
discipline: preserve intermediate rounding, time full workloads, retain losing
prototypes outside model dispatch, and separate decode from prefill/training.

Three implementations use the same D=4 slice:

1. Unchanged native Triton kernel: scalar route loops, global-memory state.
2. CUDA route-parallel global state: 256 threads, one `(route, dimension)` per
   thread. The full sequence remains a single ordered loop per partition/slice.
3. Same CUDA route-parallel arithmetic with explicitly indexed dynamic shared
   state: `state[slot*4 + local_dimension]`. One initial load and one final store.

Resident BF16 state is 32 KiB/CTA, plus 128 bytes of reduction scratch. FP32
diagnostics use 64 KiB, with explicit CUDA shared-memory opt-in. The global
variant clones initial/final cotangent state; the resident variant allocates
outputs and does the necessary input/output movement inside the kernel. Both
retain the original BF16 selected-row histories and FP32 gradient accumulation
buffers. Backward includes their initialization, state cotangent propagation,
cross-D atomics, and casts back to input dtypes.

Route sums use eight-route warp shuffle reductions followed by eight warp
partials. This changes FP32 reduction order and must pass differential checks.
Every token's state store remains BF16. Reverse read contributions are stored
and rounded before the write VJP. Barriers separate dependent stages. There is
no unordered token scan, dense chunk transition, or new block-size search.

## Evidence protocol (frozen before authoritative measurement)

Commit implementation first. Authoritative commands reject dirty/untracked code;
result files may be generated and committed afterward. New filenames preserve
authority-v2, triangular-numerical-v1, and all older negative artifacts.

Correctness compares native and both prototypes against the unchanged per-token
oracle. Pinned upstream remains a separate report, including the nonoverlapping
BF16 stress intervals (oracle 1.0, upstream approximately 0.875). All six VJPs
and joint readings/final-memory losses are checked. Unit tests additionally use
unnormalized cotangents to prevent tiny averaged gradients hiding a missing VJP,
test zero selected write weights, the reverse read cast, and unused cotangents.
Read-only and in-place decode retain their original implementation/regressions.

The persistent-state audit runs on all layers after all four accumulation
microbatches in five actual optimizer correctness steps, paired across backends
and modes. It reports finiteness, maximum/RMS error, violating elements and worst
coordinates with paired values. Its explicit **additional diagnostic** envelope
is `abs(candidate-reference) <= .02 + .02*abs(reference)`, copied from the frozen
state-forward BF16 check. It never replaces or changes the model gate. Original
GPU-computed checksum means and their `2e-6` gate remain separate; disagreement
with the elementwise diagnostic is retained.

The baseline runner executes the unchanged full model for all three seeds and
both modes, with the frozen five correctness / ten warmup / twenty measured
steps, four accumulation microbatches, and matched parameter, AdamW, batch and
persistent-state hashes. Internal gradients remain separate eager replays.
No candidate is installed in the model. SDPA is not rerun or mixed into the
Sparse Memory comparisons; its previous architectural control stays separate.

Baseline profiles observe each mode's actual CUDA state kernels in separate
passes. Conservative projections credit **only those kernel durations** as
replaceable, but charge candidates their **complete forward+backward stage**,
including copies, history traffic, zeroing, casts, allocation and orchestration.
Thus baseline nonkernel overhead is not optimistically removed. Each mode uses
its own measured native/upstream ratio, fraction and stage cost. Projections use
`R_baseline*((1-f)+f/s)` and remain screening evidence, never model acceptance.

Commands, from `projects/urm` (the upstream checkout must be clean and pinned):

```bash
export PYTHONPATH=src:benchmarks:/tmp/urm-sdm-upstream.wKxbSo
export LIBRARY_PATH=/opt/conda/lib
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest tests/test_route_parallel_experiment.py
python benchmarks/route_parallel_experiment.py correctness \
  --output results/pretraining-step/route-parallel-correctness-v1.json
python benchmarks/route_parallel_experiment.py resources \
  --output results/pretraining-step/route-parallel-resources-v1.json
python benchmarks/route_parallel_experiment.py performance \
  --correctness results/pretraining-step/route-parallel-correctness-v1.json \
  --output results/pretraining-step/route-parallel-performance-v1.json
python benchmarks/route_parallel_baseline.py \
  --output results/pretraining-step/route-parallel-model-baseline-v1.json
```

Run GPU timing and profiling serially in separate processes. Do not collect
hardware counters or launch audits during authoritative timing. Resource ceilings
from CUDA/compiler metadata are not measured occupancy; generated code and any
available hardware counters are separate evidence. Single-token residency is
not assumed beneficial and no decode change is part of this experiment.
