# Native SparseStateMixer v0 contract

**Status:** capability and benchmark grid frozen before performance tuning.

`SparseStateMixerAccess` is URM's route-to-state operation. It consumes
certified routes and never assumes how they were generated. Product-key
selection belongs to the existing composite `SparseMemoryAccess` front end and
is not part of the kernel-only native lowering.

## Frozen semantics and capability

State is logically contiguous `[partition, slot, value]`; route addresses are
partition-local `int32` or `int64` tensors `[P,T,K]`. Weights share the state
dtype and are finite, nonnegative, and normalized. Certification occurs once
outside timed dispatch and rejects out-of-bounds or non-increasing addresses,
so within-token write collisions cannot reach a kernel.

The v0 choices are:

- read-only/current-state, or decayed-delta state update;
- explicit pre-update or post-update reads for updates;
- ordered cross-token collisions and rejected within-token collisions;
- persistent inference state and one logical slot per page;
- FP32 accumulation with FP32 or BF16 stored state;
- contiguous CUDA tensors on SM80 or newer;
- `P<=16`, `T<=2048`, `S<=1,048,576`, `D<=1024`, and route widths `<=64`.

The bounds were declared before tuning. They limit compilation/resource growth,
not observed benchmark outcomes. Other dtypes, devices, layouts, merge rules,
page sizes, and dimensions receive structured declines. BF16 route
normalization uses absolute certification tolerance `0.004`, covering one
stored BF16 probability quantum for narrow Softmax rows.

For each partition/value tile, the update kernel traverses tokens sequentially:

```text
decayed[w,d] = exp(log_decay[t]) * state[address[w],d]
retrieved[d] = sum_w write_weight[t,w] * decayed[w,d]
delta[d] = beta[t] * (value[t,d] - retrieved[d])
state[address[w],d] = decayed[w,d] + write_weight[t,w] * delta[d]
reading[t,d] = sum_r read_weight[t,r] * state[read_address[t,r],d]
```

The read is moved before these equations for pre-update timing. This ordering is
structural and uses no atomics in forward. Training saves only selected rows;
its reverse ordered scan consumes both reading and final-state cotangents and
returns gradients for initial state, write weights, values, beta, log-decay,
and read weights. Relaxed atomics currently combine value-tile partials for
scalar/weight gradients, so deterministic-training requests decline.

## Frozen numerical gates

Native versus transparent PyTorch uses FP32 `atol=3e-5, rtol=3e-4` for backward
and BF16 `atol=0.03, rtol=0.03`; forward uses FP32 `2e-5/2e-5` and BF16
`0.02/0.02`. The pinned comparator changes association in chunked paths. Its
FP32/BF16 forward envelope is frozen at absolute `0.02` before tuning; the
worst observed pre-tuning collision-heavy training-state difference was
`0.010806`. Comparator backward uses the same backward gates. Tolerances may
not be relaxed based on later performance results.

The committed tests cover minimal routes, non-power-of-two dimensions,
collision-heavy recurrence, pre/post reads, persistent decode, read-only,
FP32/BF16 forward and backward, the pinned comparator, and a bounded FP64
gradcheck of the reference formulation. A guarded-import test executes the
native path without comparator packages.

## Frozen benchmark grid and acceptance

`benchmarks/sparse_state_mixer_cases.toml` is the pre-tuning grid. Kernel-only
comparison starts after identical certified routes exist. Pipeline comparison
is `not_applicable` until a native route-selection lowering exists; one side is
never charged for route work omitted from the other.

Substantial cases must satisfy the geometric-mean median, paired-bootstrap
upper-bound, p95, memory, three-process, randomized AB/BA, raw-sample, and drift
requirements in the master objective. Decode and the minimal read are
predeclared host-bound cases and are reported in microseconds with host-dispatch
share. The original repository remains only a pinned comparator/external
fallback. This native implementation imports and calls none of its code.
