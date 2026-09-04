# Native Sparse Memory E2E v0 contract

**Status:** compiler-visible native route plus state path implemented; the
three-process frozen-grid artifact is the completion authority.

The E2E boundary begins at write/read score tensors and ends at readings and
persistent final state. Model-specific projections are outside scope. The
logical program remains two typed operations:

1. `SparseRouteGeneration`: constrained score composition, top-k selection,
   ascending address canonicalization, and selected-score Softmax.
2. `SparseStateMixerAccess`: certified sparse read or ordered decayed-delta
   update/read with persistent state.

The compiler may select the composite `urm_native_sparse_memory_e2e_v0` anchor,
but the plan serializes both schedules and an explicit materialized-route
boundary. No upstream callable, physical layout, or implementation flag appears
in either semantic operation.

## Frozen native route envelope

The v0 route specialization accepts contiguous CUDA FP32 or BF16 factor-score
tensors `[P,T,2H]`, with `P<=16`, `T<=2048`, `H<=256`, source extent `H*H`, and
route width `1..64`. It uses additive pair composition, stable selected-set
top-k, ascending partition-local address order, and Softmax over selected
scores. Output addresses are `int32`; weights use the score/state dtype. Inputs
must be finite and the top-k selection boundary must be tie-free. Certification
occurs outside timed dispatch and seals tensor versions; mutation invalidates
the certificate.

The selected set is deterministic for tie-free inputs. Internal equal scores
may occur only when they do not change membership; canonical address sorting
then fixes route order. Other composition, selection, normalization, address,
index, device, dtype, or route-width choices decline structurally. Product-key
routing is this specialization's implementation target, not a universal
architectural assumption.

The native route backward differentiates selected Softmax weights back into both
score factors while treating discrete selected addresses as nondifferentiable.
The composite backward covers write scores, read scores, initial memory, values,
beta, and log-decay. FP32 and BF16 differential gates compare transparent
PyTorch, exact pinned upstream, and native paths, require exact addresses, and
reject non-finite gradients.

## Four-level measurement

`benchmarks/sparse_memory_e2e_cases.toml` freezes 19 read-only, persistent
decode, prefill, collision, maximum-width, capacity, and training cases across
FP32/BF16. `benchmarks/sparse_memory_e2e.py` reports:

1. transparent PyTorch semantic reference;
2. pinned original SDM E2E;
3. diagnostic upstream route production plus native state mixer;
4. fully native URM route production plus native state mixer.

Level 3 is never called native E2E. The authoritative comparison pairs levels 2
and 4 over the whole score-to-state pipeline. Independent stage passes report
route, state, remaining orchestration/materialization, and a state-only Amdahl
limit. Forward and backward are timed separately. First-use certification is
outside steady state because fixed immutable scores allow certificate reuse;
its construction cost is retained separately.

The frozen A10G comparator faults with a CUDA misaligned-address error for BF16
`D=257`. Native BF16 `D=257` remains boundary-tested, while overlapping E2E
non-power-of-two BF16 cases use `D=95`; FP32 training retains `D=257`. This is a
pre-performance comparator compatibility boundary, not a benchmark-driven
capability removal.

## Optimization record

- Accepted: row-owned factor top-k followed by pair top-k and canonical Softmax;
  exact routes and score gradients passed before full-grid timing.
- Accepted: BF16 pair-score storage rounding before FP32 Softmax, required to
  match the frozen score-storage semantics.
- Deferred: physical route/state fusion. The unfused explicit logical
  composition clears the E2E gate, so profiling does not justify coupling the
  two operations yet.

MFU/MBU are not reported because no meaningful sparse utilization denominator
was measured. Host-bound cases use absolute microseconds; substantial cases use
geometric-mean, hierarchical CI, p95, and memory gates. Completion requires
exactly three clean fresh processes and complete strict-schema artifacts.
