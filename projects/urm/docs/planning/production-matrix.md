# Production replacement matrix

Status: frozen release envelope. This is the bounded set of mandatory workloads
URM-native kernels must qualify against competitive upstream implementations
before the next release. It is the acceptance envelope, not the expansion
backlog; the broader [coverage register](coverage.md) tracks the wider catalog.

The machine-readable matrix is
[`benchmarks/production-matrix.json`](../../benchmarks/production-matrix.json)
(validated against
[`production-matrix-schema.json`](../../benchmarks/production-matrix-schema.json)).
Cases, dtypes, tolerances, and performance budgets are frozen **before** tuning
candidates and are not relaxed after seeing results.

## Mandatory workloads

| Workload | Family | Operation | Native candidate | Comparator | Modes | Slowdown budget |
|---|---|---|---|---|---|---|
| `k1-mha` | K1 | normalized routed reduction | `urm_native_k1_online_softmax_v1` | FlashAttention `flash_attn_func` | train fwd/bwd, prefill, decode | ≤10% |
| `k1-gqa` | K1 | normalized routed reduction | `urm_native_k1_online_softmax_v1` | FlashAttention `flash_attn_func` | train fwd/bwd, prefill, decode | ≤10% |
| `k1-masked-variant` | K1 | normalized routed reduction | `urm_native_k1_online_softmax_v1` | FlashAttention masked/varlen path | train fwd/bwd, prefill | ≤15% |
| `k2-diagonal-recurrence` | K2 | structured recurrence | `urm_native_diagonal_recurrence_v1` | FLA `chunk_hgrn` | train fwd/bwd, prefill, decode | ≤10% |
| `k2-gated-delta-recurrence` | K2 | structured recurrence | **none — native generation gap** | FLA `chunk_gated_delta_rule` | train fwd/bwd, prefill, decode | ≤10% |
| `k3-sparse-state` | K3 | sparse state | `urm_native_sparse_state_mixer_v0` | SDM `gated_write_read` | train fwd/bwd, decode | ≤10% |

The native-candidate column is pinned to the compiler by
`test_production_matrix_native_status_matches_compiler`, so the envelope cannot
overclaim native generation coverage. The K2 diagonal recurrence candidate is
the native diagonal-SSM anchor (`hgrn_ssm_core`, `mamba1_ssm_core`). The K2
matrix-state gated-delta workload is **expressible and matches the FLA
comparator through the reference and library anchors, but has no native
generated candidate yet** — the native backend lowers only diagonal K2
recurrence today. Qualifying `k2-gated-delta-recurrence` requires a native
matrix-state lowering; that is a measured blocker, not a qualified capability.
See [representation coverage](../validation/representation-coverage.md) for the
evidence separating representational coverage from native generation.

The case grid spans small latency-sensitive and larger throughput-oriented
batches, short/medium/long sequences, representative head/value/state widths,
training/prefill/stateful decode, nonzero initial states with streaming
continuation, and — for K3 — sparse route density, imbalance, and ordered
collisions. It is a representative envelope, not an arbitrary Cartesian product.

## Comparator and measurement policy

- Each workload freezes a **competitive** upstream comparator (repository,
  revision, exact callable) before independent confirmation. A slow reference
  implementation is never the sole performance baseline when a competitive
  compatible kernel exists.
- Comparisons include routing, layout conversion, allocation, dispatch, forward
  and backward, and cache/state updates. Exclusions are declared per case;
  neither side omits work the other requires.
- Kernel/device execution, ordinary dispatch-inclusive invocation, and CUDA
  graph replay are reported **separately**. Graph replay never substitutes for
  ordinary-invocation qualification; isolated kernel results never substitute
  for end-to-end replacement evidence.
- Correctness is verified against both the upstream callable and an independent
  oracle (outputs, required input/parameter gradients, final state and state
  gradients, streaming continuation, masking/collisions/boundaries) **before**
  performance qualification. A numerical failure disqualifies regardless of speed.

## Verdicts

Each workload/mode resolves to one of `qualified`, `correct_below_target`,
`numeric_failed`, `inconclusive`, or `unsupported`. A workload qualifies only
when every mandatory correctness check passes and the slowdown confidence bound
meets its frozen budget. Aggregate averages cannot hide a mandatory-case failure.

## Held-out generation probes

To demonstrate that URM compiles mathematical operations rather than recognizing
architecture names, the frozen compiler must natively execute held-out equations
composed from reusable capabilities, with no new architecture branch or
handwritten kernel body:

- **Differential attention** (K1): two softmax reductions combined by a learned
  per-head scalar — a straightforward composition.
- **Gated-delta with forgetting** (K2): a gated-delta matrix-state recurrence
  composed with a per-head forgetting gate — a demanding stateful composition
  within the declared supported language.

Arbitrary nonlinear recurrences are not claimed to admit efficient parallel scans.

## Current qualification state

None of the mandatory workloads is yet qualified for production training,
prefill, and decode. Native generation coverage today is narrower than the
envelope:

- **K1** has a native online-softmax candidate covering MHA/MQA/GQA and several
  masked/sparse variants, validated on a BF16 prefill kernel slice.
- **K2** has a native candidate only for **diagonal** recurrence (HGRN /
  Mamba-1 class). The matrix-state gated-delta workload is a native generation
  gap.
- **K3** has a native sparse-state candidate for Sparse Delta Memory.

The [upstream comparison table](../validation/upstream-comparison.md) records
kernel-slice parity and dispatch overhead for the wider catalog; those are
kernel-slice results, not production qualification. This matrix defines what
must be true for that to change, and its native-status column is kept honest by
a compiler-backed test.
