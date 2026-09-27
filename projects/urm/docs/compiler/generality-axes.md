# Generality axes (mechanically referenced by tests)

The catalog's 13 generality axes — the typed variation dimensions along which mixer
laws differ. Admission status mirrors `src/urm/backends/` and
`benchmarks/architecture-coverage.json`; the current native/references inventory is
in [../backends.md](../backends.md), the per-row mapping in [../catalog.md](../catalog.md).

| Axis | Typed dimension | Admission status at HEAD |
|---|---|---|
| **A1: slot/channel** | Logical slot, key-channel and value domains; gate broadcast; per-domain state layout | Admitted — the K1/K2 descriptors carry per-domain state layout (GLA/KDA/Rodimus clients) |
| **A2: locality / indexed K1** | Exact or approximate metric, route ties, capacity, geometric metadata; the indexed gather-attend schedule | Admitted — `K1Descriptor.indexed` + the `gather_indices` operand; native `triton/k1/indexed.py` (relaxed-atomic backward). Clients: NSA, Longformer, MoBA, DSA, Sparse Transformer |
| **A3: coordinated routing** | Assignment, ownership, collision/merge, deterministic tie and return protocol | Reference tier — MoM/external compositions |
| **A4: hierarchical chunks** | Architectural pooling versus schedule subdivision, level identity and boundary state | Admitted — the typed DyadicBankedState op (decay-forward carry cascade) with the native `triton/k2/dyadic_banks.py` branch. Clients: log-linear, LogLinearMamba2 |
| **A5: timescale banks** | Independent state instances, decay schedules and explicit weighted combination | Admitted — multi-state K2 (RetNet) with both state VJPs |
| **A6: slot interaction graph** | Edge domain, ordered propagation and state effects | External — graph-memory candidates |
| **A7: adaptive cardinality** | Allocation, birth/death, identity, reset and ragged cache ABI | External — CAT/dynamic memory |
| **A8: read/write coupling** | Independent factors, low-rank rank, augmented states, input-precomputable versus state-dependent coefficients | Admitted — generalized rank-1 K2 transition (`LinearDeltaSpec` low_rank / num_deltas); native `triton/k2/matrix_scan.py`. Clients: GDN2, RWKV-7/DPLR, IPLR, Comba, Gated DeltaProduct, Mamba-2 |
| **A9: complex/block-real state** | Rotation representation, conjugate rules, phase state and VJP | External — phase-bearing SSMs |
| **A10: stochastic routes** | RNG state, replay, estimator and distribution | External — sampled routing |
| **A11: parameter/expert/depth** | Static parameter domain, depth dependencies, grouped GEMM/dispatch and gradient accumulation | Admitted at frontend granularity — PAttention (block) and AttnRes (residual) rows; the parameter-domain K1 `map_normalize` reducer is reference-tier |
| **A12: inner optimization** | Closed loss/update/optimizer state, loop bounds, checkpoint and outer-gradient policy | External — Titans/TTT |
| **A13: score/reduction algebra** | Score map, normalization, neutral element, all-masked behavior, sum/LSE/max/positive or iterative reduction and VJP | Admitted — `K1ScoreLaw` (DOT, CHANNEL_DECAY) and `K1ReducerLaw` (SOFTMAX, THRESHOLD_RELU_POWER, SQUARED_SUM, MAP_NORMALIZE); native kernels `triton/k1/{channel_decay,threshold_relu,squared_sum,map_normalize}.py`. Clients: FoX, Wall, TDA, KATA, PAttention |

The taxonomy is closed under the four read-domains (charter invariant 2); the axes
are typed descriptor fields, never architecture-name dispatch.
