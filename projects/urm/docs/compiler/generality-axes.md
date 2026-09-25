# Generality axes

This is a **typed semantic backlog**, not a list of implemented kernels. A descriptor and independent references may be added before a native path. A new physical backend branch must pass the [charter's admission rule](compiler-charter.md#backend-branch-admission). The [architecture ledger](../planning/architecture-composition.md) maps all 76 mixer-relevant source rows to external calls and unresolved axes.

| Axis | Closed semantic fields required | Independent clients / decisive gate |
|---|---|---|
| **A1: slot/channel** | Logical slot, key-channel and value domains; gate broadcast; per-domain state layout | GLA/KDA/Rodimus; preserve gate gradients and state shape |
| **A2: locality** | Exact or approximate metric, route ties, capacity, geometric metadata | Foveal plus synthetic indexed K1; include route work and approximation quality separately |
| **A3: coordinated routing** | Assignment, ownership, collision/merge, deterministic tie and return protocol | MoM and multi-head synthetic graph; conserve tokens/state and cost packing |
| **A4: hierarchical chunks** | Architectural pooling versus schedule subdivisions, level identity and boundary state | Log-linear and CAT; no schedule-only substitution for model compression |
| **A5: timescale banks** | Independent state instances, decay schedules and explicit weighted combination | RetNet and multi-state synthetic graph; both state VJPs |
| **A6: slot interaction graph** | Edge domain, ordered propagation and state effects | Graph-memory candidates; derive graph update before admission |
| **A7: adaptive cardinality** | Allocation, birth/death, identity, reset and ragged cache ABI | CAT/dynamic memory; capacity and gradient policy |
| **A8: read/write coupling** | Independent factors, low-rank rank, augmented states, input-precomputable versus state-dependent coefficients | Generalized delta/RWKV-7/momentum; bound rank growth and prove VJP. **Admitted (partial):** the generalized rank-1 K2 transition (`LinearDeltaSpec` erase_gate/write_gate/predict_key/low_rank) and the ordered multi-delta (`num_deltas`) — verified clients GDN2 (027), RWKV-7/DPLR (042/032), IPLR (031), Comba (037), Gated DeltaProduct (029). External residual: Momentum DeltaNet (030, coupled two-state), Gated Oja (033, value-channel Oja correction) — single-client sub-laws |
| **A9: complex/block-real state** | Rotation representation, conjugate rules, phase state and VJP | Phase-bearing SSMs; exact block-real parity |
| **A10: stochastic routes** | RNG state, replay, estimator and distribution | Sampled routing; replay and estimator tests |
| **A11: parameter/expert/depth** | Static parameter domain, depth dependencies, grouped GEMM/dispatch and gradient accumulation | PAttention/AttnRes; complete replaced-layer parity |
| **A12: inner optimization** | Closed loss/update/optimizer state, loop bounds, checkpoint and outer-gradient policy | Titans/TTT; no callback or affine-scan assertion |
| **A13: score/reduction algebra** | Score map, normalization, neutral element, all-masked behavior, sum/LSE/max/positive or iterative reduction and VJP | POLAR/Foveal, TDA/KATA as distinct equations; physical reuse must be measured. **Admitted (partial):** `K1ScoreLaw` (DOT + CHANNEL_DECAY) and `K1ReducerLaw` (SOFTMAX + THRESHOLD_RELU_POWER + SQUARED_SUM) in the K1 descriptor — verified clients FoX (additive bias via `score_bias`), Wall (channel-decay), TDA (threshold), KATA (squared-sum). External residual: Parallax (multi-statistic), POLAR/Foveal (polar/null-sink) |
| **A14: cross-call composition** | Typed producer/consumer edges, output pairing, coefficients, state ownership and legal fusion | Differential/TDA/HLA; independent call reference and cost proof |

## Current execution boundary

At HEAD `4cb35c5`, 14 K1 dense-softmax graph fragments and one K3 route-to-state fragment are live. K2 has no graph executor. Several lower-level implementations and historical comparators explore portions of these axes, but their existence does not mean the descriptor, public binder, independent references, gradients, cache and native schedule are qualified. See [evidence](../validation/evidence.md) for claim levels.

For every axis, record: descriptor and serialization; rejected combinations; NumPy/Torch references; two independent client graphs; source equations and proof obligations; provider capability; native forward/backward/decode envelope; end-to-end cost. A family label or axis label alone grants none of those stages.
