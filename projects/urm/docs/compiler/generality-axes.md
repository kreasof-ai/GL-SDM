# Generality axes for production construction

These are the full-version IR extension tasks. The unified mixer compiler
implements bounded parts of several axes below; it does not close those axes or
establish named architecture coverage. The goal is shared typed semantics and
reusable lowerings, not a universal mega-kernel. See the
[source audit](../planning/unification-audit.md).

## Implemented

`compiler/unified_mixer.py` provides executable, serialized contracts for:

- shared and grouped Q/K/V heads in K1 and K2, plus scalar/head/channel gate
  broadcasting (part of A1);
- separate matrix and diagonal state layouts, ordered rank-R updates, and
  factored left/right state transitions (bounded A8 support);
- stable softmax attention and a query/key denominator for additive linear
  state (part of A13);
- ordered token updates and explicit within-token collision rejection for K3.

Timescale banks and weighted multi-state combination (A5), coordinated
multi-head/expert routing (A3), general score reductions, complex state, and
the remaining axes are still open. See the
[coverage table](unified-mixer.md#named-kernel-recipes) for the exact
architecture-kernel boundaries.

## Shared descriptor boundary

Extend existing IR with typed descriptors for logical axis/domain, state bundle,
transition algebra, score/normalizer/reduction, routing policy and update region.
Provider and tile choices remain schedule properties. A serialized descriptor
must reconstruct semantics; an arbitrary callback cannot substitute for it.

| Axis | Required contract / implementation | Coverage fixtures | Acceptance obligation | Wave |
|---|---|---|---|---|
| A1: slot/channel | Explicit slot, channel and value domains; gate broadcasting and shared/independent transitions | GLA/KDA candidates, channel-routed memory | Prove legal flattening or use separate solves; all gate gradients | 2-3 |
| A2: locality | Metric, exact/approximate selection, ties, capacity and geometric metadata | Foveal, nearest-neighbor memory | Exact oracle; approximation is a separate semantic with quality tests; include index construction | 3 |
| A3: coordinated routing | Multi-head assignment, ownership, collision merge and deterministic ties | MoM/MoE, exclusive slots | Assignment oracle, conservation, route gradients and coordination cost | 3 |
| A4: hierarchical chunks | Distinguish schedule subdivision from architectural pooling; explicit boundary maps | Long-context recurrence, compressed attention | Nested schedule preserves recurrence; model hierarchy needs a new reference; bound rank/memory growth | 3-4 |
| A5: timescale banks | Parallel states, decay schedules and explicit weighted combination | RetNet, multi-memory fixtures | Independent-bank oracle, combination gradients and state continuation | 2 |
| A6: slot interaction graph | Edge propagation, ordering and state effects | BDH/graph-memory candidates | Derive graph update and VJP; collision and edge-traffic budgets | 4 |
| A7: adaptive cardinality | Ragged-state ABI, allocation, birth/death, identity and reset policy | CAT/compression, dynamic memory | Cache migration, bounded capacity and discrete-event differentiation policy | 4 |
| A8: read/write coupling | Independent low-rank factors, augmented state, precomputable versus state-dependent coefficients | RWKV-7, generalized delta, momentum, preconditioning | Derive transition closure or rank growth; no scalar-beta substitution without proof | 3 |
| A9: complex/block-real state | Rotations, representations and conjugate/adjoint rules | Phase-bearing SSM candidates | Complex/block-real reference parity, phase state and gradients | 3 audit, 4 broader support |
| A10: stochastic routes | RNG state, distribution, replay and gradient estimator | Sampled-routing fixtures | Replay and estimator tests; distinguish estimator from exact discrete-selection derivative | 4 |
| A11: parameter/expert/depth | Static operands, pointwise gates, grouped matmul, dispatch/combine and depth dependencies | Pattention, SwiGLU, MoE, AttnRes | Full subgraph and parameter-gradient parity including every projection | 3 |
| A12: inner optimization | Typed loss/update region, loop bounds, optimizer state, checkpointing and outer-gradient policy | TTT, FwPKM, MesaNet audit, MAML/Reptile | Update-trace and outer-gradient tests; explicit loop or external anchor, no callback escape hatch | 3 audit, 4 implementation |
| A13: score/reduction algebra | Separate score map, normalization and sum/LSE/max or iterative aggregation | POLAR, normalized linear, Hopfield candidates | Neutral elements, stable masking and gradients; dedicated kernels when needed | 2 common, 4 broader support |

## Avoid false equivalences

A channel axis is not merely a shape change when it changes transition factors.
SwiGLU requires two input projections, a SiLU/product gate and an output projection;
one bare attention contraction is insufficient. Static parameter memory and mutable
sequence state have different effects. Depth reduction is not sequence recurrence.

Hierarchical schedules can optimize a fixed operation; hierarchical memories can
change the model. Approximate locality selection must not replace exact routing
under an unchanged contract. Inner optimizer state and differentiability cannot
be inferred from a function pointer. Shared semantics can require different
physical kernels to preserve performance.

## Deliverables per axis

1. Descriptor, shape/effect validation, serialization and structured declines.
2. Independent reference and differential/adjoint tests across state boundaries.
3. One named architecture fixture and a synthetic boundary case demonstrating
   reuse without an architecture-specific condition in the kernel.
4. Capability matching and an executable compiler plan.
5. Native, adapter and end-to-end parity evidence on applicable workloads.

Start with A1/A5 and ordinary composition, then A3/A8/A11. Graph and stochastic
extensions do not block the initial three vertical slices, but remain explicit
full-version work. Publish which axes and [named targets](../planning/coverage.md)
are qualified and which remain blocked. Do not replace qualification with a claim
that an interface could theoretically express the operation.
