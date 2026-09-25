# URM construction roadmap

**Target:** a small semantic core that compiles legal combinations of K1/K2/K3 regions, plus external model modules that compose those calls and ordinary operators. The [compiler charter](../compiler/compiler-charter.md) and [76-row composition ledger](architecture-composition.md) are binding. This roadmap is an ordered set of completion gates, **not** a claim that any listed future capability works now. No transitional architecture dispatch, callback escape hatch, or temporary source-named backend branch is admitted.

## Current baseline (with Batches 1, 2 and 3 landed)

- The public graph catalog has 14 dense-softmax K1 fragments, one route-to-state K3 fragment, and **two K2 `linear_delta_state` fragments (DeltaNet, GLA)** that now execute end-to-end through the public graph path on the reference tier.
- **Batch 1 landed (all 16 composition-now rows):** external model modules under `architectures/` for the head-map cluster (MHA/MQA/GQA, one parameterized module), BitAttention, Hopfield, Conformer, TPA, Tucker, CAT, SDM, Samba (attention branch), Pattention (parameter-domain, composed as typed matmuls + external normalizer), AttnRes (depth-domain — admitted `LogicalDomain.DEPTH` into the IR), MLA, H3 and Hyena (external FFT compositions, zero URM kernel calls).
- **Batch 2 landed (all 14 composition-k2-gated rows):** external model modules composing external projections with the typed K2 `linear_delta_state` graph — DeltaNet (exact U2.D), GLA (channel gate), Simple GLA + Lightning (head gate, data-dependent / static), RetNet (static per-head decay), Linear Attention (plain + normalized), Gated DeltaNet (U2.D + head decay), KDA (U2.D channel decay), Based + ReBased (normalized + external feature maps), LightNet + HGRN2 + Rodimus (channel-decay frontends), YOCO (self U2.A + cross U1.S over shared KV). Each has a `recipes/architectures/<name>.json` at `generic_model_integration` (validated) and a parity gate against the pinned fla op (GPU) under `tests/test_architectures_*.py`. Row 015 also corrected the K2 normalized law (scale inside the denominator, additive ε) across both the torch executor and the numpy oracle to match the pinned source.
- **Batch 3 landed (all 6 combinator rows):** the typed **Merge** ordinary operator admitted into the IR/compiler (axis A14 — `out = Σ_i c_i·x_i` with closed coefficients and per-term runtime scale operands; unfused by default, fusion a later proven rewrite), plus the 6 combinator modules — Differential (flagship Diff: two U1.S calls + λ-merge), HLA (flagship Nest: two dependent prefix accumulations + strict-causality correction, external recurrence), ABC + GSA (two-stage slot-summary: external stage-1 recurrence + typed K2 stage 2, with the recorded gate-axis finding that the K2 channel gate decays the key axis and does not cover stage 1), Raven (external top-k router + GSA), MoM (routed-expert: external top-k router + per-memory U2.D + scatter-add merge). Nest composes via producer/consumer edges (no new op).

All 30 composition rows + 6 combinator rows (36 total) are landed as validated external model modules. Residual blockers (cache/decode modes, router tie-policy identity for SDM, Mamba axis for Samba, compressed-cache equality for MLA, the U2 rewrite for H3/Hyena, the packed-varlen/per-memory-state forms for MoM, and the per-row external-stage variants recorded in each recipe) are recorded, not claimed.
- **Batch 0 landed:** the closed K1 descriptor (`K1Descriptor`: scale law, head map, causal, all-masked-row policy) and the closed K2 descriptor (`LinearDeltaSpec`: delta/additive, gate scope, read timing, scale rule, normalized) are typed in IR; `weighted_reduce` and `linear_delta_state` bind operands by **role, not name**; unknown JSON semantic fields are rejected at normalize; the retired string-set `CapabilityRegistry` (`select/registry.py`, `lower/plan.py`, `test_backend.py`) is removed, leaving the anchor/selector contract as the single provider story; the missing differentiable Torch K2 reference (`backends/reference/torch/k2.py`) is added and verified against the independent NumPy VJP oracle; and DeltaNet/GLA match the pinned FLA source for output **and** final state (~1e-7).
- **Provider contract unified:** every backend (reference Torch, native Triton, trusted SDPA library) for K1/K2/K3/K3-route sits behind the single `Provider` contract (`backends/provider.py`): structured `decline(request)` before execution, then `execute(request, role-bound operands)`. The request carries the closed family descriptor, intent mode and accumulation policy. `runtime/bind.py` is a pure dispatcher — no equation logic, no name lookup. The K1 softmax equation moved from the runtime into `backends/reference/torch/k1_attention.py`.
- Still open from the baseline: K3 route-provenance/read-timing hardening, the SDM probe removal, native K2/K3 Triton schedules admitted through the same contract (Gate 2 — the two native K2 anchor names currently execute the reference recurrence as a fail-closed placeholder, an explicit non-claim), and the external model modules (Gate 3). `architectures/` and `inference/` remain marker packages; `train/loop.py` is still SDM-tied. The old 62-recipe master table and 76 kernel-slice comparisons are historical; neither certifies current model coverage. See [evidence](../validation/evidence.md).

## Gate 0 — Semantic truth before breadth

**Files:** `recipes/schema/graph-recipe.schema.json`, `src/urm/ir/`, `src/urm/compiler/normalize/`, `src/urm/compiler/pipeline.py`, `src/urm/frontend/recipes.py`.

1. Replace opaque recurrence strings with closed state/transition/read descriptors; do the same for K1 score/selection/reducer and K3 route/update/collision/read timing. Parse every accepted JSON field, including K2 layout/update and K3 additive/sum, or reject it. Separate architecture metadata from equation fields.
2. Bind operand **roles**, not tensor names. Normalize a graph under alpha-renaming to the same descriptor; changing a semantic field changes its hash and behavior or returns a decline. Keep logical domains independent of device layout.
3. Move architecture registration out of `src/urm`. The `recipes/architectures/{samba,pattention}.json` placeholders must not claim `complete_model_graph` until external modules implement their source graphs. Kernel fragments remain external compositions of general call contracts.

**Close when:** mutation tests for every semantic field and alpha-renaming pass; arbitrary transition strings fail before provider selection; the frontend imports no model/comparator registry; no accepted node is silently dropped.

## Gate 1 — One provider contract and complete execution plans

**Files:** `src/urm/backends/interface.py`, `compiler/select/`, `compiler/partition/`, `compiler/schedule/`, `compiler/lower/`, `compiler/verify/`, `runtime/bind.py`, `runtime/certification.py`.

1. Unify `BackendRequest`/`CapabilityRegistry` and `ExecutionAnchor` around one immutable semantic request, capability decline and executable entry point. Remove name sets, first-match-by-kind, empty wildcard contracts, global provider mutation and SDM upstream override from core.
2. Partition the complete graph into effect-safe regions. Select one exact provider and schedule per region; serialize role bindings, placement, materialization, state ABI, mode, backward/decode and numerical envelope. Runtime invokes that plan without re-deciding.
3. Implement a public K2 graph step and mixed K1/K2/K3 execution. Preserve K3 certified-route provenance, validate external routes, and pass read timing to reference and native. Move per-call value checks and state lifecycle from backend launchers to runtime; compiler owns static launch selection.
4. Refuse unknown cost or unschedulable regions. Independent plan verification rejects missing, reordered, duplicated or altered steps.

**Close when:** K1, K2, K3 and a mixed graph run from serialized plans through reference providers; invalid shapes, states, route values, modes and tampered plans decline before a kernel runs; runtime has no attention equation or architecture branch.

## Gate 2 — Independent references and canonical native slices

**Files:** `src/urm/backends/reference/{numpy,torch}/`, `src/urm/backends/triton/{k1,k2,k3}/`, `tests/`.

1. NumPy and Torch implement the same K1/K2/K3 descriptor/operand/result ABI. NumPy remains an independent high-precision equation, Torch remains differentiable; both cover route production, state continuation and claimed gradients. Unsupported descriptors decline explicitly.
2. Qualify native dense K1 MHA/MQA/GQA including cache positions; canonical additive/delta K2 with chunk training/prefill and recurrent decode; K3 route generation plus sparse delta state with collision, overlap and state VJP. Recover any removed equation only as a pinned external fixture before considering a shared axis.
3. Run an exact public-path baseline against pinned source units. Keep native, library and reference tiers distinct. A proposed extra physical branch follows [backend admission](../compiler/compiler-charter.md#backend-branch-admission).

**Close when:** every qualified descriptor has independent reference, Torch/autograd and native mode evidence; all operand and final-state cotangents pass; the compiler can legally decline unsupported cross-products. Canonical K2 long-prefill/training performance is measured, not inferred from fast decode.

## Gate 3 — External model composition and applications

**Files:** `architectures/`, `recipes/architectures/`, `train/`, `inference/`, `benchmarks/models/`, `benchmarks/comparators/`.

1. Build complete external model modules with source configuration, parameter mapping, ordered layers, ordinary operators, public URM calls and state/cache edges. Start with a dense-attention decoder, a K2 decoder, an SDM K3 model and one mixed graph. Then implement **Samba** with its configuration-derived Mamba/attention/retention/GLA schedule (including the mode that runs Mamba then attention in one block) and **PAttention** as a model reparameterization, not an attention fragment.
2. Replace the SDM-specific training loop with a model-agnostic data/optimizer/accumulation/checkpoint runner. The same runner must train tiny K1/K2/K3/mixed modules, take two optimizer steps and resume with matching next batch/update.
3. Build a model loader, single-request prefill/repeated-decode path, sampling and typed state sessions; then variable-length batching, request isolation and cancellation. Source checkpoints/logits and teacher-forced continuation are the correctness oracle.

**Close when:** architecture modules import only the public frontend/runtime for URM work; one unchanged trainer and inference engine serve K1/K2/K3/mixed modules; source-model parity includes full layers, parameters, gradients and cache behavior. No source adapter is registered inside core.

## Gate 4 — Expand generality only through reusable axes

**Files:** [axis register](../compiler/generality-axes.md), `src/urm/ir/`, `compiler/rewrite/`, references, admitted backends, external model modules.

Work in portfolios that falsify the broad claim early:

| Portfolio | Required independent clients | Failure that narrows the claim |
|---|---|---|
| Indexed K1 | NSA/MoBA/Longformer or Sparse Transformer with distinct route patterns | Dense-mask execution or route cost erases sparse saving |
| Cross-call composition | Differential/TDA and two-call HLA | Coefficient/state gradients wrong or launch/materialization dominates |
| K2 transition breadth | Canonical delta/GLA plus low-rank RWKV-7/generalized delta | No bounded stable chunk/scan or training schedule |
| Stateful non-affine K2 | PGDN/MesaNet or Titans/TTT as independent update graphs | State-dependent solve/optimizer cannot use the assumed generic schedule |
| Indexed mutable K3 | SDM plus a truly independent sparse update graph | No reusable collision/write law or address cost dominates |
| Ordinary long convolution | H3 and Hyena | FFT/filters cannot be collapsed into the three mixer families without losing exactness or speed |

For each axis, transcribe source equations, define descriptor and negative cases, produce NumPy/Torch references, prove legal rewrites and VJPs, then test two independent clients before any new physical backend branch. If the second client or physical case fails, keep the model-specific implementation external. The 76-row ledger's `B[...]` entries remain blocked until those records exist.

## Gate 5 — Analytical solver, benchmarks and release evidence

**Files:** `compiler/cost/`, `compiler/solve/`, `compiler/verify/`, `benchmarks/`, `results/`, [evidence rules](../validation/evidence.md).

Cost features include useful and wasted arithmetic, route/index/gather/scatter work, SRAM/HBM movement, state carry, scan depth, backward, launches, workspace, placement/communication and numerical recomputation. The solver ranks legal bounded candidates; independent verification precedes execution. Record predicted versus measured cost and regret. Unknown costs cannot silently rank as zero.

Rebuild paired benchmarks through **public** model modules and plan steps. Report fragment, generic-decoder and faithful source-model levels separately for training, prefill and decode, including output/gradient/state parity, throughput, memory and confidence. The committed historical master table stays in `results/validation/master-table.json` with its old runner identity; it is never treated as current graph-path evidence. A row is released only with the closeout record specified in the [architecture ledger](architecture-composition.md#required-closeout-record-for-each-id).

## Completion definition

“Three families cover a source architecture” means its external model graph calls only admitted K1/K2/K3 semantics **for mixer work**, with ordinary operators and their cost visible, and passes source-model parity for the declared modes. “Good enough performance” additionally means its complete public path passes a predeclared paired workload gate. A merely representable graph, external library fallback, historical fragment comparison or near-parity on a different mode is a separate, weaker claim.
