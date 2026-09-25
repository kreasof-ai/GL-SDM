# URM construction roadmap

**Target:** a small semantic core that compiles legal combinations of K1/K2/K3 regions, plus external model modules that compose those calls and ordinary operators. The [compiler charter](../compiler/compiler-charter.md) and [76-row composition ledger](architecture-composition.md) are binding. This roadmap is an ordered set of completion gates, **not** a claim that any listed future capability works now. No transitional architecture dispatch, callback escape hatch, or temporary source-named backend branch is admitted.

For the architecture modules currently running their mixer outside the public URM path, follow the [architecture-to-URM integration instructions](architecture-urm-integration.md). They include the source-equation review, HLA two-K2 factorization, Mamba state-law split, K1 reducer work for PAttention, hierarchical state and PaTH score-transform candidates, provider capability correction, and per-claim acceptance checks.

## Current baseline (with Batches 1, 2, 3 and the Gate 4 axis admissions landed)

- The public graph catalog has 14 dense-softmax K1 fragments, one route-to-state K3 fragment, and **two K2 `linear_delta_state` fragments (DeltaNet, GLA)** that now execute end-to-end through the public graph path on the reference tier.
- **Batch 1 landed (all 16 composition-now rows):** external model modules under `architectures/` for the head-map cluster (MHA/MQA/GQA, one parameterized module), BitAttention, Hopfield, Conformer, TPA, Tucker, CAT, SDM, Samba (attention branch), Pattention (parameter-domain, currently composed with Torch matmuls and an external normalizer), AttnRes (depth-domain — admitted `LogicalDomain.DEPTH` into the IR), MLA, H3 and Hyena (external FFT compositions, zero URM kernel calls).
- **Batch 2 landed (all 14 composition-k2-gated rows):** external model modules composing external projections with the typed K2 `linear_delta_state` graph — DeltaNet (exact U2.D), GLA (channel gate), Simple GLA + Lightning (head gate, data-dependent / static), RetNet (static per-head decay), Linear Attention (plain + normalized), Gated DeltaNet (U2.D + head decay), KDA (U2.D channel decay), Based + ReBased (normalized + external feature maps), LightNet + HGRN2 + Rodimus (channel-decay frontends), YOCO (self U2.A + cross U1.S over shared KV). Each has a `recipes/architectures/<name>.json` at `generic_model_integration` (validated) and a parity gate against the pinned fla op (GPU) under `tests/test_architectures_*.py`. Row 015 also corrected the K2 normalized law (scale inside the denominator, additive ε) across both the torch executor and the numpy oracle to match the pinned source.
- **Batch 3 landed external combinator modules:** the typed **Merge** ordinary operator entered the IR/compiler (axis A14 — `out = Σ_i c_i·x_i` with closed coefficients and per-term runtime scale operands; unfused by default, fusion a later proven rewrite), alongside modules for Differential, HLA, ABC, GSA, Raven and MoM. HLA currently runs an external recurrence despite its exact two-K2 decomposition. MoM currently runs each selected memory over the full token stream, which changes the routed state law; its source-composition claim must remain open until corrected. See the [integration review](architecture-urm-integration.md).

The 30 composition rows and six combinator rows have external modules and scoped tests; that count is not 36 source-faithful compositions or complete models. Residual blockers include cache/decode modes, SDM router identity, Mamba configurations of Samba, MLA compressed-cache equality, the H3/Hyena U2 rewrite, MoM's routed-stream semantics, and per-row external stages.
- **Gate 4 partial progress:** typed IR/descriptor work has landed for Merge (A14), K1 score/reducer (A13), indexed K1 (A2), and generalized rank-1/multi-delta K2 transitions (part of A8). The selective diagonal-SSM law in `architectures/mamba.py`, the A4 hierarchical law in `architectures/log_linear_attention.py`, and the UT transforms in `architectures/{deltaformer,path_attention}.py` are external source-equation implementations; they are **not yet public typed graph operations**. Mamba-1/Mamba-2 currently share an external scan function but need different gate scopes; Log-linear/LogLinearMamba2 share one mixer with different frontends. Neither pair alone proves a broadly reusable physical branch. The 54 architecture recipe documents validate as metadata, not as 54 complete executable source models. See the [integration review](architecture-urm-integration.md) for the next admission gates.
- **Batch 0 landed:** the closed K1 descriptor (`K1Descriptor`: scale law, head map, causal, all-masked-row policy) and the closed K2 descriptor (`LinearDeltaSpec`: delta/additive, gate scope, read timing, scale rule, normalized) are typed in IR; `weighted_reduce` and `linear_delta_state` bind operands by **role, not name**; unknown JSON semantic fields are rejected at normalize; the retired string-set `CapabilityRegistry` (`select/registry.py`, `lower/plan.py`, `test_backend.py`) is removed, leaving the anchor/selector contract as the single provider story; the missing differentiable Torch K2 reference (`backends/reference/torch/k2.py`) is added and verified against the independent NumPy VJP oracle; and DeltaNet/GLA match the pinned FLA source for output **and** final state (~1e-7).
- **Provider contract unified:** every backend (reference Torch, native Triton, trusted SDPA library) for K1/K2/K3/K3-route sits behind the single `Provider` contract (`backends/provider.py`): structured `decline(request)` before execution, then `execute(request, role-bound operands)`. The request carries the closed family descriptor, intent mode and accumulation policy. `runtime/bind.py` is a pure dispatcher — no equation logic, no name lookup. The K1 softmax equation moved from the runtime into `backends/reference/torch/k1_attention.py`.
- Still open from the baseline: the SDM probe removal (deferred — marked as debt below) and the native K3 schedule. **Gate 2 native K2 is now honest:** the two native K2 anchors execute the fused Triton matrix-state scan (not the reference), and a K2 equation contract (`k2_canonical_linear_delta_v1` vs the normalized/elementwise/generalized-transition contracts) makes the native anchors decline the descriptor features whose kernel semantics diverge from the pinned law — verified for output, final state and cotangents across the canonical envelope. **Gate 2 K3 route-provenance/read-timing is now hardened:** the reference K3 provider validates route provenance (partition-local in-bounds, strictly-increasing/unique, the declared route width, finite nonnegative normalized weights) before the equation — reusing the `CertifiedSparseStateRoutes` value semantics — so malformed routes get a structured decline instead of a raw `IndexError` or silent acceptance; read timing (before/after-update) is verified honored end-to-end. `architectures/` now has external layer modules but not complete source-model applications; `inference/` and the SDM-tied `train/loop.py` still need Gate 3 work. The old 62-recipe master table and 76 kernel-slice comparisons are historical; neither certifies current model coverage. See [evidence](../validation/evidence.md).

> **Deferred debt — the SDM support probe.** `set_sdm_support_probe` / `_default_sdm_support_probe` in `compiler/select/anchors.py` is the injectable capability hook for the pinned-SDM K3 fallback (the comparator consumer installs a revision-aware selector through it). Removing it is a Gate 3 provisioning decision — it is load-bearing for the K3 fallback path (`benchmarks/comparators/anchors.py`), so it is deferred, not deleted, and recorded here as known debt rather than silently kept.

> **Gate 3 training harness (partial).** `train/` is now a model-agnostic ATMA-pattern harness: `data` (finewebedu shards + a synthetic stream), `model` (a pluggable-mixer `URMDecoderLM`), `registry` (the architecture→upstream map with honest capability flags — HLA has no reference kernel, TDA no decode kernel, both recorded), `optimizer` (AdamW+Muon with a coverage assertion), `harness` (the runner + the three correctness gates), and `run` (the CLI). It trains any registered public-path mixer; the **checkpoint-alignment, gradient-alignment, and KL-divergence gates pass** for the registered mixers (`dense_attention`, `gla`, `deltanet`, `forgetting_attention`), and loss decreases with the tuned AdamW+Muon rates. **MFU status (measured, honest):** the harness runs the native-tier mixer with a compiled bf16 surround (the mixer is an opaque graph boundary — the public-path plan executes eagerly, the surround fuses). On the A10G this reaches **~16.6% MFU** for the native K2 GLA model at ~100M params (eager native is 5.9%; the compile+bf16 surround is worth ~3×). **The 50% target is a large-GPU number, not reachable on the A10G**: a pure-torch, fully-compiled, bf16 dense transformer with no URM mixer — the absolute best case — caps at **~30–35% MFU** on this hardware (memory-bandwidth-bound at this batch/sequence), so the URM harness at 16.6% is within ~2× of the hardware ceiling and the residual is the per-layer graph break + the K2 backward, not a correctness gap. **Open:** closing the graph-break residual (a fused multi-layer mixer schedule) and the SDM K3 model + complete-model sources (Samba, PAttention) remain Gate 3 work. `train/loop.py` is retained for the SDM comparison evidence.

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
