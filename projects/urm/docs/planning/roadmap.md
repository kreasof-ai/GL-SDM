# URM construction roadmap

**Target:** a small semantic core that compiles legal combinations of K1/K2/K3 regions, plus external model modules that compose those calls and ordinary operators. The [compiler charter](../compiler/compiler-charter.md) and [76-row composition ledger](architecture-composition.md) are binding. This roadmap is an ordered set of completion gates, **not** a claim that any listed future capability works now. No transitional architecture dispatch, callback escape hatch, or temporary source-named backend branch is admitted.

## Current baseline (HEAD `4cb35c5`)

- The public graph catalog has 14 dense-softmax K1 fragments and one route-to-state K3 fragment. K2 anchors can be selected but the graph binder cannot execute K2. No complete source-model module is qualified through the public graph path.
- Core still has two capability-selection stories, literal `query`/`key` name matching, omitted JSON semantic fields, runtime equation/dispatch logic, source-comparator policy leakage, and incomplete NumPy/Torch family symmetry. K3 route provenance and reference read timing need repair.
- `architectures/` and `inference/` are marker packages. `train/loop.py` remains tied to an SDM comparison model. The old 62-recipe master table and 76 kernel-slice comparisons are historical; neither certifies current model coverage. See [evidence](../validation/evidence.md).

## Gate 0 — Semantic truth before breadth

**Files:** `recipes/schema/graph-recipe.schema.json`, `src/urm/ir/`, `src/urm/compiler/normalize/`, `src/urm/compiler/pipeline.py`, `src/urm/frontend/recipes.py`.

1. Replace opaque recurrence strings with closed state/transition/read descriptors; do the same for K1 score/selection/reducer and K3 route/update/collision/read timing. Parse every accepted JSON field, including K2 layout/update and K3 additive/sum, or reject it. Separate architecture metadata from equation fields.
2. Bind operand **roles**, not tensor names. Normalize a graph under alpha-renaming to the same descriptor; changing a semantic field changes its hash and behavior or returns a decline. Keep logical domains independent of device layout.
3. Move architecture registration out of `src/urm`. The `recipes/architectures/{samba,pattention}.json` placeholders must not claim `complete_model_graph` until external modules implement their source graphs. Kernel fragments remain external compositions of general call contracts.

**Close when:** mutation tests for every semantic field and alpha-renaming pass; arbitrary transition strings fail before provider selection; the frontend imports no model/comparator registry; no accepted node is silently dropped.

## Gate 1 — One provider contract and complete execution plans

**Files:** `src/urm/backends/interface.py`, `compiler/select/`, `compiler/partition/`, `compiler/schedule/`, `compiler/lower/`, `compiler/verify/`, `runtime/bind.py`, `runtime/state.py`.

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
