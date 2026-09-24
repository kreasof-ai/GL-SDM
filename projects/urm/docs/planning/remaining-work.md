# Remaining work — URM core refactor

Status: gap analysis against `refactor-list.md` after the relocation cutover, plus
progress on the follow-up "executor instructions." This document records what is
done, what remains, and one item that was previously misreported.

Suite state: 944 passed, 123 skipped, 0 failed (comparator sources provisioned at
pinned revisions under `/tmp/urm-comparator-pins`).

## Completed since the relocation cutover (step 1 vertical slice)

- **Graph recipe schema v2** (`recipes/schema/graph-recipe.schema.json`): closed,
  discriminated typed-operation documents (nodes/edges/state), rejecting unknown
  fields and references.
- **`compiler/normalize/graph.py`**: a graph document normalizes into a typed
  `SemanticProgram`; dangling edges and unknown operations are rejected.
- **Graph loader** (`load_graph_recipe_document/file` in `frontend/recipes.py`):
  validates the closed op vocabulary and structure before normalization.
- **`compiler/pipeline.compile_graph`**: compiles a typed program with semantic
  anchor selection restricted to a target tier (reference/library/native); returns
  a `runtime.bind.BoundGraphPlan`.
- **Semantic legality gate**: `ExecutionAnchor.semantic_contracts` + the
  equation-contract check in `make_selector`; Polar anchors decline plain softmax
  MHA. A `WeightedReduce` is attention only when it is dense + softmax +
  sequence→sequence over query/key operands; gather-fed reduces stay routed
  reduction.
- **`runtime/bind.BoundGraphPlan`**: plan-authority executor; walks plan steps in
  graph order, binds operands by name, fails on missing/tampered steps.
- **`recipes/kernels/mha.json`** migrated to a graph document. Regression gates in
  `tests/test_graph_vertical_slice.py` prove JSON authority (causal toggle changes
  IR and output), float64 NumPy reference parity, incompatible-anchor decline, and
  plan-binding failure on tampering.

## Architecture-specific content moved out of the compiler core

Review feedback confirmed: architecture-specific knowledge must leave the core.

- **`compiler/select/anchors.py`**: now holds only the 16 URM-owned/generic anchors
  (9 `urm_native_*`/`urm.unified.*` + torch_linear/SDPA/routed-reduction/collective),
  6 URM constants, the generic selector machinery, the equation-contract legality
  gate, and a provider API (`register_anchor_provider`/`register_anchor_selector`).
- **`benchmarks/comparators/anchors.py`** (new): the 48 upstream/arch-named anchor
  constants + 57 declarations (FLA/ATMA/Mamba/XMA/Tucker/KATA/Longformer/BDH/H3/
  Hyena/TDA/FwPKM/FlashAttention/SDM), the SDM revision probe selector, and the
  sparse-state fallback selector; registered via `executors.register_all()`.
- `default_registry()` = URM-owned sparse selectors + core anchors + registered
  consumer providers. Suite: 941 passed, 0 failed.

Remaining arch-specific residue in core: ~80 string literals in `pipeline.py`'s
legacy `compile_mixer` recipe-name→anchor dispatch (deleted in step 3), the Python
catalog in `frontend/recipes.py` (deleted once the graph path covers the catalog),
and the `MixerBackend`/`RecurrentAlgorithm`/`EXTERNAL_OPAQUE` semantics in
`ir/graph.py` + `frontend/spec.py`.

## Step 2 done: legacy registry deleted

- `runtime/registry.py` (the legacy `MixerSpec`-dispatched `BackendRegistry`),
  its root/runtime re-exports, and the `NumpyBackend` adapter are deleted.
- The compiler `CapabilityRegistry` is the only selection API; the layout tests
  assert the public plan-binder surface (`BoundGraphPlan`). Suite: 941 passed.

## Environment / provisioning state

- GPU: NVIDIA A10G (23 GB), torch 2.14, triton 3.8, FLA 0.5.2 (pip).
- All 24 comparator sources provisioned at pinned revisions under
  `/tmp/urm-comparator-pins`; CUDA toolchain + flash_attn SDPA shim set up.
- The mamba1 `selective_scan_cuda` extension failed to build against torch 2.14
  (upstream `mamba @ e9594ce1` source); affects only the `mamba1_ssm_core`
  upstream column. Retry or pin-fix is open.
- `benchmarks/master_table.py` does not yet call
  `benchmarks.comparators.register_all()`; post-cutover it must, since external
  executors are consumer-registered. Probe also shows a `float != BFloat16`
  dtype bug on several K2 probe paths.

## Remaining (in the plan's order)

Step 1 (MHA vertical slice) and step 2 (legacy registry) are done. The remaining
core work, all required before the master table can be reproduced *through the
graph path*:

- **Recipe migration (bulk):** 73 spec-dump recipes remain on the v1 catalog.
  By family: 28 K1, 44 K2, 1 K3 (sparse_delta_memory). Only `mha` is a graph
  document. Each K2/K3 recipe needs its equation decomposed into typed nodes
  (transitions, scans, FFT convs, ordered state update/read) — this is steps
  3–4 and the largest single block of work.
- **Step 3 — multi-node graphs:** retire the special `SparseMemoryPlan` /
  `runtime/bind.py` SDM binder / `backends/triton/k3/memory.py` monolith in
  favor of a route→update→read typed graph through the common path.
- **Step 4 — per-family partition/schedule/cost/probes:** `partition/graph.py`,
  `partition/{k1,k2,k3}.py`, `schedule/{model,k1,k2,k3}.py`, `cost/{k1,k2,k3}.py`,
  `schedule/probes/triton_{k2,k3}.py`; per-region schedules; critical path by
  graph edges.
- **Semantic corrections:** remove `UnifiedMixerAccess`, `EXTERNAL_OPAQUE`,
  `MixerBackend`, `RecurrentAlgorithm` from core; `compile_mixer` recipe-name
  dispatch deleted once the graph path covers the catalog; `frontend/api.py`.
- **Step 5 — evidence closure:** demote one-node Samba/PAttention architecture
  JSONs to fragments; remove `urm::` registrations from
  `benchmarks/comparators/sdm/compiled.py`; relabel master-table "model-level"
  → "generic decoder integration".
- **Applications (ATMA-derived):** `train/{data,optimizer,loop,config}.py` +
  `architectures/urm_decoder/`; frozen SDM model → `benchmarks/models/`;
  `inference/{api,engine,scheduler,cache,model_runner,sampling}.py` around
  bound plans (borrow from `/tmp/urm-comparator-pins/atma` structure).
- **Verification:** migrate `benchmarks/master_table.py` to the public JSON →
  compile → bind path; run the 62-recipe sweep; regenerate
  `results/validation/master-table.json` + `docs/validation/master-table.md`.

## Corrections to earlier completion claims

- **`runtime/registry.py` survives.** The legacy `MixerSpec`-dispatched
  `BackendRegistry` is still present at its old import path, still re-exported by
  `runtime/__init__.py`, and still imported by `tests/test_backend.py` and
  `tests/test_project_layout.py`. The earlier claim that "0 of 92 old paths survive"
  was wrong; completion gate 1 of `refactor-list.md` is not met while this file
  exists.

## Gate-level gaps (end-state rules / required semantic corrections)

1. `EXTERNAL_OPAQUE` remains in `ir/graph.py` and is exercised by
   `recipes/kernels/bdh_attention_core.json`. Plan: remove it as a coverage
   shortcut; per end-state rule 4, BDH becomes explicitly unsupported.
2. `MixerBackend` remains in `ir/graph.py` with ~279 references across
   src/tests/benchmarks. Plan: remove backend identity from semantics.
3. `RecurrentAlgorithm` (MAMBA, GATED_DELTANET, KIMI_DELTA_ATTENTION, ...) remains
   the core descriptor of `frontend/spec.py`. Plan: replace architecture names and
   fixed model shortcuts with typed transition/route/state nodes.
4. `UnifiedMixerAccess` remains the compile target (7 references in `compiler/`).
   Plan's first required correction: compile a graph of typed operations, not an
   opaque spec inspected by a separate executor. This is the largest remaining item.
5. `benchmarks/comparators/sdm/compiled.py` still registers `urm::` production
   namespaces (two `custom_op` registrations). Plan: remove them.

## Named-file splits not done

| Plan target | Current state |
|---|---|
| `frontend/api.py` | `compile` exposed via `frontend/__init__.__getattr__` instead |
| `compiler/normalize/` | empty package; builders still in `ir/program.py` |
| `ir/ops.py`, `ir/state.py` | missing; `ir/mixer.py` split incomplete |
| `compiler/schedule/model.py` | missing; `kernel_plan` split incomplete |
| `compiler/schedule/{k1,k2,k3}.py` | missing; per-family knobs not split from `space.py` |
| `compiler/select/{k1,k2,k3}.py` | missing; per-family capability envelopes inline |
| `runtime/operands.py` | missing; route/operand certification still in k3 launchers |
| `backends/triton/k2/convolution.py` | missing; FFT convolution not split from `inner_state.py` |

Also: `recipes/kernels/__init__.py` exists — a JSON data directory should not be a
Python package.

## Deeper required work not done

- Recipe-name dispatch is still the runtime path: `compiler/pipeline.py` calls
  `named_mixer_recipe` from the Python catalog; the 74 JSON kernel recipes exist but
  are not consumed by the compile path. Plan: JSON is authoritative; delete the
  Python catalog.
- `runtime/state.py` sessions construct from raw parameters, not the selected plan.
- `compiler/cost/model.py` covers K1 only; no K2/K3, route, memory, backward, or
  end-to-end critical-path estimates.
- Rewrite rules not expanded (two rules exist; the plan requires sparse-work
  elimination, fusion/locality, legal scan reassociation, and composed K3 routing).
- `benchmarks/master_table.py` still contains `RecipeMixer` and "model-level"
  labels; evidence reclassification to "generic decoder integration" is not done.
- `train/` contains only `loop.py`; the model/data/optimizer split,
  `benchmarks/accounting/`, and `architectures/urm_decoder/` do not exist.
- `inference/` is an empty package (no API/scheduler/cache/runner).
- `architectures/` is an empty package (no samba/hla/atma components).
- Full source-model validation for Samba/PAttention (source revision, exact graph,
  training gradients, decode continuity, measured shapes/devices, comparator
  identity) is not done; their rows must stay "generic decoder integration" until
  then.

## Decisions needed before proceeding

1. **Full graph decomposition** (remove `UnifiedMixerAccess`; normalize → per-op
   partition/select/schedule/lower over typed multi-op graphs): do it now, or land
   the small gate fixes first (registry deletion, `urm::` removal, `frontend/api.py`,
   normalize extraction)?
2. **Removing `EXTERNAL_OPAQUE` / `MixerBackend` / `RecurrentAlgorithm`** changes the
   public spec schema and makes BDH (and any recipe relying on opaque recurrence)
   explicitly unsupported per plan rule 4. Confirm that is intended.
3. **`inference/`, `architectures/`, full `train/`** are new applications, not
   relocations. Minimal working scaffolding against the public API, or full
   Atma-style implementations? The pinned ATMA checkout at `/tmp/opencode/atma-src`
   can serve as the structural reference.

## Suggested landing order

1. Small gate fixes: delete the legacy runtime registry (rewrite the two importing
   tests against the public API), remove `urm::` registrations, create
   `frontend/api.py`, populate `compiler/normalize/`, drop
   `recipes/kernels/__init__.py`.
2. JSON-as-source cutover: the compile path loads recipes from `recipes/`; delete
   the Python catalog.
3. Semantic corrections: `EXTERNAL_OPAQUE` → explicit unsupported; `MixerBackend` →
   capability facts; `RecurrentAlgorithm` → typed nodes; state sessions consume the
   selected plan.
4. Full graph decomposition (`UnifiedMixerAccess` removal) with the per-family
   select/schedule/model splits and the cost-model extension.
5. Consumer builds: `master_table.py` reclassification, `train/`, `inference/`,
   `architectures/`.

## Contested boundaries (objections to forcing the letter of the plan)

- Route/operand certification is currently backend-owned: the k3 launchers certify
  their own operand layouts. Moving it to `runtime/operands.py` must not break
  backend self-containment or the K3 GPU path; the split should extract the
  runtime-facing validation surface, not relocate the certificates blindly.
- `ir/ops.py` vs `ir/program.py` boundaries should follow actual usage and import
  direction, not the filename list alone.
