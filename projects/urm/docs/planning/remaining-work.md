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

## Transitional apparatus retired (hard cut)

The transitional machinery was removed outright rather than carried through
the migration:

- **`compiler/mixer.py` deleted** (the whole legacy family-dispatch path:
  `compile_mixer`, `CompiledMixerPlan`, `mixer_semantic_program`,
  `compile_frontend_mixer`, the external-executor registry, and the ~30
  family-specific reference/native executors). `compiler/pipeline.py` is the
  pure planner + graph path (`compile_graph`/`_GRAPH_TARGETS`).
- **`compiler/schedule/probes/` deleted** (all three Triton probes plus the
  probe machinery in `schedule/search.py`: `CompileProbe`, `CompileContext`,
  `CompileProbeResult`, `KernelResourceUsage`, `CompileStatus`, probe feedback
  nogoods, and resource fields on `ScheduleDecision`). The schedule search is
  now solve→verify→retry (analytical only); exact-specialization verification
  is designed into the per-family schedule stages when they land (step 4),
  not revived as a bolt-on probe directory.
- **The legacy spec vocabulary deleted**: `ir/graph.py` (`UnifiedMixerSpec` +
  family enums), `ir/k1.py`, `ir/k2.py` (spec validators), `frontend/spec.py`
  (`MixerSpec` + frontend enums), `ir/program.py`'s `UnifiedMixerAccess` op,
  the NumPy canonical core (`backends/reference/numpy/{graph,k1,k1_routed}.py`),
  and the legacy K1 launcher (`backends/triton/k1/launcher.py`).
- **`frontend/recipes.py`** is the v2 graph + architecture JSON loader only;
  the 59 schema-v1 spec-dump JSONs are deleted (the catalog is exactly the 15
  schema-v2 graph documents; the remaining recipes are re-authored as typed
  graphs from the pinned upstream sources during the K2/K1-variant migration).
- **Consumers**: the legacy test files and benchmark harnesses were deleted,
  not migrated (`test_unified_mixer.py`, the per-family `unified_mixer_*.py`
  benchmarks, `master_table.py`, `representation_coverage.py`,
  `native_coverage_table.py`, `comparators/executors.py`, the probe-driven
  epilogue/compilation-matrix harnesses and their artifacts). The all-recipe
  forward/backward sweep now runs through the graph path in
  `tests/test_graph_vertical_slice.py`. The architecture register
  (`benchmarks/architecture-coverage.json`) records 17 architectures with a
  live prototype through the graph path and 59 `pending_graph_migration`.

**Consequence**: the recipe catalog currently compiles 15 recipes (14 K1 +
1 K3) through the public graph path. The master-table driver is rebuilt
graph-native as the K2 bulk migration lands (the acceptance artifact remains
the 62-recipe reproduction). Suite: 421 passed, 0 failed; core imports without
torch.

## Backend hygiene: source-named kernels and the fragile fast-launch removed

- **Source-named K2 kernels deleted from core**: `triton/k2/nonlinear.py`
  (RWKV4/6, Mamba2 structured SSM, trapezoidal SSM, Hyena/H3 FFT convolution),
  `triton/k2/inner_state.py` (TTT/Titans-style inner-state, momentum, Oja,
  slot-attention), `triton/k2/backward.py` (their torch-autograd recomputation
  wrappers), `triton/k2/second_order.py` (HLA second-order cumsum), and the
  `numpy/k2_operators.py` canonical oracle that served them. These were
  forward-only single-architecture kernels with zero remaining consumers after
  the legacy path was deleted, and they fail the backend branch admission rule
  (no typed axis, no second independent client). The retained K2 native bodies
  are `matrix.py` (canonical delta-law) and `diagonal.py` (structured SSM) —
  the reusable axes consumed by `runtime/state.py` decode sessions and the
  formulation tests. The deleted equations return only as typed generality
  axes with two independent clients (see
  `docs/planning/backend-unification.md#admission-rule-for-a-backend-branch`).
- **`triton/k1/selected.py` deleted** (the K1 selected-logit softmax/value
  kernel that served the deleted NSA/DSA-style recipes; zero consumers).
- **The exception-driven fast-launch removed from `triton/k1/routed_reduce.py`**:
  the Triton private-API probe (`triton.knobs.runtime`, `jit_fn.warmup`,
  `CompiledKernel.__getitem__`), the module-global `_FAST_LAUNCH_CAPABLE` flipped
  by a runtime `except (AttributeError, TypeError, KeyError)`, and the
  `_DIRECT_LAUNCH_CACHE` runner cache are gone. Kernels now dispatch only through
  the standard `jit_fn[grid](...)` entry point (identical semantics, no
  version-fragile private API). This was the one true "exception fallback" in
  the backends; the remaining `support_status`/`SupportStatus.no` call sites
  are explicit capability declines, not fallbacks.

## Step 3 done: K3 graph path + SDM composite retired

The `sparse_delta_memory` recipe is a v2 graph document (`sparse_route_generation`
→ `sparse_state_mixer`) compiled by the ordinary stages and executed by
`BoundGraphPlan`. **The SDM-shaped composite is deleted from the core:**
`SDMExecutionMode`, `SparseMemoryMixerSpec` (with its frozen-facebook-adapter
`__post_init__` limits), `SparseDeltaMemorySpec`/`SparseMemoryAccess`/
`SparseDeltaMemoryAccess`, `sparse_delta_memory_program`, `partition/k3.py`,
`runtime/bind.py`'s `compile_sparse_memory_plan`, `backends/triton/k3/memory.py`'s
monolith, the `SPARSE_DELTA_MEMORY` anchor kind, and the SDM e2e selectors. Consumers
migrated: `train/loop.py`'s `SparseMemoryMixer` binds a config-built K3 graph through
`compile_graph` (fullgraph training stays traceable via the trusted route bridge);
the K3 GPU differential suite runs on the narrow route+state launchers. Suite: 919
passed, 0 failed.

Remaining SDM-named residue in core is benign: the recipe *name*
`sparse_delta_memory`, the reference anchor `urm.unified.k3.sparse_delta_reference.v1`,
the generic K3 fallback's capability probe (comparator-injected, no SDM import in
core), legacy-dispatch string literals pending catalog deletion, and docstring
provenance notes.

## Step 3 (first half) done: K3 recipe is a graph document

The `sparse_delta_memory` recipe is now a v2 graph document composing
`sparse_route_generation` → `sparse_state_mixer`, compiled by the ordinary
partition/select/lower stages and executed by `BoundGraphPlan` — no special
`SparseMemoryPlan`, no monolithic score-to-state backend. Gates: the graph compiles
to the two native route anchors + the native state anchor and executes on GPU; the
native state mixer matches the independent reference exactly (0.00 max err).
Suite: 943 passed.

This unblocks retiring the SDM composite (next): `compiler/partition/k3.py`,
`runtime/bind.py`'s `compile_sparse_memory_plan`, `backends/triton/k3/memory.py`'s
monolith, and the `SparseMemoryMixerSpec`/`SDMExecutionMode`/`sparse_delta_memory_program`
composite in `ir/program.py`, whose provider constraints (perfect-square, div-by-8,
frozen-subset widths, training length) must move to backend capability facts.

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

The arch-specific residue this section tracked is gone: the legacy
`compile_mixer` dispatch, the Python catalog, and the
`MixerBackend`/`RecurrentAlgorithm`/`EXTERNAL_OPAQUE` vocabulary were all
deleted in the transitional-apparatus cut (see above).

## Step 2 done: legacy registry deleted

- `runtime/registry.py` (the legacy `MixerSpec`-dispatched `BackendRegistry`),
  its root/runtime re-exports, and the `NumpyBackend` adapter are deleted.
- The compiler `CapabilityRegistry` is the only selection API; the layout tests
  assert the public plan-binder surface (`BoundGraphPlan`). Suite: 941 passed.

## Remaining (current state after the transitional-apparatus cut)

Steps 1–3 are done (MHA vertical slice, legacy registry deleted, K3 graph path +
SDM composite retired), and the transitional apparatus is gone (legacy family
path, probes, spec vocabulary, v1 catalog). What remains, in the plan's order:

- **Recipe migration (bulk):** the catalog compiles 15 recipes (14 K1 + 1 K3)
  through the graph path. The remaining families are re-authored as typed graph
  documents from the pinned upstream sources: 44 K2 equations (transition/scan
  graphs + per-family schedule/cost) and the K1 variant equations. Largest
  single block of work.
- **Step 4 — per-family partition/schedule/cost:** `partition/graph.py`,
  `partition/{k1,k2,k3}.py`, `schedule/{model,k1,k2,k3}.py`,
  `cost/{k1,k2,k3}.py`; per-region schedules; critical path by graph edges.
  Exact-specialization verification is designed into these stages (the
  transitional probe apparatus was removed, not extended).
- **Step 5 — evidence closure:** demote one-node Samba/PAttention architecture
  JSONs to fragments; remove `urm::` registrations from
  `benchmarks/comparators/sdm/compiled.py`; the master-table rebuild labels
  model-level rows "generic decoder integration".
- **Applications (ATMA-derived):** `train/{data,optimizer,loop,config}.py` +
  `architectures/urm_decoder/`; frozen SDM model → `benchmarks/models/`;
  `inference/{api,engine,scheduler,cache,model_runner,sampling}.py` around
  bound plans (borrow from `/tmp/urm-comparator-pins/atma` structure).
- **Verification (acceptance):** build the graph-native master-table driver;
  run the 62-recipe sweep (native + upstream columns); regenerate
  `results/validation/master-table.json` + `docs/validation/master-table.md`.

## Environment / provisioning state

- GPU: NVIDIA A10G (23 GB), torch 2.14, triton 3.8, FLA 0.5.2 (pip).
- All 24 comparator sources provisioned at pinned revisions under
  `/tmp/urm-comparator-pins`; CUDA toolchain + flash_attn SDPA shim set up.
- The mamba1 `selective_scan_cuda` extension failed to build against torch 2.14
  (upstream `mamba @ e9594ce1` source); affects only the future mamba1 upstream
  column. Retry or pin-fix is open.
