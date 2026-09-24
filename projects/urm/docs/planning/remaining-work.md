# Remaining work — URM core refactor

Status: gap analysis against `refactor-list.md` after the relocation cutover. This
document records what the cutover did **not** complete, including one item that was
previously misreported as done. It supersedes any summary claims of full completion.

Suite state at time of writing: 933 passed, 128 skipped, 0 failed (ATMA + SDM
provisioned at pinned revisions under `/tmp/opencode`). The items below are
deliberately open; none is blocked on missing information except where a scoping
decision is flagged.

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
