# URM core refactor list

Status: target design and complete disposition list for the 92 Python files currently under `src/urm/`. This is a refactor plan, not a claim that the target layout or coverage already works.

## End-state rules

1. `src/urm/` contains only the public mixer frontend, typed semantic IR, compiler, plan binder/runtime, three native mixer backend families (K1/K2/K3), and independent NumPy/PyTorch reference backends. No model architecture, named coverage catalog, upstream comparator, training loop, inference engine, benchmark, test fixture, or profiling harness lives there.
2. Every compiler pass has one owner and a subdirectory: `normalize/`, `rewrite/`, `partition/`, `select/`, `placement/`, `schedule/`, `lower/`, `verify/`, `cost/`, and `solve/`. Shared diagnostics may live in `compiler/common/`. `ir/` owns semantic types and effects; backends must not import compiler internals.
3. `backends/triton/k1`, `k2`, and `k3` contain GPU kernel bodies, autograd/backward implementations, narrow launchers, and declarative capability facts. Candidate choice, rewrite, probing policy, cost, tuning, partitioning, and route/state composition belong to compiler stages. Runtime binds and invokes **the compiled plan**; it does not redispatch from the recipe name or rederive a launch schedule.
4. Every named coverage recipe is a versioned JSON document outside `src/urm/`. A recipe may declare a kernel fragment or a complete model graph, but it must state which. JSON refers only to registered typed operations, parameter transforms, state/cache contracts, and graph edges. It cannot hide Python callbacks or use an architecture name as an execution switch. Model-specific components, when a typed reusable operation is insufficient, live under `architectures/` and are explicitly outside URM core. An unrepresentable component makes the full-model row unsupported rather than silently becoming generic attention.
5. `train/`, `inference/`, `architectures/`, `benchmarks/`, and `tests/` are project-level consumers of the public frontend and runtime APIs. Adopt Atma's separation of inference API/engine/scheduler/cache/model runner and training model/data/optimizer/loop, not Atma-specific model code inside URM.
6. No old-path compatibility modules, alias exports, dual registries, temporary name dispatch, or retained SDM-only plan/binder survive the cutover. Rewrite in-repository callers and documentation in the same change. Remove unsupported claims rather than keeping a fallback that bypasses the typed plan.

Proposed project tree (new `__init__.py` files in packages are implied):

```text
projects/urm/
  src/urm/
    frontend/                 # compile API, input schema/validation, JSON loader
    ir/                       # typed graph, K1/K2/K3 semantics, effects, state ABI
    compiler/
      common/ normalize/ rewrite/ partition/ select/ placement/
      schedule/ lower/ verify/ cost/ solve/
    runtime/                  # plan binding, operand validation, state sessions
    backends/
      interface.py            # pure capability and launcher contracts
      reference/numpy/        # independent float64/correctness equations
      reference/torch/        # differentiable reference equations
      triton/k1/ k2/ k3/      # native kernels and their narrow launchers
  recipes/
    kernels/*.json            # component/equation mappings and scope
    architectures/*.json     # complete graph, layer order, parameters, cache
    schema/                   # versioned recipe JSON schemas
  architectures/<name>/      # only model-specific components/loading/checkpoints
  train/                     # training application
  inference/                 # serving/generation application
  benchmarks/                # comparators, workloads, profiling, measurements
  tests/                     # fixtures, independent parity and integration tests
  results/                   # versioned evidence, not compiler inputs by default
```

## Required semantic and evidence corrections

- Compile a **graph of typed operations**, not an opaque `UnifiedMixerAccess` whose spec is later inspected by a separate executor. K1 covers routed/normalized reductions, K2 covers explicit state transitions and scans, and K3 covers selected state reads/writes. Route generation, parameter-axis transforms, projections, normalization, MLPs, and cache management are distinct graph nodes or external model components. A full model can invoke K1/K2/K3 multiple times; the three families are not three source files or three launches for a model.
- Remove architecture booleans, recipe-name branches, `EXTERNAL_OPAQUE` as a coverage shortcut, arbitrary `("...",)` tensor shapes, and candidate/anchor overrides that can select an incompatible equation. Semantic equivalence, input/output shape, dtype, state effects, training backward, decode state, and device support must be checked before an anchor is selectable.
- Make the serialized plan authoritative: each step names a typed operation, concrete backend capability, operands, layout, schedule, state transition, and backward/decode contract. Runtime executes those steps. A compiler result with no real schedule/cost for K1/K2/K3 is an explicit incomplete result, not a zero-cost choice.
- Extend analytical cost and schedule search to all three families. Account for useful versus wasted computation, route/index construction, gather/scatter traffic, SRAM/HBM movement, launch and synchronization costs, occupancy/register pressure, numerical/recompute cost, scan depth, state/cache traffic, backward, and graph critical path. Calibrate against measurements; retain uncertainty and rejected candidates. Probe exact compiled specializations only after legal analytical pruning.
- Separate evidence levels: semantic representation; independent reference parity; native kernel parity and performance; generic-model integration; **source-architecture** training/inference parity and performance. The current `master-table.md` and `benchmarks/master_table.py` put mixer fragments into one generic decoder, so their “model-level” label must become “generic decoder integration.” Those rows do not prove full Samba, PAttention, or other source models. Preserve measured numbers with corrected scope and provenance rather than rewriting them as stronger evidence.
- Samba needs a complete layer graph with its actual interleaving and state/cache branches. `samba_attention_core` may remain a K1 fragment JSON only, without full-Samba coverage credit. PAttention needs the parameter-token construction/reparameterization, parameter gradients, and model arrangement; its softmax contraction may remain a K1 fragment only. Apply the same boundary check to every other named recipe, especially hybrid, factorized, MoE, and stateful models.
- Reference backends must be independently implemented and reject unsupported semantics. Do not use the same optimized PyTorch path as both implementation and oracle. Keep float64 NumPy formulations, recurrence/slot VJPs, and source-specific comparator math where needed, but remove the separate `oracles` package.

## File-by-file disposition

Paths in the **Source** column are relative to `src/urm/`. Paths in **Target and required change** are relative to `projects/urm/`. “Delete” means no module remains at the old import path after its responsibility is moved or retired.

### Package root and upstream adapters (17 files)

| Source | Target and required change |
|---|---|
| `__init__.py` | Keep as a small public API for typed specs, `compile`, and plan binding. Remove oracle, legacy registry, and old routed-reduction reexports. |
| `adapters/__init__.py` | Delete with the package; no upstream comparator barrel in core. |
| `adapters/atma_gated_delta.py` | `benchmarks/comparators/atma_gated_delta.py`; keep pinned ATMA kernel identity and comparison, while a full Atma model belongs under `architectures/atma/`. |
| `adapters/bdh.py` | `benchmarks/comparators/bdh.py`; move source loader and comparison outside core. Describe any reusable BDH equation with generic IR operations. |
| `adapters/compiled_sparse_delta_memory.py` | `benchmarks/comparators/sdm/compiled.py`; comparator-only custom op/autograd wrapper. Remove its `urm::` production namespace registration. |
| `adapters/dense_attention.py` | `benchmarks/comparators/flash_attention.py`; upstream FlashAttention binding and pinning. Native K1 selects on semantics and capability, not this adapter. |
| `adapters/gated_delta_reference.py` | `benchmarks/comparators/fla_gated_delta_reference.py`; frozen FLA-specific reference. Put general gated-delta equations in `backends/reference/{numpy,torch}/k2.py`. |
| `adapters/gated_delta_rule.py` | `benchmarks/comparators/fla_gated_delta.py`; move FLA version checks and wrapper out of core. |
| `adapters/kata.py` | `benchmarks/comparators/kata.py`; preserve pinned source comparison, express its supported generic operation separately. |
| `adapters/longformer.py` | `benchmarks/comparators/longformer.py`; source sliding-chunk loader/comparison, with route/mask semantics declared in recipe JSON. |
| `adapters/samba.py` | `benchmarks/comparators/samba.py` for pinned attention comparison; full hybrid layer composition under `architectures/samba/` and `recipes/architectures/samba.json`. |
| `adapters/sparse_delta_memory.py` | `benchmarks/comparators/sdm/upstream.py`; move upstream checkout, identity, support probe, config, and execution. Core K3 capability does not import SDM. |
| `adapters/sparse_delta_memory_reference.py` | Split generic selected-slot math into `backends/reference/{numpy,torch}/k3.py`; keep upstream product-key and source-specific differential helpers in `benchmarks/comparators/sdm/reference.py`. Delete old module. |
| `adapters/sparse_state_mixer_external.py` | `benchmarks/comparators/sdm/sparse_state.py`; remove production external fallback. Core compiler declines when native K3 is unsupported. |
| `adapters/sparse_transformer.py` | `benchmarks/comparators/sparse_transformer.py`; pinned source loader and mask fixture outside core. |
| `adapters/tda.py` | `benchmarks/comparators/tda.py`; pinned source wrapper outside core. |
| `adapters/tucker.py` | `benchmarks/comparators/tucker.py`; source comparator outside core; factor production/contraction belongs to typed model graph or architecture component, not name dispatch. |

### Backend contracts, reference implementations, and Triton (30 files)

| Source | Target and required change |
|---|---|
| `backends/__init__.py` | Keep minimal backend package exports; no eager Torch/Triton imports or selection policy. |
| `backends/interface.py` | Keep pure launcher/capability descriptions here. Move `ExecutionPlan` to `compiler/lower/plan.py`; capability facts describe complete equations and modes. |
| `backends/numpy/__init__.py` | Replace with `backends/reference/numpy/__init__.py`; delete old path. |
| `backends/numpy/softmax.py` | `backends/reference/numpy/k1.py`; implement the reference backend directly over independent K1 equations, rather than forwarding to `oracles.routed`. |
| `backends/pytorch/__init__.py` | Replace with `backends/reference/torch/__init__.py`; delete old path. |
| `backends/pytorch/softmax/__init__.py` | Merge into `backends/reference/torch/__init__.py`; delete the extra package layer. |
| `backends/pytorch/softmax/routed_reduction.py` | `backends/reference/torch/k1.py`; keep transparent differentiable routed reduction and strict signature checks. |
| `backends/pytorch/sparse_state.py` | `backends/reference/torch/k3.py`; import K3 semantics from `ir/`, not `compiler.semantic`. |
| `backends/registry.py` | `compiler/select/registry.py`; selection, declines, and fallback policy are compiler decisions. Backends only publish immutable capability facts. |
| `backends/triton/__init__.py` | Keep lazy native package marker only. |
| `backends/triton/recurrence/__init__.py` | Replace with `backends/triton/k2/__init__.py`; delete old family alias. |
| `backends/triton/recurrence/backward.py` | `backends/triton/k2/backward.py`; retain only VJPs for typed K2 transitions. Move model-specific operand assembly out of core. |
| `backends/triton/recurrence/diagonal_recurrence.py` | `backends/triton/k2/diagonal.py`; keep scan/decode kernels and launch entrypoints, move schedule choice to `compiler/schedule/k2.py`. |
| `backends/triton/recurrence/hla.py` | General second-order scan kernel to `backends/triton/k2/second_order.py`; HLA-specific operand construction/comparison to `architectures/hla/` and `benchmarks/comparators/`. No HLA-named core anchor. |
| `backends/triton/recurrence/inner_state.py` | Split by typed K2 operator into `backends/triton/k2/inner_state.py` and `backends/triton/k2/convolution.py`; put operand planning and variant selection in compiler. FFT convolution must retain its own explicit operation/cost, not masquerade as a plain recurrence. |
| `backends/triton/recurrence/matrix_state.py` | `backends/triton/k2/matrix.py`; generic matrix-state scan and decode launchers, with schedule choice in compiler. |
| `backends/triton/recurrence/nonlinear.py` | `backends/triton/k2/nonlinear.py`; keep only explicitly typed transition kernels. Move architecture-specific RWKV/Mamba wrapper assumptions to model recipes/components. Mark inherently serial transitions as such. |
| `backends/triton/softmax/__init__.py` | Replace with `backends/triton/k1/__init__.py`; delete old family alias. |
| `backends/triton/softmax/online.py` | Split native normalized, masked, decode, positive-feature, and thresholded kernels across `backends/triton/k1/` implementation files; accept compiler-selected configs and typed operands. |
| `backends/triton/softmax/online_backend.py` | Low-level call adapter to `backends/triton/k1/launcher.py`; move semantic capability checks and candidate selection to `compiler/select/k1.py`. |
| `backends/triton/softmax/routed_reduce.py` | `backends/triton/k1/routed_reduce.py` for forward/backward kernels and explicit-config launcher. Move heuristic launch-parameter selection to `compiler/schedule/k1.py`. |
| `backends/triton/softmax/routed_reduction.py` | Execution wrapper to `backends/triton/k1/routed_launcher.py`; capability/selection to `compiler/select/k1.py`; unify with the typed K1 routed operation. |
| `backends/triton/softmax/routed_scale_epilogue.py` | Kernels and explicit launch path to `backends/triton/k1/row_scale.py`; schedule-point/config logic to `compiler/schedule/k1.py`, exact compile probe/resource collection to `compiler/schedule/probes/triton_k1.py`. |
| `backends/triton/softmax/selected.py` | `backends/triton/k1/selected.py`; selected-logit reduction kernel and autograd only. Routing policy is an IR/compiler concern. |
| `backends/triton/sparse_state/__init__.py` | Replace with `backends/triton/k3/__init__.py`; delete old family alias. |
| `backends/triton/sparse_state/backend.py` | Low-level state launcher to `backends/triton/k3/state_launcher.py`; capability envelope to `compiler/select/k3.py`; route/operand certification to `runtime/operands.py`; semantic types come from `ir/`. |
| `backends/triton/sparse_state/memory.py` | Delete the monolithic score-to-state backend. Route-plus-state composition becomes `compiler/partition/k3.py` and `compiler/lower/k3.py`; plan binding and state lifecycle become `runtime/bind.py`/`runtime/state.py`. |
| `backends/triton/sparse_state/mixer.py` | `backends/triton/k3/state.py`; retain read/update/backward Triton kernels and narrow launchers. Remove direct profiling-hook imports and backend-owned schedule decisions. |
| `backends/triton/sparse_state/route_backend.py` | Route launcher to `backends/triton/k3/route_launcher.py`; support decisions to `compiler/select/k3.py`; score/route certificates to `runtime/operands.py`. |
| `backends/triton/sparse_state/route_selection.py` | `backends/triton/k3/route.py`; retain route-selection kernels and VJP as an explicit K3 prerequisite operation, with schedule selected by compiler. |

### Compiler (21 files)

| Source | Target and required change |
|---|---|
| `compiler/__init__.py` | Rebuild as a small compiler API; remove old `unified_mixer` imports and broad compatibility exports. |
| `compiler/constraints.py` | `compiler/solve/constraints.py`; generic named constraint vocabulary, independent of any recipe. |
| `compiler/cost.py` | `compiler/cost/model.py`; split device profile loading into `compiler/cost/device.py`. Add K1/K2/K3, route, memory, backward, and end-to-end critical-path estimates with provenance/uncertainty. |
| `compiler/diagnostics.py` | `compiler/common/diagnostics.py`; stable error codes shared by stages, with stage and offending IR node. |
| `compiler/effects.py` | `ir/effects.py`; effects are semantic facts used by rewrites and verification, not a compiler-only vocabulary. |
| `compiler/execution.py` | Generic anchor/capability contracts and complete semantic matching to `compiler/select/anchors.py`. Remove SDM, sparse-memory, and library-specific selectors; lower to typed K1/K2/K3 subplans. |
| `compiler/kernel_plan.py` | Split candidate constraints into `compiler/select/model.py`, schedule constraints into `compiler/schedule/model.py`, assignment checking into `compiler/verify/assignments.py`; remove routed-epilogue-only assumptions. |
| `compiler/locality.py` | `compiler/placement/locality.py`; state operand/result residency and legal movement independent of architecture name. |
| `compiler/placement.py` | Mesh, binding, exchanges to `compiler/placement/plan.py`; physical `PlanStep` and executable serialization to `compiler/lower/plan.py`. |
| `compiler/placement_solver.py` | `compiler/placement/solver.py`; integrate only when the graph has explicit distributed route/state edges. Keep deterministic fallback as a verified candidate, not silent default. |
| `compiler/planner.py` | Split `UrmCompiler` orchestration into `compiler/pipeline.py`, candidate enumeration into `compiler/select/`, route distribution into `compiler/placement/`, schedule solving into `compiler/schedule/`, and plan/result types into `compiler/lower/plan.py`. Remove benchmark-entrypoint tickets from core. |
| `compiler/rewrite.py` | `compiler/rewrite/{rules,engine,proof}.py`; retain typed forward/backward and numerical obligations; expand rewrites for sparse work elimination, fusion/locality, legal scan reassociation, and composed K3 routing. |
| `compiler/route_protocols.py` | `compiler/placement/routes.py`; keep route conservation and communication contracts, selected only for explicit distributed graph edges. |
| `compiler/schedule_space.py` | `compiler/schedule/{space,k1,k2,k3}.py`; generic enumeration plus per-family implemented knobs. Move exhaustive model sweeps used only as independent test oracles to `tests/solver/`. |
| `compiler/search.py` | `compiler/schedule/search.py`; common legalize → cost-rank → exact-probe → verify → retry flow for all three families, not just routed reduction. |
| `compiler/semantic.py` | Split types/ops/program into `ir/{types,ops,program,k1,k2,k3}.py`; normalize builders into `compiler/normalize/`. Remove `UnifiedMixerAccess` opaque wrapping and SDM-specific semantic identities. |
| `compiler/solver.py` | `compiler/solve/z3.py`; optional backend for generic feasibility/optimization, always checked by independent verification. |
| `compiler/sparse_memory_plan.py` | Delete. Its route → state graph and exact schedule are compiled by generic K3 partition/select/schedule/lower stages; no special SDM plan class or forced anchor. |
| `compiler/unified_mixer.py` | Delete after extraction: public compile API to `frontend/api.py`; typed graph construction to `compiler/normalize/`; family selection to `compiler/select/`; executable plan to `compiler/lower/plan.py`; binding/dispatch to `runtime/bind.py`; native launchers to K1/K2/K3 backends; independent equations to reference backends; upstream/library calls to `benchmarks/comparators/`. Eliminate recipe-name dispatch entirely. |
| `compiler/unsat_catalog.py` | `tests/fixtures/unsat_catalog.py`; impossible-case generators are test material, not installed compiler code. |
| `compiler/verification.py` | `compiler/verify/{assignments,plan}.py`; independently verify complete semantics, shapes, effects, backward/decode support, schedule resources, route conservation, and every serialized plan binding. |

### Frontend and semantic IR (8 files)

| Source | Target and required change |
|---|---|
| `frontend/__init__.py` | Keep minimal exports for the public typed frontend, JSON loader, and compile API. |
| `frontend/mixer_recipes.py` | Delete Python catalog and helper-generated named specs. Put each named mapping in `recipes/kernels/*.json`; complete models in `recipes/architectures/*.json`; implement one versioned loader/validator in `frontend/recipes.py`. No recipe-specific `if alias` or compiler name dispatch. |
| `frontend/spec.py` | `frontend/spec.py` as user-facing graph/schema input only. Preserve generic axes; replace `RecurrentAlgorithm` architecture names and fixed model shortcuts with typed transition/route/state nodes. Move physical residency hints to compiler input hints. |
| `ir/__init__.py` | Rebuild exports for typed graph, family ops, effects, shapes, and state contracts; no backend or recipe enum in IR. |
| `ir/mixer.py` | Split into `ir/{k1,k2,k3,graph,state}.py`. Remove `MixerBackend` from semantics, architecture booleans and names from equations, and `EXTERNAL_OPAQUE` coverage. Represent combinations as connected typed nodes rather than one expanding spec. |
| `ir/recurrence.py` | `ir/k2.py` validation section; validate transition equations, state bundles, scan legality, output timing, gradients, and decode state. |
| `ir/softmax.py` | `ir/k1.py` validation section; validate mask/routing semantics, normalization, dimensions, and sparse work avoidance. |
| `ir/sparse_state.py` | `ir/k3.py` validation section; validate route/index, collision order, state mutation, read timing, and backward contract. |

### References, fixtures, runtime, and root-level modules (16 files)

| Source | Target and required change |
|---|---|
| `oracles/__init__.py` | Delete package; reference backend API replaces it. |
| `oracles/composition.py` | `backends/reference/numpy/graph.py`; independently execute typed K1/K2/K3 graphs in float64 and reject underspecified nodes. Do not dispatch by recipe name. |
| `oracles/matrix_state.py` | `backends/reference/numpy/k2.py`; retain recurrence, chunked form, and VJP for independent differential checks. |
| `oracles/nonlinear_recurrence.py` | `backends/reference/numpy/k2_operators.py`; retain distinct equations, explicit serial/parallel traits, and float64 behavior. |
| `oracles/routed.py` | `backends/reference/numpy/k1.py` for routed reduction and `backends/reference/numpy/k3.py` for ordered writes; remove the old `MixerSpec`-only entrypoint. |
| `oracles/softmax_attention.py` | `backends/reference/numpy/k1.py`; retain probabilities and VJP independently of native K1. |
| `oracles/sparse_slot.py` | `backends/reference/numpy/k3.py`; retain recurrent/chunked/VJP formulations and collision-order policy. |
| `presets.py` | Declarative named cases to `recipes/kernels/*.json`; test-only synthetic specs to `tests/fixtures/specs.py`. Delete Python presets from core. |
| `pretraining.py` | Split config/model/block/projections into `architectures/urm_decoder/`, optimizer/data/loop into `train/`, and FLOP/memory/gradient accounting into `benchmarks/accounting/`. The model requests compiled mixers through the frontend; delete the core module. |
| `routed_reduction.py` | Typed equation/signature to `ir/k1.py`, tensor metadata to `ir/types.py`, backend protocol to `backends/interface.py`, selection registry to `compiler/select/registry.py`, result type to `runtime/result.py`. Remove separate v1 registry/API. |
| `runtime/__init__.py` | Keep only binder/session public exports; remove the old `BackendRegistry`. |
| `runtime/decode.py` | `runtime/state.py` for generic compiled-plan state sessions and operand/state ABI checks. K1 cache policy belongs to `inference/`; session construction must consume the selected plan rather than defaulting from a spec. |
| `runtime/registry.py` | Delete legacy `MixerSpec` backend registry. Use compiler capability selection plus runtime plan binding; update tests and root exports. |
| `runtime/sparse_memory.py` | Delete SDM-only binder. Move generic plan binding, launch-config verification, and K3 state lifecycle into `runtime/bind.py`/`runtime/state.py`. |
| `sparse_state_mixer.py` | Split semantic contract into `ir/k3.py`, capability envelope into `compiler/select/k3.py`, launch schedule into `compiler/schedule/k3.py`, NumPy equation into `backends/reference/numpy/k3.py`; delete root module. |
| `sparse_state_profile.py` | `benchmarks/profiling/state_stage.py`; remove direct production-kernel import. Profiling instruments public plan steps externally. |

## External consumers to change with the cutover

- Rewrite imports and fixtures in **all** `tests/` and `benchmarks/` callers of the old paths. Replace tests of aliases/registries with public API, typed-plan, and independent-reference tests; keep source comparator tests under the comparator suite. No tests import private backend kernels merely to infer coverage.
- Move the `RecipeMixer` generic decoder model out of `benchmarks/master_table.py` into a benchmark model module. Make the benchmark consume JSON recipes through the frontend, and emit separate fragment and complete-architecture result sets. Reclassify existing results without discarding provenance.
- Add full source-model validation for Samba and PAttention before assigning them full architecture coverage. For each named model record source revision, exact graph, external components, parameter/state/cache mapping, training gradients, decode continuity, measured shapes/devices, and comparator identity.
- Build `inference/` with API, request/sampling schema, scheduler, cache/block manager, model runner, architecture loader, and measurements. Build `train/` with model adapter, data, optimizer, loop, reproducibility, and accounting. Both compile via the public frontend and bind through runtime. Avoid copying Atma's architecture implementation into URM.
- Update `README.md`, `docs/README.md`, compiler/runtime/validation documents, packaging configuration, and result schemas to the final paths and evidence taxonomy. Remove text that promises retained public old paths or uses “model-level” for a generic decoder swap.

## Completion gates for the single clean cutover

1. An inventory check accounts for all 92 old Python paths: each is replaced, split, or deleted as above; none survives as a compatibility shim. Static imports from `src/urm` to `architectures/`, `train/`, `inference/`, `benchmarks/`, `tests/`, upstream source trees, or old `adapters/`/`oracles/` are zero.
2. Backends import only IR and pure backend contracts; compiler stages import IR and capability facts; runtime imports verified plans and launchers; external applications import the public frontend/runtime. No backend imports a compiler search/plan type.
3. K1, K2, and K3 plans survive serialize/deserialize and execute the exact selected steps for forward, backward where claimed, and persistent decode where claimed. Invalid anchor/spec combinations, missing operands, conflicting state modes, and unsupported schedules decline before execution.
4. Independent reference parity, gradients/VJPs, state continuity, and numerical tolerances pass for the supported equations; schedule/solver decisions agree with exhaustive small cases and measured probes. Unsupported compositions fail explicitly.
5. Representative full-model train and inference paths use only public URM APIs. Coverage tables distinguish fragments, generic-model integration, and faithful source models; claims track the corresponding evidence and performance budgets.

This plan intentionally does not prescribe a compatibility period. Land the refactor as one coordinated API, import, test, benchmark, and documentation cutover before treating `src/urm` as the stable core boundary.
