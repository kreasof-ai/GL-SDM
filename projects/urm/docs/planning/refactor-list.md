# URM refactor: remaining work

Status: active backlog audited against the tree at `0377bb3` on 2026-09-25.
This replaces the old 92-file transfer inventory. File moves, deletion of the
legacy mixer path, the Python recipe catalog, the SDM composite, and the old
probe directory are complete and are not work items here. A moved file is not
evidence that its current behavior meets the contract.

The [backend unification audit](backend-unification.md) defines the provider
contract and [branch admission rule](backend-unification.md#admission-rule-for-a-backend-branch).
The [generality axes](../compiler/generality-axes.md) define which semantics are
represented and which remain open. Use those contracts when implementing this
list.

## Current baseline

- `src/urm` has the graph loader, typed program, compiler, plan binder,
  NumPy/Torch references and K1/K2/K3 native modules. The old architecture
  adapters and oracle package are outside core or deleted.
- The live v2 catalog has 15 kernel graph documents: 14 K1 `weighted_reduce`
  fragments and one K3 route-to-state fragment. There is no live K2 graph
  recipe. MHA has a reference graph test (CPU when Torch is available); K3 has
  a native GPU graph test.
- `compile_graph` can choose a K2 anchor, but `BoundGraphPlan` has no K2 graph
  executor. Selecting an anchor does not establish executable coverage.
- `train/loop.py` remains an SDM comparison model and helper collection;
  `inference/` and `architectures/` are marker packages. The committed master
  table is historical evidence from a runner that is no longer in this tree.

## End-state boundary

1. Core contains typed semantic operations, compiler passes, bound execution,
   independent references and only reusable K1/K2/K3 backend implementations.
   Model arrangements, upstream comparators, training, inference, benchmarks and
   profiler policy stay outside `src/urm`.
2. K1/K2/K3 are execution families, not three universal binary kernels. New
   backend branches implement a typed generality axis or a reusable
   shape/device/mode specialization. Two structurally independent client graphs
   must exercise each new branch. A single-model implementation stays in its
   recipe/comparator package while its axis is derived.
3. JSON describes operations and model arrangement; architecture names are
   metadata, never an equation or backend switch. An unrepresentable component
   is explicitly unsupported. A kernel fragment cannot claim source-model
   coverage.
4. The compiler selects one executable provider for every lowered region.
   Runtime invokes the serialized plan with checked operand/state ABI. Reference
   and library execution are explicit tiers, with their selection and cost
   visible; no fallback silently counts as native coverage.

## 1. Make the graph semantics truthful

| Live files | Remaining work | Close when |
|---|---|---|
| `recipes/schema/graph-recipe.schema.json`, `src/urm/compiler/normalize/graph.py`, `src/urm/ir/program.py` | Parse every accepted semantic field into IR or reject it. K3 JSON currently accepts `update_rule="additive"` and `collision_policy="sum"`, yet normalization creates decayed-delta/ordered defaults; route/state dtype is hardcoded BF16. K2 JSON exposes layout/update fields that normalization drops. Remove placeholder shapes where a provider needs concrete constraints. | Mutating any declared equation, timing, collision, dtype or layout field changes the normalized signature and execution, or fails before compilation. Unsupported combinations cannot normalize as a different equation. |
| `src/urm/ir/program.py` and `src/urm/compiler/pipeline.py` | Replace `OrderedRecurrence.algorithm: str` with a closed, serialized transition/state/readout descriptor. `_equation_contract_for` currently constrains only attention; match K2/K3 equations, effects, backward/decode modes and numerical policy too. Remove the `ir → compiler.common.diagnostics` dependency. | An arbitrary transition string cannot receive a K2 anchor. A legal transition produces the same descriptor after serialization; a mismatched provider declines with a specific reason. |
| `src/urm/compiler/pipeline.py::_weighted_reduce_anchor_kind`, `src/urm/runtime/bind.py` | Bind typed operand roles instead of literal tensor names `query` and `key`; eliminate runtime lookup of global names. Validate actual shapes, head maps, masks, state order and output arity against the selected region ABI. | Alpha-renaming every recipe/node/tensor leaves its normalized semantics, selected provider and results unchanged. Invalid operands fail before a kernel runs. |
| `src/urm/frontend/recipes.py`, `recipes/architectures/{samba,pattention}.json` | Add executable architecture-graph validation and resolution, or remove the `complete_model_graph` declaration from the two fragment-only placeholders. Their `coverage.level` already says `kernel_fragment`; loader acceptance must not imply a runnable model. Keep the K1 fragments in `recipes/kernels/`. | A complete model document resolves all layers, parameter transforms, state/cache edges and external components; a fragment-only document cannot be loaded as a complete model. |

## 2. Replace parallel selection and hardcoded binding

| Live files | Remaining work | Close when |
|---|---|---|
| `src/urm/backends/interface.py`, `src/urm/compiler/select/{registry,anchors}.py`, `src/urm/compiler/pipeline.py` | Replace the test-only `BackendRequest`/`CapabilityRegistry` path and separate `ExecutionAnchor` catalog with one provider contract. Match equation, operand roles, shape, device, dtype/layout, effects, forward/backward/decode, state ABI, numerical policy and launcher. Remove `_GRAPH_TARGETS` name sets, unconstrained empty contract matching, first-match-by-kind behavior, global provider mutation and the SDM upstream probe/adapter override in core. | Every selected provider is executable and exact for the requested descriptor. Forced incompatible providers decline before plan emission. Core imports and native selection need no comparator checkout or registration side effect. |
| `src/urm/compiler/lower/plan.py`, `src/urm/compiler/pipeline.py`, `src/urm/runtime/bind.py` | Serialize a complete per-region dispatch: provider ID, typed operand/state bindings, layout, launch config, mode and backward/decode contract. Replace binder branches on K1/K3 anchor strings and embedded Torch attention equations with provider invocation. Account for every declared operation in a proven fused region or its own executable step, or reject it at compile time; current `Score`, `Select`, `Transform`, `Matmul`, `OrderedRecurrence` and other nodes cannot silently disappear. | A K1, K2, K3 and mixed graph execute from the selected serialized steps. Missing, extra, duplicate, reordered or changed steps fail binding; a compiled K2 recurrence cannot fail with “no graph executor.” |
| `src/urm/backends/triton/k3/{route_launcher,state_launcher}.py`, `src/urm/runtime/bind.py` | Preserve the certified native route output through graph edges and use `from_native_generation` for trusted routes. Validate externally supplied route values or require a separately declared trust contract. Pass K3 read timing and other semantic fields to the Torch reference as well as native. | Generated routes have verifiable provenance; arbitrary input addresses cannot reach `certify_trusted`. Before/after-update reference and native results agree. |
| `src/urm/runtime/state.py` | Construct state/decode sessions from the bound plan ABI instead of independent family flags; specify allocation, aliasing, mutation, ownership and lifetime. Keep request scheduling and cache eviction in `inference/`. | Prefill followed by repeated decode matches reference continuation, and incompatible state or cache modes fail binding. |

## 3. Complete references, then qualify native axes

| Live files | Remaining work | Close when |
|---|---|---|
| `src/urm/backends/reference/numpy/`, `src/urm/backends/reference/torch/` | Add one graph interpreter over the same typed region ABI. NumPy needs routed K1 and full route production; Torch needs dense K1 attention, K2 transitions and route production. Keep NumPy float64 math independent of native kernels and Torch equations transparent and differentiable. Split the `k2_operators.py` collection by semantic operator only when the operator boundaries are real. | Every qualified K1/K2/K3 descriptor executes on the reference tier, including full route→state K3 on CPU, gradients/final-state cotangents where claimed, and state continuation. Unsupported equations decline explicitly. |
| `src/urm/backends/triton/k1/`, `k2/`, `k3/` | Audit every branch against the generality-axis admission rule. Keep shared equation kernels; move source-named, single-use RWKV/Mamba/TTT/Titans/HLA or similar wrappers and operand assembly outside core until a reusable axis and second independent client are shown. Retain different physical kernels only for a verified dependency, memory, numerical or performance regime. FFT convolution gets its own typed operator/cost instead of posing as a recurrence. | Core dispatch contains no architecture ID/name condition. Each retained branch documents its descriptor, two independent consumers, limits, forward/backward parity, and why an existing launcher cannot serve it well. |
| `src/urm/backends/triton/k1/row_scale.py`, `src/urm/ir/k3.py`, `src/urm/backends/triton/k3/state.py` | Move schedule vocabulary and policy out of backend/IR. `row_scale.py` imports compiler schedule types; `ir/k3.py` computes a launch schedule; K3 state contains an injectable profiler hook. Pass backend-facing immutable configs and profile plan steps externally. | Import direction is `compiler → IR/backend` and `runtime → backend`; backends and IR import no compiler search/schedule internals, and production kernels have no benchmark-owned profiler policy. |

## 4. Make compiler decisions per region

| Live files | Remaining work | Close when |
|---|---|---|
| `src/urm/compiler/partition/`, `pipeline.py` | Implement graph regions with explicit dependencies, effects, materialization and state boundaries, plus K1/K2/K3 legality. `partition/` is currently only a package marker. Do not encode a model-specific composite plan. | Dense/sparse K1, matrix/diagonal K2, and route→state K3 each show legal and rejected fusion/split cases; mixed graphs preserve edge and state order. |
| `src/urm/compiler/schedule/{search,space}.py`, `pipeline.py` | Select and verify a schedule for every schedulable region. Current compilation takes `schedulable_anchors[0]`, `_search_schedule` can return `None`, and K3 launch settings are built inline. Put family-specific choices and exact-specialization verification in their schedule owners, keyed by plan step. The retired `schedule/probes/` apparatus is not a task to restore. | A mixed graph serializes distinct, verified K1/K2/K3 configs. Applying the wrong family config, omitting one or selecting an illegal compiled specialization fails. |
| `src/urm/compiler/cost/{model,device}.py`, `pipeline.py` | Estimate useful and wasted work, route creation, index/gather/scatter traffic, SRAM/HBM movement, state carry, scan depth, backward, launches, occupancy, workspace and numerical/recompute cost. Follow graph dependencies for critical path and retain calibration uncertainty. The current generic `_cost_for` returns zero for unsupported shapes/ops. | Hand-counted cases and measured schedule ranks bound the estimates. Unknown cost or schedule is an incomplete result, never a successful tuned plan. |
| `src/urm/compiler/{normalize,rewrite,lower,verify,select,placement,solve}/` | Make each claimed stage own a callable input/output contract and independent checks. Split remaining orchestration out of `pipeline.py` by real ownership, not by empty files. Preserve solver-independent verification and typed rewrite obligations. | K1, K2, K3 and mixed graph fixtures traverse each claimed stage and fail when a stage output is tampered with. |

## 5. Migrate recipes and applications

| Live files | Remaining work | Close when |
|---|---|---|
| `recipes/kernels/`, `recipes/architectures/`, `benchmarks/architecture-coverage.json` | Re-author the remaining K2 and K1-variant coverage as typed graph documents from exact source equations. Add model compositions only after their layer, projection, state/cache and gradient contracts are represented. Do not promote a fragment because it bears an architecture name. | Every represented recipe has reference execution; native and model qualification are separate recorded levels. Two independent clients exercise every new core backend axis. |
| `train/loop.py`, `train/`, `architectures/`, `benchmarks/models/`, `benchmarks/accounting/` | Move the fixed SDM comparison model and ledger out of the general trainer. Build a graph-configured decoder and one model-agnostic data/optimizer/accumulation/checkpoint loop around public frontend and bound plans. Preserve the frozen SDM comparison as a benchmark fixture. | The same runner trains tiny K1, K2, K3 and mixed models without family/name branches; gradients, two optimizer steps, state policy, checkpoint/resume and accounting match references. |
| `inference/`, `src/urm/runtime/state.py` | Build the model loader, request API, prefill/decode runner, sampling, scheduler and cache/state manager around bound plans. Start with one request and repeated decode, then add variable-length batching, cancellation and isolation. | Checkpoint-loaded model logits and state match teacher-forced/reference runs; interleaved requests match separate runs and release resources on cancellation. |
| `benchmarks/comparators/sdm/compiled.py` | Move its `urm::` custom-op registrations to a comparator-specific namespace. Keep upstream revision checks and external adapters outside core. | Importing the comparator cannot register a production `urm::` op. |

## 6. Rebuild measurement and claims

| Live files | Remaining work | Close when |
|---|---|---|
| `docs/validation/master-table.md`, `results/validation/master-table.json`, `benchmarks/` | Mark the old 62-row table as historical generic-decoder evidence: its stated generator `benchmarks/master_table.py` is absent. Build a graph-native runner and regenerate results with explicit fragment, generic-decoder and source-model levels. Preserve old artifacts with their runner/commit identity. | Every row identifies recipe version, exact graph/plan, comparator revision, device, mode, timed boundary, parity and fallback tier. No source-model claim comes from a fragment swap. |
| `benchmarks/inference_report.py`, `docs/validation/inference-throughput.md` | Relabel current kernel prefill/decode figures as kernel evidence. Add a separate request-level serving benchmark after `inference/` exists; remove placeholder mappings and account for tokenization, scheduling, cache allocation and sampling boundaries. | Time to first token, decode latency, tokens/s, peak/cache memory, concurrency and reference logits are reported from actual requests. |
| `benchmarks/pretraining_step.py`, `profile_pretraining_step.py`, `benchmarks/accounting/`, `benchmarks/release_coverage.py` | Make the frozen paired benchmark call the public trainer, move reusable data/checkpoint logic into `train/`, version the FLOP/memory ledger, and add missing K2/K3 state-gradient evidence. Profiling observes public plan steps. | Resume reproduces next batch/update; measured and semantic accounting are distinguishable; release gates consume actual parity and performance fields. |
| `README.md`, `docs/README.md`, packaging and test suites | Align commands and scope with implemented applications. Keep core packaging lean and make project-level apps runnable from a clean checkout with declared optional dependencies. Add public-boundary negative tests and report GPU/upstream skips separately. | A clean environment can train a tiny graph model and generate from its checkpoint; documentation and tables state exactly which gates passed. |

## Completion gates

1. **Semantic authority:** JSON and programmatic graphs have one closed equation
   vocabulary. Name changes do not change meaning; semantic changes cannot be
   silently ignored. Unsupported axes and combinations decline before selection.
2. **Executable plan authority:** all selected K1/K2/K3 and mixed regions have
   providers, operand/state bindings, schedules and costs. Serialized plan
   tampering fails before execution. Runtime has no recipe or architecture
   dispatch path.
3. **Reference and native truth:** independent NumPy/Torch execution, state
   continuation and all claimed VJPs precede native coverage. GPU parity and
   performance are reported only for measured modes, dtypes and shapes.
4. **Axis reuse:** every new core backend branch passes the two-client admission
   rule; single-architecture implementations remain outside core. Legal
   cross-axis combinations run without adding a recipe-name conditional.
5. **Applications:** one unchanged trainer and one inference engine consume
   graph-built models and bound plans for K1, K2, K3 and mixed fixtures, with
   checkpoint/resume, state isolation and reference parity.
6. **Evidence:** tables separate representation, reference execution, native
   kernel qualification, generic-model integration and faithful source-model
   comparison. Historical measurements remain identifiable and are not
   silently promoted to current product claims.

Close each gate with the exact commit, environment, tests, measured artifacts
and unsupported cases. A clean directory tree alone closes none of them.
