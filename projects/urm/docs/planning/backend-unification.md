# Backend unification: current gap and hard cut

Audit date: 2026-09-25. Scope: the current `src/urm` graph path, backend implementations,
15 v2 kernel recipes, and the contracts in `docs/kernels/` and
`docs/compiler/generality-axes.md`. This is a source and CPU-path audit, not GPU
qualification of the proposed lowerings.

## Decision

Keep **K1, K2, K3 as three execution families**, defined by dependency and
addressing, rather than as three source files or a promise of exactly three GPU
binaries:

| Family | Semantic region | Physical choices |
|---|---|---|
| K1 | Stateless score, select, normalize and reduce over a logical source domain | Dense online reduction, sparse traversal, decode, grouped heads, segmented reduction |
| K2 | Ordered evolution of a compact, fixed-address state bundle | Token loop, affine scan, chunk solve, decode, recompute or saved-state backward |
| K3 | Ordered evolution of an indexed state with explicit route and collision semantics | Route production, gathered state update/read, chunked sparse solve where legal, decode |

A graph can compose these regions with ordinary typed operators such as projections,
pointwise transforms, GEMM, FFT convolution, and collectives. Calling those operators
K2 solely to preserve a count of three would erase their cost and semantic boundaries.
K2 and K3 can share the same delta algebra; they need different physical schedules
because K3 has route/index and collision costs. One family can require several
specialized launchers. A new equation belongs in a family only after its state,
read/write order, gradients, and numerical policy are represented and checked.

The present source does **not** yet realize this contract end to end. The three-family
idea remains viable as an organizing and code-generation basis, while unrestricted
combinations and near-ideal performance remain unqualified claims. **Core backend
growth is axis-driven:** an additional native branch must serve a reusable semantic
axis or a shape/device/mode specialization of one, never one architecture's identity.

## Observed breaks in the current path

| Location | Observed behavior | Consequence |
|---|---|---|
| `recipes/kernels/` and `runtime/bind.py` | The 15 live graphs contain 14 `weighted_reduce` nodes and one route→sparse-state graph. Binding executes softmax K1 and K3 only. | No executable graph-native K2 recipe. A directly constructed `OrderedRecurrence(algorithm="arbitrary_transition")` compiles to the K2 reference anchor, then raises `PlanBindingError` on execute. |
| `compiler/pipeline.py::_weighted_reduce_anchor_kind` and `runtime/bind.py::_execute_k1_attention_node` | K1 recognition requires operands literally named `query` and `key`; execution also fetches those names globally. | Alpha-renaming the unchanged MHA graph to `q` and `k` makes reference compilation fail with `no_anchor_available`. Operand *roles* must be typed, not inferred from tensor names. |
| `ir/program.py::OrderedRecurrence`, `compiler/normalize/graph.py` and `compiler/pipeline.py::_equation_contract_for` | K2 carries an algorithm string. JSON exposes additional transition fields but normalization drops them. Only attention receives an equation contract; K2 anchors have unconstrained contract sets. | A K2 anchor can be selected without establishing which recurrence equation it executes. |
| `recipes/schema/graph-recipe.schema.json` and `compiler/normalize/graph.py` | K3 JSON accepts `update_rule="additive"` and `collision_policy="sum"`; normalization omits both and instantiates the defaults `decayed_delta` and `ordered`. Route/state dtype is hardcoded BF16. | A validated recipe can silently change its equation. A local reproduction confirmed this for additive/sum. Reject unsupported combinations or preserve and implement them; never default over an explicit field. |
| `backends/interface.py`, `compiler/select/registry.py`, `compiler/select/anchors.py`, and `compiler/pipeline.py` | The `BackendRequest`/`CapabilityRegistry` path is used by its own tests, whereas graph compilation selects `ExecutionAnchor` from a second catalog and `_GRAPH_TARGETS` name sets. | There are two different capability stories. Neither binds a complete equation, shape, state effect, mode, VJP, precision, and executable launcher in one place. |
| `runtime/bind.py` | Runtime branches on anchor strings and embeds Torch attention math and K3 operand assembly. It binds plan steps by `note`, not a typed region/operand ABI. | The binder is another backend dispatcher and reference implementation; selected anchors are not independently executable plan steps. |
| `backends/reference/numpy/` and `backends/reference/torch/` | NumPy has K1 attention, canonical K2, a collection of K2 operators, and K3. Torch has routed K1 and K3 only; Torch attention reference lives in the binder. K3 has no reference route producer in the graph path. | There is no uniform all-family reference graph interpreter, no Torch K2 reference, and no CPU execution of the full K3 recipe. The Torch package docstring overstates its coverage. |
| `backends/triton/k2/` (resolved) | The source-named RWKV/Mamba/TTT/Titans/Hyena kernels and their autograd wrappers were removed from core (`nonlinear.py`, `inner_state.py`, `backward.py`, `second_order.py`, and the `numpy/k2_operators.py` oracle). They had no consumers after the legacy path was deleted and failed the branch admission rule. They can return only as typed axes with two independent clients. | Resolved by deletion; distinct equations stay outside core until axis-qualified. |
| `backends/triton/k3/{route_launcher,state_launcher}.py` and `runtime/bind.py` | A certified native route output type and `from_native_generation` bridge exist, but graph binding returns raw route tensors and calls `certify_trusted`, which checks structure without scanning values. | The graph loses route provenance. An input or future transform that supplies addresses can receive the trusted path without proof of bounded, sorted, unique routes. |
| `compiler/select/anchors.py` and `compiler/pipeline.py` | The generic compiler still contains an SDM upstream support probe, a pinned adapter string and override-specific branches. | Benchmark comparator policy leaks into core selection, even though `compile_graph` itself restricts selection to trusted target sets. |

One further K3 mismatch: the binder does not pass `node.spec.read_timing` to
`torch_sparse_state_mixer`; that reference defaults to `CURRENT_STATE`. This can
execute a before-update read for an after-update graph when the K3 reference state
node is bound directly.

## Semantic basis to implement

For the stateful families, a useful *typed composition* is:

```text
routes_t = Route(inputs_t, state_(t-1)?)
base_t   = Transition(state_(t-1), inputs_t, routes_t)
read_t   = Read(base_t or state_(t-1), inputs_t, routes_t)
state_t  = Commit(base_t, Write(read_t, inputs_t, routes_t), collision_policy)
out_t    = Readout(state_t or base_t or state_(t-1), inputs_t, routes_t)
```

The alternatives are explicit descriptor choices, not runtime guessing. K2 uses
fixed dense/compact addressing; K3 uses indexed addressing. The canonical delta
law is one legal `Transition`/`Write` pair, not the definition of every stateful
operator. A scan/chunk lowering is legal only when the update admits a proved
bounded composition and when route/coefficient dependencies permit it. A route
depending on evolving state imposes an ordered dependency even if route generation
alone is parallel.

1. **Typed operands and effects.** Give every region named *roles* (`query`,
   `key`, `value`, `state`, `route`, `gate`), independent of user tensor names.
   Describe logical axes, shape constraints, state bundle, read timing, mutation,
   route ownership, collision policy, determinism, dtype and accumulation/cast
   points. Validate runtime shapes against the compiled constraints.
2. **Closed equation descriptors.** Replace opaque `algorithm` and ignored JSON
   fields with serialized descriptors. K1 describes score map, visibility,
   selection, normalizer and reduction. K2 describes state transition, update,
   readout and state-dependent versus input-precomputable coefficients. K3 adds
   address generation or supplied-route policy, selected-slot decay, collision
   order and read/write timing. Changing a semantic field must change the
   descriptor and either change execution or produce a structured decline.
3. **Proved equivalence rules.** Canonicalize only with a derivation and parity
   fixture. For example, a K2 additive matrix update and a structured SSM can
   share an affine-state lowering after mapping axes and read timing. A K2/K3
   delta update can share a recurrence law after proving selected-slot decay and
   collision behavior. Do not treat a nonlinear inner optimizer, a numerically
   stabilized log-state, or FFT convolution as the same transition by label.
4. **One provider contract.** A provider declares a semantic descriptor pattern,
   constraints on device/shape/layout/mode/precision, forward and backward
   support, numerical policy, schedule space and the exact callable/codegen
   entrypoint. The compiler selects and serializes a provider plus validated
   region ABI and launch config. Runtime only invokes the selected callable and
   checks the ABI. An unsupported specialization declines at compile time.
5. **Reference is a complete execution tier.** Build a typed graph interpreter
   from the same region ABI. NumPy supplies independent float64 equations;
   Torch supplies differentiable functional versions. Both cover each *qualified*
   K1/K2/K3 descriptor, including route production and state continuation. A
   native implementation qualifies by parity against these, including all
   operand gradients and final-state cotangents where training is claimed.
6. **Fallback is explicit evidence, not coverage.** Native, library and reference
   are separate selectable tiers with recorded reason and cost. A comparator
   adapter stays outside `src/urm`. A recipe that runs only in reference mode is
   represented, not native-covered. A recipe that merely parses is not executable.

Avoid a Cartesian-product mega-kernel. Start from the law of a region, then
generate or choose a small set of verified physical variants. If a combination
cannot be fused lawfully or profitably, partition it into typed graph regions and
account for materialization, launch, state traffic, and critical path. If its
equation cannot be expressed, propose a typed generality axis and prove reuse
before adding a core backend path; otherwise leave it as an external comparator.

### Admission rule for a backend branch

The IR defines the axis and the compiler checks its legal combinations; a
backend may only declare and implement supported points or ranges on that axis.
Every conditional, specialized launcher, and backward path under
`src/urm/backends/` must answer all of these questions:

1. **Which axis does it implement?** The branch key is a typed property such as
   read timing, state bundle, transition law, addressing, normalization,
   collision policy, precision, shape, or execution mode. It cannot inspect a
   recipe/model name or source-specific tensor naming convention.
2. **Where else can it run?** Demonstrate two structurally independent client
   graphs: two unrelated named recipes, or one named recipe plus a nontrivial
   synthetic composition that combines the axis with another legal axis. Merely
   renaming the original recipe or varying a batch size is not reuse.
3. **Does the combination remain lawful?** The descriptor, reference equation,
   capability constraints, forward/backward parity, state continuation, and cost
   must hold for both clients. Unsupported cross-products decline explicitly.
4. **Why is a physical branch needed?** A separate launcher is justified by a
   different dependency, memory-access, numerical-stability or measurable
   performance regime. A new semantic axis can often use an existing generic
   launcher. A shape specialization can be shared without adding a new equation.

An implementation serving only one architecture stays in its recipe/comparator
package while its general axis is derived. If its semantics are general but no
second client is known, add the descriptor and reference first; do not promote
its source-specific native kernel into core on that evidence alone. Once the
axis qualifies, replace model-named dispatch with axis-based provider matching.

## K2 classification that prevents false merging

| Existing example | Construction route |
|---|---|
| `k2/matrix.py` canonical additive/delta/dual-gate/rank-R | Typed matrix-state transition and explicit mutually exclusive update forms; schedule by dimensions, rank, precision and mode. This is the strongest current reusable native base. |
| `k2/diagonal.py`, Mamba-2 style structured state | Affine/diagonal K2 state bundle, grouped-axis map, explicit read timing. Prove any matrix-state specialization before sharing a launcher. |
| RWKV-6 bonus correction | Matrix state plus **pre-update read with bonus term**; it is not the current after-update delta rule. May share K2 machinery after a readout rewrite is proved. |
| RWKV-4 log-scaled numerator/denominator | Candidate normalized-state and rescaling axis. Preserve its three-part stable representation. Keep a source-specific native path outside core until another independent client exercises the axis. |
| TTT/Titans/MesaNet inner updates | Typed ordered inner-update/solve region with optimizer state and outer-gradient policy (A12); keep serial execution when no bounded associative summary is proved. |
| Mamba-3 trapezoidal/phase state | Candidate phase/multi-component state axis (A9), with exact update order and cache ABI. Admit a core branch only after cross-architecture or nontrivial cross-axis reuse is shown. |
| Hyena/H3 FFT paths | Typed convolution and graph composition, with FFT cost/provider, outside the K2 state kernel unless a true recurrent equivalent is chosen and qualified. |

The family decision is made from data dependency and address topology. Architecture
IDs remain in recipe metadata, benchmark fixtures and comparator wrappers only.
They never select a core backend.

## File-level hard cut

| Files | Required change |
|---|---|
| `ir/program.py`, `compiler/normalize/graph.py`, `recipes/schema/graph-recipe.schema.json` | Define/parse the same closed semantic fields and operand roles. Remove `OrderedRecurrence.algorithm: str` as an execution contract. Reject explicit but unsupported fields. Make shapes/dtypes, including K3 route dtype, authoritative. |
| `compiler/select/{anchors,registry}.py`, `backends/interface.py`, `compiler/pipeline.py` | Replace the parallel registries with one provider catalog. Remove string-set `_GRAPH_TARGETS`, first-match by kind, empty-contract wildcard, SDM probe and pinned-adapter override. Keep target tier as a capability filter; resolve each region to an executable provider. |
| `compiler/partition/`, `compiler/schedule/`, `compiler/cost/` | Implement family legality and per-region schedule/cost models. Include route production, sparse traversal, state carry, VJP, workspace, and launch/synchronization costs. Emit a verified plan for every region. |
| `runtime/bind.py` | Remove K1/K3 hardcoded branches and embedded reference equations. Bind role-indexed operands, preserve certified route provenance, check plan completeness/uniqueness and invoke the provider named in each serialized step. |
| `backends/reference/numpy/{k1_attention,k2,k2_operators,k3}.py`, `backends/reference/torch/{k1,k3}.py` | Organize around semantic operators, add missing Torch K1 attention/K2 and reference route generation, and expose both through the common graph interpreter. Retain independent equations for parity; move source-specific operand assembly to recipes/comparators. |
| `backends/triton/k1/` | Keep native reduction bodies and narrow launchers; express dense/sparse/threshold/feature variants as typed capabilities. Place choice of traversal, split and decode schedule in the compiler. |
| `backends/triton/k2/{matrix,diagonal}.py` | Done: the source-named wrappers (`nonlinear`, `inner_state`, `backward`, `second_order`) were deleted from core. The retained matrix/diagonal bodies are the canonical delta-law/structured-SSM axes; audit their branches per the admission rule as they gain clients. A future FFT convolution enters as a typed operator/provider only with two independent consumers. |
| `backends/triton/k3/{route,route_launcher,state,state_launcher}.py` | Expose typed route output and state input contracts; accept trusted route shortcuts only from a certified native producer. Keep launch policy in compiler. |
| `docs/compiler/generality-axes.md`, `docs/planning/coverage.md`, `docs/validation/master-table.md` | Report parser representation, reference execution, native forward, native backward and end-to-end qualification separately. Remove the current implication that `compile_graph` executes K2. |

Perform the cut in executable vertical slices (K1 attention, canonical K2, K3
route→state), but do not ship a second fallback dispatcher or compatibility
aliases as the final structure. Each slice replaces its previous route.

## Gates before claiming clean unification

1. Renaming any recipe, node and tensor leaves its normalized semantic signature,
   provider choice and outputs unchanged. Changing an equation field changes the
   signature and either behavior or a compile-time decline.
2. Every provider selected by compilation binds and executes the corresponding
   plan step; no compiled `OrderedRecurrence` may fail with "no graph executor".
   Plans reject missing, extra, duplicate and reordered steps or incompatible
   operand roles.
3. Every qualified descriptor has independent NumPy and differentiable Torch
   graph execution, native differential tests in each claimed mode/dtype, state
   continuation, final-state VJPs, and numerical edge cases.
4. A K3 route is certified by its producing node or validated as an external
   input. `certify_trusted` cannot be reached merely because a graph contains a
   sparse-state node.
5. Every backend branch has two structurally independent consumers as specified
   by the admission rule, and selection uses semantic/shape axes without any
   architecture-name condition. A changed read timing, update law or state
   bundle declines or selects another proven provider.
6. Publish performance for whole graph execution and individual regions,
   including route generation, fallback tier, materialization, backward and
   state traffic. Efficiency is measured per qualified envelope; it is not
   inferred from expressibility.

The first blocking milestone is **semantic truthfulness**: repair ignored
fields, operand-name dispatch and unchecked K2 selection before bulk recipe
migration or tuning. Then complete the reference execution tier and provider
binding. Optimize native variants against that stable contract.
