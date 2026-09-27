# The compiler pipeline

Mirrors `src/urm/`. One entry path: a typed graph fragment in, a bound executable
plan out. There is no second, architecture-specific path.

```
JSON fragment ── frontend/recipes.py ──▶ typed document
              ── normalize/graph.py ──▶ SemanticProgram (ir/program.py, ir/types.py, ir/effects.py)
              ── compiler/pipeline.py: compile_graph(program, target=, intent=)
                   rewrite/   candidate equivalence rules + proofs (rules.py, proof.py, engine.py)
                   solve/     independent constraints + z3 model (constraints.py, z3.py)
                   placement/ locality, routes, plan, solver
                   partition/ region partition
                   cost/      analytical device cost model (cost/model.py, cost/device.py)
                   select/    anchor registry + selector (select/anchors.py, select/model.py)
                   schedule/  schedule space + search (schedule/space.py, schedule/search.py)
                   verify/    independent imperative recheck (verify/plan.py, verify/assignments.py)
              ──▶ CompilationResult (serialized plan + candidate/decline trace)
              ── runtime/bind.py: BoundGraphPlan ──▶ .execute(**operands)
```

## Facts that matter when working here

- **`CompilationIntent`** (`compiler/pipeline.py`) — `inference` vs `training`.
  Training requires the certified backward; the anchor selector declines otherwise.
- **Anchor selection is semantic** (`compiler/select/anchors.py`): each node's
  equation contract must be implemented by the selected anchor; an unknown
  transition string or a renamed recipe never selects a provider.
- **`BoundGraphPlan.execute`** (`runtime/bind.py`) runs the serialized plan steps in
  graph order and never redispatches. Steps bind operands by *role* from caller
  inputs and earlier step outputs.
- **Opaque invocation** (`runtime/opaque.py`): native K2-family layers run the mixer
  as a `torch.library` custom op so `torch.compile` treats it as an opaque boundary
  (the surround fuses; the provider runs eagerly). Kernel files carry no
  `torch.library` knowledge; opacity is a compiler config decision.
- **The backend inventory is the IR op inventory** — see [charter](charter.md) and
  [backends](backends.md).

## The training harness's use of the pipeline

`architectures/*.py` build graph documents (typed `weighted_reduce` / state-scan /
route fragments) and call `load_graph_recipe_document` → `normalize_graph_document`
→ `compile_graph`, then hold the returned plan and call `plan.execute` in forward.
`train/model.py` is a generic surround (embeddings, norms, MLP, lm_head) that never
owns a mixer equation; `train/harness.py` marks `plan.execute`-based mixers as
dynamo boundaries (`make_compile_safe`) so the model compiles around the opaque op.
