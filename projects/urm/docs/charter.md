# Compiler charter

**Normative for `src/urm`.** URM compiles typed routing, reduction, state and
communication semantics into complete executable plans. The public model interface
supplies tensors and a semantic request; architecture-specific modules, comparators,
training, inference and benchmarks live outside core. This page documents the
invariants the code enforces today; each clause names where it lives.

## Ownership

| Layer | Owns | Must not own |
|---|---|---|
| `src/urm/frontend/` | Loading name-agnostic, versioned typed kernel or graph fragments (`recipes.py`) | Architecture registry or source-model dispatch |
| `src/urm/ir/` | Logical domains, role-indexed operands, closed equations, state/effect/route semantics and numerical policy (`program.py`, `types.py`, `effects.py`) | Device layout, tiles, upstream APIs, optimizer or benchmark policy |
| `src/urm/compiler/` | Verified rewrite candidates, independent constraints, placement, region partition, provider/schedule selection, complete serialized plan and cost trace (`normalize/`, `rewrite/`, `partition/`, `placement/`, `cost/`, `select/`, `schedule/`, `solve/`, `verify/`, `lower/`) | Tensor-value checks, runtime state mutation, architecture names, GPU bodies |
| `src/urm/runtime/` | Bind and validate the selected plan, own state sessions and invoke its provider (`bind.py`, `certification.py`, `opaque.py`, `result.py`) | New semantic selection, model scheduling or hidden fallback |
| `src/urm/backends/` | Reference and native implementations of admitted typed axes and physical schedules (`numpy/`, `torch/`, `triton/`) | Model-named branches, compiler policy, profiler hooks or untyped callbacks |
| `architectures/`, `train/`, `extra/` | Model arrangements, ordinary operators, source adapters, application loops and measurement | Correctness rules for core providers |

## Semantic invariants

1. **Name independence.** A graph is independent of tensor *names* and backend
   layout. Each operation binds typed roles. Every accepted JSON field changes the
   normalized descriptor or is rejected. An unknown transition string never selects
   a provider — provider selection keys on closed typed properties only.
2. **The mixer families are read-domains, and the taxonomy is closed at four.**
   K1 is a streamed score/select/normalize/reduce over a logical source domain
   (external source streams; parallel over queries; no feedback). K2 evolves a
   compact **fixed-address** state bundle in token order (serial in t, O(1) state
   per step). K3 evolves **indexed mutable** state with explicit route, collision,
   read/write and commit rules. K4 is an **exact feedback substitution** over the
   op's own emitted history through a given triangular transition operator (the
   operator arrives as an operand; `backends/*/k4/triangular_solve.py`). A graph may
   compose these with ordinary typed operators (GEMM, convolution, FFT, collectives)
   that retain their own cost and effects. A proposed K5 must demonstrate a fifth
   read-domain; anything reading none of the four is an ordinary typed operator.
3. **Routes name logical domains.** Route creation, tie/capacity policy, ownership
   and provenance are semantic; physical addresses and communication arise only
   after placement. State reads, mutation, ordering, collisions and version/commit
   boundaries are explicit effects (`ir/effects.py`).
4. **Reparameterization requires a registered rule** with algebraic preconditions,
   exact-vs-fp equivalence class, numerical envelope, full operand/state VJP status,
   saved-state plan and traffic delta. The unfused base candidate always remains
   available. A schedule never changes semantics.
5. **No arbitrary tensor callback, model name, source-specific flag or opaque
   library callable enters the serialized semantic IR.** Upstream implementations
   are external comparators (`extra/comparators/`) or explicitly labeled
   library provider tiers; they do not define the equation.
6. **Compilation intent and provider support are exact**: training requires a
   certified backward and state cotangent path; inference requires state
   continuation. Unsupported dtype, shape, mode, equation or schedule returns a
   structured decline before binding (`runtime/certification.py`).
7. **The solver ranks bounded legal candidates**; it does not prove the equation or
   synthesize unchecked code. Every solver model is rechecked by an independent
   imperative verifier (`compiler/verify/`). Unknown cost or missing schedule is an
   incomplete plan, not success.
8. **Every declared node is executed, covered by a proved fusion, or rejected.** A
   selected provider, typed bindings, effects, placement, schedule, mode, numerical
   policy, state ABI and reasoned fallback tier are serialized. Runtime executes
   exactly that plan. Tests must reject altered, missing or reordered steps.

## Backend branch admission

The backend tree is `<tier>/<family>/<op>.py` — one file per admitted typed op per
family (K1/K2/K3/K4) per tier (numpy/torch/triton). **The backend op inventory is
exactly the IR `SemanticOp` inventory**: no IR op → no backend file; no backend file
→ the op has no physical realization at that tier. Ordinary typed operators live in
flat `<tier>/<op>.py` modules at the tier root (`backends/torch/merge.py`). This
correspondence is the guard against sliding into a generic tensor compiler:
elementwise/tensor composition stays in `architectures/` external modules.

Compile-opacity (the `torch.library` custom-op wrapper) is an *invocation* mechanism
owned by `runtime/opaque.py` and applied generically — kernel files carry no
`torch.library` knowledge; whether a plan step is invoked opaquely is a compiler
config decision, not a kernel property.

A new **physical** branch under `backends/` is admitted only when all of:

1. Its selection key is a closed typed property (equation, state shape,
   score/reducer, addressing, read timing, precision, layout or mode), never an
   architecture name or tensor spelling.
2. Two structurally independent client graphs exercise it: two unrelated source
   models, or one source model plus a nontrivial synthetic graph combining another
   legal axis. A renamed recipe or another batch size does not count.
3. Both clients pass independent reference, forward/VJP/state-continuation and
   legal cross-axis tests. Unsupported combinations decline.
4. A distinct dependency, memory access, numerical stability or measured performance
   regime justifies the physical schedule. Otherwise reuse an existing launcher.

A source-only implementation remains in its external comparator package until this
admission record exists. Fusing adjacent calls additionally requires a registered
equivalence rule and a measured benefit including launch, materialization, route and
state traffic.

## Current admission status

See [backends.md](backends.md) for the admitted op inventory and
[catalog.md](catalog.md) for which registry rows each family serves. The one
standing charter debt: `mamba1` remains reference-tier (the K2 elementwise gate has
a single client; clause 2 blocks a native branch).
