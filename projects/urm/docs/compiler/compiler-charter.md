# Compiler charter

**Normative for `src/urm`.** URM compiles typed routing, reduction, state and communication semantics into complete executable plans. The public model interface supplies tensors and a semantic request; architecture-specific modules, comparators, training, inference and benchmarks live outside core.

## Ownership

| Layer | Owns | Must not own |
|---|---|---|
| `frontend/` | Loading name-agnostic, versioned typed kernel or graph fragments | Architecture registry or source-model dispatch |
| `ir/` | Logical domains, role-indexed operands, closed equations, state/effect/route semantics and numerical policy | Device layout, tiles, upstream APIs, optimizer or benchmark policy |
| `compiler/` | Verified rewrite candidates, independent constraints, placement, region partition, provider/schedule selection, complete serialized plan and cost trace | Tensor-value checks, runtime state mutation, architecture names, GPU bodies |
| `runtime/` | Bind and validate the selected plan, own state sessions and invoke its provider | New semantic selection, model scheduling or hidden fallback |
| `backends/` | Reference and native implementations of admitted typed axes and physical schedules | Model-named branches, compiler policy, profiler hooks or untyped callbacks |
| `architectures/`, `train/`, `inference/`, `benchmarks/` | Model arrangements, ordinary operators, source adapters, application loops and measurement | Correctness rules for core providers |

## Semantic invariants

1. A graph is independent of tensor *names* and backend layout. Each operation binds typed roles. Every accepted JSON field changes the normalized descriptor or is rejected. An unknown transition string never selects a provider.
2. The mixer families are defined by their **read-domain** — what a token's computation reads. K1 is a streamed score/select/normalize/reduce over a logical source domain (external source streams; parallel over queries; no feedback). K2 evolves a compact **fixed-address** state bundle in token order (serial in t, O(1)-state per step). K3 evolves **indexed mutable** state with explicit route, collision, read/write and commit rules (sparse addressing). K4 is an **exact feedback substitution** over the op's own emitted history through a given triangular transition operator (serial in t, full-history read; the transition operator arrives as an operand, produced by a prior op). A graph may compose these families with ordinary typed operators such as GEMM, convolution, FFT and collectives. Those operators retain their own cost and effects. **The taxonomy is closed at four**: the read-domains {external streams, compact state, routed memory, own history} partition the sequence-mixing recurrence space — a proposed K5 must demonstrate a fifth read-domain, and anything reading none of the four is an ordinary typed operator, not a family member.
3. Routes name logical domains. Route creation, tie/capacity policy, ownership and provenance are semantic; physical addresses and communication arise only after placement. State reads, mutation, ordering, collisions and version/commit boundaries are explicit effects.
4. Reparameterization requires a registered rule with algebraic preconditions, exact versus floating-point equivalence class, numerical envelope, full operand and state VJP status, saved-state/recomputation plan and traffic delta. The unfused base candidate always remains available. A schedule never changes semantics.
5. No arbitrary tensor callback, model name, source-specific flag or opaque library callable enters the serialized semantic IR. Upstream implementations are external comparators or explicitly labeled library provider tiers; they do not define the equation.
6. Compilation intent and provider support are exact: training requires a certified backward and state cotangent path; inference requires state continuation. Unsupported dtype, shape, mode, equation or schedule returns a structured decline before binding.
7. The solver ranks bounded legal candidates; it does not prove the equation or synthesize unchecked code. Every solver model is rechecked by an independent imperative verifier. Unknown cost or missing schedule is an incomplete plan, not success.
8. Every declared node is executed, covered by a proved fusion, or rejected. A selected provider, typed bindings, effects, placement, schedule, mode, numerical policy, state ABI and reasoned fallback tier are serialized. Runtime executes exactly that plan. Tests must reject altered, missing or reordered steps.

## Backend branch admission

**Backend layout.** The backend tree is organized as `<tier>/<family>/<op>.py` — one file per admitted typed op within a family directory (K1/K2/K3/K4), per tier (numpy/torch/triton). **The backend op inventory is exactly the IR `SemanticOp` inventory**: no IR op → no backend file; no backend file → the op has no physical realization at that tier. Ordinary (non-recurrence) typed operators live in a flat `<tier>/<op>.py` module at the backend root. This correspondence is the guard against sliding into a generic tensor compiler: elementwise/tensor composition stays in `architectures/` external modules; an op file exists only where a mixer-relevant law with a closed descriptor exists. Compile-opacity (the `torch.library` custom-op wrapper) is an *invocation* mechanism owned by `runtime/opaque.py` and applied generically — kernel files carry no `torch.library` knowledge, and whether a plan step is invoked opaquely is a compiler config decision, not a kernel property.

A new semantic axis may enter IR and independent NumPy/Torch references before native execution exists. A new **physical** branch under `backends/` is admitted only when all of the following are recorded:

1. Its selection key is a closed typed property (equation, state shape, score/reducer, addressing, read timing, precision, layout or mode), never an architecture name or tensor spelling.
2. Two structurally independent client graphs exercise it: two unrelated source models, or one source model plus a nontrivial synthetic graph combining another legal axis. A renamed recipe or another batch size does not count.
3. Both clients pass independent reference, forward/VJP/state-continuation and legal cross-axis tests. Unsupported combinations decline.
4. A distinct dependency, memory access, numerical stability or measured performance regime justifies the physical schedule. Otherwise reuse an existing launcher.

A source-only implementation remains in its external comparator package until this admission record exists. Fusing adjacent calls additionally requires a registered equivalence rule and a measured benefit including launch, materialization, route and state traffic.

## Required compiler output

Compilation returns a deterministic serialized executable plan, a trace of accepted/rejected candidates and declines, analytical cost features separate from measurement, and mode-specific unresolved obligations. No selected anchor may fail later merely because the binder lacks that family. See the [runtime contract](../runtime/execution.md), [generality axes](generality-axes.md) and [roadmap](../planning/roadmap.md).
