# URM compiler charter

**Status:** normative for `src/urm/compiler/`.
**Definition.** URM is a **semantic-to-execution compiler for routed sequence
models** - not a collection of optimized kernels and not a general tensor
compiler. Researchers describe routing, state, and communication semantics;
URM verifies algebraic reparameterizations, plans placement and communication,
and lowers onto trusted execution anchors (FA / FLA / grouped GEMM / scan /
SDM-style page ops / collectives / generated kernels).

```text
architecture/NAS specification
  -> semantic routing and state IR          (compiler/semantic.py)
  -> verified algebraic reparameterization  (compiler/rewrite.py)
  -> immutable candidate enumeration        (compiler/planner.py)
  -> backend-independent constraint IR      (compiler/constraints.py)
  -> optional Z3 feasibility/optimization   (compiler/solver.py)
  -> independent imperative verification    (compiler/verification.py)
  -> placement, sharding, communication     (compiler/placement.py,
                                             compiler/route_protocols.py,
                                             planner.py)
  -> trusted execution anchors + visitors   (compiler/execution.py)
  -> FA / FLA / grouped GEMM / scan / SDM /
     collectives / generated kernels        (urm/adapters, urm/backends,
                                             compiler/anchors)
```

The normative kernel-generation pipeline lives in
[docs/kernel-generation.md](kernel-generation.md).

Performance is an acceptance requirement of individual lowerings; it is not
the definition of URM. A lowering that cannot beat its baseline is recorded as
such and retained; a semantic that cannot be expressed at all is a URM failure.

## Invariants

1. **Architecture semantics are independent of backend implementation.**
   `SemanticProgram` says nothing about devices, tiles, threads, or schedules.
   The same program must compile to different anchors without rewriting.
2. **Routing operates over logical domains**, never physical tensor indices.
   `RouteSpec` names logical query/source domains (`sequence`, `expert`,
   `memory_page`, ...). Physical placement is decided later and may turn any
   logical edge into memory access *or* communication.
3. **Placement decides realization.** Whether a logical route becomes local
   memory access, a kernel dispatch, or an explicit exchange is a placement
   decision (`placement.py`), never a property of the semantic expression.
4. **State mutation, collision policy, ordering, and version/commit behavior
   are explicit effects.** `effects.py` classifies them; ordered scans and
   transactional commits are movement barriers; nothing mutates implicitly.
5. **Reparameterization only through registered, verified rewrite rules.**
   Each rule declares preconditions, equivalence class, numerical envelope,
   backward status, saved-state/recomputation requirements, and traffic
   effects (`rewrite.py`). `exact` equivalence is reserved for execution
   models promising bitwise-equivalent results; rewrites that change
   floating-point operation order are `floating_point` with dtype envelopes.
6. **Performance hints and schedules must not alter semantic meaning.**
   `ScheduleParams` may pick among legal lowerings; it may never change
   routing results, merge policies, or commit boundaries. Hints are validated;
   invalid hints produce structured diagnostics, never silent reinterpretation.
7. **Existing production kernels remain valid lowering targets.** FlashAttention
   and FLA adapters are first-class anchors; wrapping upstream beats replacing
   upstream whenever overhead is measured to be low.
8. **No arbitrary tensor callback or untyped escape hatch enters the core IR.**
   Visitors and epilogues are typed descriptors interpreted by registered
   anchors - never Python callables over tensors. Serialized artifacts report
   `escape_hatch_count`; it must stay zero.
9. **Backends decline explicitly.** An anchor that cannot honor a request
   returns a structured decline with a reason code (`execution.Decline`);
   silently changing semantics or dropping work is a contract violation.
10. **NAS-facing architecture parameters remain separate from backend schedule
    parameters** in every API surface and serialized artifact
    (`planner.ArchitectureParams` vs `planner.ScheduleParams`).
11. **Compilation intent is explicit.** `CompilationIntent` (inference /
    training / forward-only analysis) gates legality: training rejects
    forward-only anchors, rewrites without certified backward, missing
    gradient coverage for operand dtypes, and unresolved recomputation or
    forward-only obligations.
12. **Candidate selection never mutates the program implicitly.** The base
    plan is always a candidate; every rewrite occurrence has a stable ID;
    callers may select explicitly; automatic selection runs through the
    solver or the documented cost heuristic; traces record the selected
    candidate and rejected alternatives.
13. **Solver models are untrusted.** Every Z3 model passes an independent,
    solver-free verifier before any plan or kernel is generated from it.
14. **Solver expressions never leak.** Z3 lives behind `compiler/solver.py`
    behind the backend-independent constraint vocabulary in
    `compiler/constraints.py`; semantic IR, execution IR, serialized
    architecture specifications, and public adapter APIs contain no solver
    objects.
15. **Push and pull communication protocols are never conflated.**
    Token-to-expert dispatch with required return (`PUSH_DISPATCH_RETURN`)
    and query-owner gather (`PULL_GATHER`) are distinct typed protocols with
    their own directions, conservation laws, and return obligations.

## What the compiler owes each compiled program

A successful compilation produces:

- an executable plan (deterministic, serializable);
- a deterministic trace: rules considered/accepted/rejected with reasons,
  chosen anchors, estimated costs, and remaining semantic obligations;
- analytical cost features (clearly separated from measured counters);
- structured diagnostics instead of silent fallbacks when anything declines.

## Non-goals

- Replacing upstream kernel families that already meet their gates.
- Arbitrary kernel synthesis from tensor programs (no autotuned search over
  unchecked code).
- Hiding distributed execution inside ordinary tensor ops: remote exchange is
  always a first-class effect and plan step.
