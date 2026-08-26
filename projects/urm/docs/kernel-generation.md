# Kernel-generation guide

**Status:** normative for `src/urm/compiler/`. Every compiler change must
keep this pipeline honest; deviations are contract violations, not
optimizations.

The pipeline is:

```text
semantic program
  -> rewrite candidate generation        (planner.enumerate_candidates)
  -> feasibility constraints             (constraints.py, kernel_plan.py)
  -> Z3 feasibility and diagnostics      (solver.FeasibilityPass)
  -> bounded optimization/ranking        (solver.OptimizationPass)
  -> imperative model verification       (verification.ModelVerifier)
  -> anchor/code generation              (execution.py, anchors/)
  -> compile feedback                    (kernel_plan.apply_compile_feedback)
  -> empirical benchmarking              (benchmarks/, committed artifacts)
```

## The role of Z3

Z3 proves that **symbolic side conditions are satisfiable**, and nothing else.

- It proves named constraints are simultaneously satisfiable (SAT) or not
  (UNSAT with an explainable core), or reports UNKNOWN under explicit limits.
- It selects among bounded legal candidates using analytical objectives -
  it does not discover kernels, does not replace profiling, and its cost
  model is an estimate, never a measurement.
- Z3 is an optional dependency (`pip install urm-kernel-lab[solver]`, pinned
  to exactly `z3-solver==4.15.3.0`). Core URM functionality, existing
  adapters, and semantic inspection work without it. When absent, automatic
  candidate selection falls back to the documented deterministic cost
  heuristic - recorded in the trace as `cost_heuristic`, never silently.

## Pipeline stages

1. **Semantic validation.** `SemanticProgram.build` validates name binding,
   structural rules, commit uniqueness, and rejects anything needing an
   untyped escape hatch. Compilation intent (`inference`, `training`,
   `forward_only_analysis`) is declared up front.

2. **Rewrite/equivalence candidate generation.**
   `UrmCompiler.enumerate_candidates(program, intent)` returns the immutable
   candidate list: the unfused/base plan first (`"base"`), then one stable id
   per rewrite occurrence (`rewrite:<rule>@<op>`). Enumeration never mutates
   the program. Candidates carry equivalence class (`exact` is reserved for
   bitwise-equivalent execution; reassociations of floating-point order are
   `floating_point` with dtype envelopes), backward certification,
   saved-state policy, and traffic deltas.

3. **Hard-constraint construction.** `build_constraints` encodes legality in
   the backend-independent constraint IR: Boolean choices, bounded integers,
   enumerated choices, equality/inequality over small linear expressions,
   divisibility, implication, at-most-one/exactly-one, capacity and resource
   bounds, nogoods, objectives, plus stable names, categories, human-readable
   explanations, provenance, severity, and variables for every assertion.
   Raw solver expressions never appear in semantic IR, execution IR,
   serialized artifacts, or public adapter APIs.

4. **Feasibility solving.** A standard solver pass with tracked, named
   assertions returns SAT with a model, UNSAT with an unsat core mapped to
   structured diagnostics, or UNKNOWN with reason and resource statistics.
   Timeouts, resource limits, and candidate limits are explicit
   (`solver.SolverLimits`). Diagnostic feasibility and optimization are two
   separate calls, never one opaque solve.

5. **Unsat diagnostics.** Cores map to concise messages that name the
   conflicting constraints ("No legal training plan: candidate requires
   row-scale gradient, but the selected anchor declares forward-only
   execution"). Raw formulas are never the primary diagnostic. Representative
   impossible problems live in `unsat_catalog.py` and their mapped cores are
   committed in `results/compiler/solver/unsat-diagnostics.json`.

6. **Cost estimation.** Analytical only: useful FLOPs, logical/estimated
   physical bytes, launches, temporaries, communication path estimates.
   Estimates rank candidates; they do not replace measurement and are never
   presented as measurements.

7. **Candidate and schedule selection.** Explicit selection always wins.
   Automatic selection runs through the solver's lexicographic objectives:
   zero unresolved obligations, minimum peak temporary memory, minimum
   communication critical path, minimum communication bytes, minimum
   materialization bytes, minimum launch count, minimum analytical runtime,
   then a deterministic stable-ordering tie-break. Without the solver extra,
   the deterministic cost heuristic applies and says so. Schedule parameters
   (`ScheduleParams`) tighten the space legally: invalid hints produce
   structured `schedule_hint_invalid` diagnostics; valid hints demonstrably
   alter selected plans.

8. **Imperative verification of the solver model.** Every SAT model is
   untrusted until `verification.ModelVerifier` re-checks it without any
   solver: variable ranges, divisibility, allowed sets, implications,
   exactly-one/at-most-one, capacity bounds, nogoods, plus domain facts -
   anchor capability compatibility, dtype/layout support, training/backward
   compatibility, shared-memory limits, declared register/resource estimates,
   locality requirements, effect barriers, placement ownership, communication
   conservation, capacity, stable-order requirements, collision/merge
   policies, and transaction commit obligations. Verification failure rejects
   the model outright: record the error, optionally add one bounded exact
   nogood, request another schedule - and never generate or run a kernel from
   the rejected model.

9. **Backend anchor selection.** Trusted anchors with typed capability
   contracts (`forward_only`, `backward_verified_dtypes`,
   `honored_obligations`, determinism) answer typed visitor requests and
   decline explicitly. Training compilations reject forward-only anchors,
   anchors whose backward is not verified for the operand dtypes, and any
   unresolved recomputation/forward-only obligation.

10. **Code generation or upstream dispatch.** Generated anchors (the routed
    epilogue family) execute compiler-owned Triton kernels that have passed
    differential gates; upstream families dispatch through FA/FLA adapters.
    There is no third path: no unchecked synthesis, no silent fallbacks.

11. **Compilation and resource feedback.** When available, compile feedback
    (registers/thread, shared memory bytes, success/failure) is captured. A
    failed compilation adds an exact nogood for that schedule and requests
    another; nogoods are bounded (`SolverLimits.max_nogoods`). Exhausted
    budgets produce explicit diagnostics.

12. **Empirical autotuning.** Measured medians/p95 on committed shapes decide
    performance questions. The measured-best schedule is reported next to the
    solver-selected one so analytical regret stays visible.

### Integration closure (normative since the schedule-search iteration)

Stages 2 through 11 run INSIDE `UrmCompiler.compile()` for every program with
routed-reduction work, orchestrated by `compiler/search.py`
(`CompilationSearch` -> serializable `ScheduleDecision`):

- The schedule model is CANDIDATE-BOUND: `kernel_plan.plan_kinds_for_candidate`
  pins the execution plans each selected candidate's lowering implements, so
  candidate selection and schedule selection can never contradict each other.
- Without Z3, the deterministic heuristic optimum is lifted to a complete
  assignment (`schedule_point_to_assignment`) and passes the SAME independent
  verifier; the fallback is recorded in the decision.
- Every unverified assignment is rejected before lowering or probing.
- A compile probe (GPU callers inject one, e.g.
  `make_triton_compile_probe()`) exercises the EXACT selected configuration;
  failures add an exact nogood for THAT assignment and the search re-solves
  within `SolverLimits.max_nogoods`. Without a probe the decision records
  `compile_status=not_probed` and never claims compile success.
- The decision's launch configuration is serialized into every
  anchor-dispatch `PlanStep` and into `CompilationResult.schedule_decision`;
  the production anchor (`RoutedEpilogueLaunchConfig`) honors every field,
  and benchmarks execute those same production implementations.

## Invariants recap

13. **Differential correctness.** Forward and backward match eager references
    inside dtype-specific envelopes on degenerate, non-power-of-two, repeated-
    route, and zero-scale shapes before any schedule is trusted.

14. **Acceptance or structured rejection.** A compilation either produces a
    verified executable plan with trace, obligations, analytical costs, and
    solver statistics - or a structured diagnostic. Unsupported programs and
    timed-out solving return explicit diagnostics, never a silent fallback.

### Integration closure (normative since the schedule-search iteration)

Stages 2 through 11 run INSIDE `UrmCompiler.compile()` for every program with
routed-reduction work, orchestrated by `compiler/search.py`
(`CompilationSearch` -> serializable `ScheduleDecision`):

- The schedule model is CANDIDATE-BOUND: `kernel_plan.plan_kinds_for_candidate`
  pins the execution plans each selected candidate's lowering implements, so
  candidate selection and schedule selection can never contradict each other.
- Without Z3, the deterministic heuristic optimum is lifted to a complete
  assignment (`schedule_point_to_assignment`) and passes the SAME independent
  verifier; the fallback is recorded in the decision.
- Every unverified assignment is rejected before lowering or probing.
- A compile probe (GPU callers inject one, e.g.
  `make_triton_compile_probe()`) exercises the EXACT selected configuration;
  failures add an exact nogood for THAT assignment and the search re-solves
  within `SolverLimits.max_nogoods`. Without a probe the decision records
  `compile_status=not_probed` and never claims compile success.
- The decision's launch configuration is serialized into every
  anchor-dispatch `PlanStep` and into `CompilationResult.schedule_decision`;
  the production anchor (`RoutedEpilogueLaunchConfig`) honors every field,
  and benchmarks execute those same production implementations.

## Invariants recap

- Z3 proves side conditions; it does not generate kernels and does not replace
  profiling.
- Cost estimates rank; measurements decide.
- Solver models are untrusted until independently verified.
- Compile feedback may add a bounded nogood and request another schedule.
- Every search has time, resource, and candidate limits.
- Architecture parameters and schedule/placement variables live in separate
  namespaces in every API surface and serialized artifact.
