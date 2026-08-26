# Kernel-generation guide

**Status:** normative for `src/urm/compiler/`; the routed-epilogue schedule
integration and measurement-validation tranche is complete. Every subsequent
compiler change must keep this pipeline honest; deviations are contract
violations, not optimizations.

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

### Integration closure: anchor-owned schedule truthfulness

Stages 2 through 11 run INSIDE `UrmCompiler.compile()` for every program with
routed-reduction work, orchestrated by `compiler/planner.py` and
`compiler/search.py` (`CompilationSearch` -> serializable `ScheduleDecision`):

- **Anchor-first lowering identity and override consumption:** Effective candidate
  and anchor identity are resolved before schedule search begins. Compatible explicit
  overrides (e.g. `{"reduce": "routed_reduction_row_scale_epilogue_v0"}`) compile
  successfully, while unconsumed explicit overrides (rewritten-away operations,
  interior gathers, non-anchorable operations) and missing semantic inputs fail closed
  with structured `SCHEDULE_HINT_INVALID` or `ANCHOR_DECLINED` diagnostics.
- **Anchor capability single source of truth:** Schedulable `ExecutionAnchor`
  contracts define the single source of truth for schedule domains (`supported_blocks`,
  `supported_warps`, `supported_stages`, `supported_decompositions`, `supported_schedules`,
  `supported_plan_kinds`). Schedulable anchors must be complete and fail closed upon
  construction or model building. Allowed plans derive exclusively from the anchor's
  supported plan kinds without caller-provided escape hatches. Unscheduled lowerings
  (such as base `routed_reduction_v1`) reject tuning knob hints with `SCHEDULE_HINT_INVALID`
  and return `schedule_decision=None` and `launch_config=None`.
- **Explicit compilation-matrix probe modes:** The compilation matrix supports explicit
  modes (`--probe auto`, `--probe off`, `--probe required`). The `--probe off` mode never
  imports Torch/Triton and produces backend-independent CPU results, while `--probe required`
  fails immediately without CUDA and verifies all schedulable decisions with GPU resources.
- **Exact specialization compile probe:** `CompileProbe` accepts `CompileContext`
  exclusively without exception-driven legacy fallbacks. Probing compiles and launches
  the exact compile-time specialization constants (operand dtypes, route width, value
  dimension, launch configurations) across forward and backward passes under training,
  while distinguishing compile-time specializations from bounded representative runtime
  extents (Q/S). Probe exceptions fail closed into structured nogoods and retries.
- **Per-kernel resource evidence preservation:** Forward and backward compiled handles
  yield per-kernel resource records (`forward`, `grad_weights`, `grad_values`,
  `grad_row_scale`) that survive `ScheduleAttempt`, `ScheduleDecision`, and serialized
  JSON artifacts, preserving genuine zero shared memory versus unavailable metadata.
- **Exact schedule domain:** `per_route/full_row` is excluded from legal
  schedule domains (both the reference model and Z3 formulation) because
  `per_route` is segmented across program instances by construction. Stage counts
  are strictly bounded to `{1, 2, 4}`.
- **Dispatch equivalence evaluation:** Decoded `ExecutablePlan` launch configurations
  are asserted equal to direct configurations and evaluated through repeated batched
  launches with paired randomized sampling to minimize microsecond timer noise.
- **Cross-run exploratory stability and noise-aware regret:** Multiple independent
  fresh-process runs (`routed_epilogue_stability.py`) capture pre/post GPU
  operating conditions via read-only `nvidia-smi` queries, preserve raw
  CUDA-event samples, evaluate pairwise Spearman rank correlations and top-5
  Jaccard overlap, and classify solver/heuristic regret as `pass`, `fail`, or
  `inconclusive` using bootstrap confidence intervals against the 10% target.
  Its marginal-CI candidate set is explicitly exploratory and is not a
  statistical-equivalence decision.
- **Paired confirmation:** `routed_epilogue_confirmation.py` freezes the
  discovery shortlist/reference before fresh measurement, randomizes paired
  AB/BA blocks, records full-precision raw samples and per-child provenance,
  rejects persistent sentinel drift, validates GPU/configuration invariants,
  and applies a deterministic hierarchical bootstrap. A schedule is confirmed
  only when its 95% upper slowdown bound is within the declared 2.5% practical
  margin. `results/compiler/solver/routed-epilogue-confirmation.json` is the
  canonical deployment decision; the committed result confirms three
  schedules and excludes the analytical solver choice.

## Invariants recap

13. **Differential correctness.** Forward and backward match eager references
    inside dtype-specific envelopes on degenerate, non-power-of-two, repeated-
    route, and zero-scale shapes before any schedule is trusted.

14. **Acceptance or structured rejection.** A compilation either produces a
    verified executable plan with trace, obligations, analytical costs, and
    solver statistics - or a structured diagnostic. Unsupported programs and
    timed-out solving return explicit diagnostics, never a silent fallback.
