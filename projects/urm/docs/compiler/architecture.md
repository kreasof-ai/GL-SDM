# Compiler architecture

Status: construction specification. The [charter](compiler-charter.md) defines
invariants; [execution contracts](../runtime/execution.md) define the runtime boundary.

## Frontend and semantic IR

Model authors declare routing, reductions, state transitions and communication.
`ir/program.py` owns the typed operation graph; `frontend/recipes.py` loads the declarative JSON recipes.
Architecture names belong in presets and external adapters. Kernels accept typed
operands and semantic parameters, not architecture names as correctness rules.

```text
route generation -> certified routes -> gather/reduce -> explicit state update
```

Route generation and state updates remain separate semantic operations even when
a lowering fuses them. Routes describe logical domains; physical addresses,
ownership and communication arise during placement. Preserve selection,
collision, ordering, read timing, mutation and commit effects explicitly.

## Transformation and planning

Registered rewrites declare preconditions, equivalence class, numerical policy,
backward obligations, saved state and traffic effects. Floating-point
reassociation is not bitwise equivalence. Arbitrary tensor callbacks do not enter
core IR. See [kernel generation](kernel-generation.md) for the complete pipeline.

`UrmCompiler` enumerates legal candidates, verifies constraints independently,
selects placement and schedules, and emits executable plans. Optional solving and
bounded schedule selection optimize implementations; they do not redefine model
semantics or prove numerical equivalence.

## Execution boundary

`compiler/select/anchors.py` owns anchor capability matching. `runtime/` binds selected
plans to executable backends; the compiler holds no GPU execution bodies. Serialized
schedules drive actual launches. `runtime/__init__.py` holds the backend protocol,
not a competing compiler.

External libraries are execution options, not semantic definitions. Physical
layout conversion, version probing and library flags stay in adapters. Training,
prefill and decode are explicit capabilities. Unsupported requests decline.

## Initial kernel families

- Softmax attention: score transformation, masks, normalization and reduction.
- Linear/delta recurrence: declared state transitions, decay and delta correction.
- Sparse routed updates: supplied routes, reads and explicit state mutation.

Families may contain multiple algorithms and hardware variants. The
[sparse-delta formulation](../kernels/sparse-delta.md) specifies one reusable update
rule, not a universal interpretation of sparsity. Additional rules require
verified mappings.

## Construction boundaries

Keep frontend semantics independent of PyTorch, Triton and provider APIs. NumPy
oracles validate equations; framework adapters own autograd integration; providers
own compilation and loading. Hardware specialization preserves the declared
operation and numerical envelope.

Build one frontend-to-runtime vertical slice per family, then complete shared
serialization, caching, diagnostics and provider integration. Follow the
[roadmap](../planning/lowering-roadmap.md) and
[acceptance requirements](../validation/acceptance.md). Existing implementations
are building blocks; historical milestone completion does not certify new paths.

The [generality-axis contract](generality-axes.md) describes typed extensions for
state bundles, transition factors, parameter/expert/depth axes, routing and inner
updates. These extend the semantic compiler without embedding model names in kernels.
