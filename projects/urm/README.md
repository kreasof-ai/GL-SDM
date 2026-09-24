# Unified Routed Mixer compiler

URM compiles routing, state, and communication semantics into verified execution
plans. Model semantics remain independent of backend kernels. The normative
rules are in [the compiler charter](docs/compiler/compiler-charter.md).

Start with the [documentation index](docs/README.md) for the reading order and
separation between compiler contracts, kernels, adapters, validation and planning.

```text
frontend specification → semantic IR → verified rewrites → planning
  → execution anchors → backend/library binding → runtime invocation
```

## Code map

| Location | Responsibility |
|---|---|
| `src/urm/frontend/` | Declarative model specification (`MixerSpec`) and the versioned JSON recipe loader |
| `src/urm/ir/` | Typed semantic IR: family ops (`k1`/`k2`/`k3`), graph, effects, program, and tensor types |
| `src/urm/compiler/` | Compiler passes in `normalize/`, `rewrite/`, `partition/`, `select/`, `placement/`, `schedule/`, `lower/`, `verify/`, `cost/`, `solve/`, plus shared diagnostics in `common/` and the orchestration `pipeline.py` |
| `src/urm/runtime/` | Plan binding, operand validation, state sessions, and the semantic-family backend registry |
| `src/urm/backends/` | Pure capability contracts (`interface.py`), independent NumPy/PyTorch reference backends (`reference/`), and native Triton K1/K2/K3 kernels (`triton/`) |
| `recipes/` | Versioned JSON kernel fragments (`kernels/`), complete model graphs (`architectures/`), and their schemas (`schema/`) |
| `architectures/` | Model-specific components that a typed reusable operation cannot represent |
| `train/`, `inference/` | Training and serving applications consuming the public frontend/runtime APIs |
| `tests/`, `benchmarks/` | Contract regressions and maintained acceptance harnesses; `benchmarks/comparators/` holds the pinned upstream comparators |
| `results/` | Retained acceptance evidence and provenance consumed by regressions |

`urm.ir` is the canonical IR module; the former `urm.backend` and `urm.reference`
wildcard compatibility shims were removed. Import `BackendRegistry` from
`urm.runtime` and the NumPy reference (`execute`, `merge_writes`) from
`urm.backends.reference.numpy`. The upstream comparators moved from the retired
`benchmarks.comparators` package to `benchmarks/comparators/`; the NumPy reference equations
moved from the retired `urm.backends.reference.numpy` package to `urm.backends.reference.numpy`.
The distribution name `urm-kernel-lab` is retained for installation compatibility.
Production selection uses the validated sparse-state mixer backend.

## Implementation scope

The first three lowering families are softmax attention, linear/delta recurrence,
and sparse-slot memory. These are engineering boundaries, not a proof of universal
coverage or a promise of three universal kernel source files.

- [Runtime and lowering contracts](docs/runtime/execution.md)
- [Unified mixer kernel compiler](docs/compiler/unified-mixer.md)
- [Coverage and remaining work](docs/planning/lowering-roadmap.md)
- [Verified sparse-slot formulation](docs/kernels/sparse-delta.md)
- [Compiler architecture](docs/compiler/architecture.md)
- [Kernel generation](docs/compiler/kernel-generation.md)
- [Compiler acceptance requirements](docs/validation/acceptance.md)
- [Master coverage table](docs/validation/master-table.md)

## CPU verification

From `projects/urm` with the project and test extras installed:

```sh
python -m pip install -e '.[test]'
python -m pytest tests
python -m pytest tests/test_sparse_slot_formulation.py
```

GPU tests require their optional dependencies and supported hardware. Compiler
imports and NumPy references require neither PyTorch nor Triton. Training support
requires complete backward coverage; inference support requires state continuity.
