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
| `src/urm/frontend/` | Declarative model specification (`MixerSpec`) |
| `src/urm/compiler/` | Semantic IR, effects, rewrites, legality, planning and verification |
| `src/urm/compiler/anchors/` | Bind compiler-selected plans to executable implementations |
| `src/urm/runtime/` | Backend protocols and explicit runtime registry |
| `src/urm/backends/`, `src/urm/adapters/` | Native implementations and external-library boundaries |
| `src/urm/oracles/` | NumPy correctness references, including sparse-slot algebra and VJP |
| `src/urm/experimental/`, `src/urm/experiments/` | Uncertified prototypes, outside production selection |
| `tests/`, `benchmarks/` | Contract regressions and maintained acceptance harnesses |
| `results/` | Retained acceptance evidence and provenance consumed by regressions |
| `archive/` | Historical reports, exploratory measurements and superseded demonstrations |

`urm.ir`, `urm.backend`, and `urm.reference` remain compatibility imports.
Existing semantic IR and executable binders retain their public paths. The
distribution name `urm-kernel-lab` is retained for installation compatibility.
Sparse routed delta-update implementations use the architecture-neutral name
`sparse_delta`. Legacy `dual_form_sdm` paths are compatibility aliases only and
are not advertised in the production backend catalog.

## Implementation scope

The first three lowering families are softmax attention, linear/delta recurrence,
and sparse-slot memory. These are engineering boundaries, not a proof of universal
coverage or a promise of three universal kernel source files.

- [Runtime and lowering contracts](docs/runtime/execution.md)
- [Unified mixer kernel compiler prototype](docs/compiler/unified-mixer.md)
- [Coverage and implementation milestones](docs/planning/lowering-roadmap.md)
- [Verified sparse-slot formulation](docs/kernels/sparse-delta.md)
- [Compiler architecture](docs/compiler/architecture.md)
- [Kernel generation](docs/compiler/kernel-generation.md)
- [Compiler acceptance requirements](docs/validation/acceptance.md)
- [Archive index and migration manifest](archive/README.md)

The experimental dual-form implementations have known decay, backward and
cross-block correctness defects. The NumPy formulation is independently checked;
it is not certification of those GPU implementations or of BF16 reassociation.
Historical MFU claims do not establish current end-to-end performance.

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
