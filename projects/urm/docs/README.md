# URM documentation

This tree documents the code **as it exists at HEAD** — not plans. Anything here that
disagrees with the code is a bug in the docs; file an issue or fix the doc.

| Page | Mirrors | Contents |
|---|---|---|
| [charter.md](charter.md) | `src/urm/` invariants | The normative contract the compiler/runtime/backends enforce: ownership, the closed K1–K4 taxonomy, semantic invariants, backend branch admission |
| [compiler.md](compiler.md) | `src/urm/frontend`, `ir`, `compiler`, `runtime` | The pipeline from a typed graph fragment to a bound executable plan, and the opaque-op invocation path |
| [backends.md](backends.md) | `src/urm/backends/` | The op inventory per family/tier, the native Triton kernels, numerical policies (relaxed atomics), hardware envelopes measured on this A10G |
| [catalog.md](catalog.md) | `architectures/`, `train/registry.py` | The 52-row architecture registry: granularity taxonomy, tiers, upstream mapping per row |
| [benchmark.md](benchmark.md) | `train/`, `results/sweep/`, `results/upstream/` | The training-harness benchmark: config, drivers, gates, protocol, and how to reproduce |
| [sdm-optimization.md](sdm-optimization.md) | Native K3 chunk scheduling and actual pinned CUDA baseline | Historical MFU reproduction, compiler selection, state/VJP correctness and performance limits |
| [evidence.md](evidence.md) | `tests/`, `extra/` | The claim policy (five independent verdicts) and what the current evidence supports |

Top-level: [project README](../README.md), [agent notes](../AGENTS.md),
[training-harness report](../results/report.md).
