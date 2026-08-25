# Global Liquid SDM Research Program

This repository is a research monorepo for the three independently falsifiable
proposals described in the [research program](Global_Liquid_SDM_Research_Program.pdf).

## Projects

| Project | Research focus | Relationship |
| --- | --- | --- |
| [Global Liquid SDM](projects/gl-sdm/README.md) | Global sparse memory, tied recurrent reasoning, adaptive depth, and snapshot-and-commit writes | Establishes the core memory semantics |
| [Consolidated SDM](projects/csdm/README.md) | Fast/slow overlays, wake/sleep consolidation, stability, provenance, and rollback | Builds on GL-SDM semantics |
| [Unified Routed Mixer](projects/urm/README.md) | Restricted routed-mixer IR, scheduling, and specialized kernels | Independently testable; GL-SDM is the flagship workload |

Shared interfaces, benchmark definitions, experimental controls, and result
schemas belong in [`shared/`](shared/README.md). Project-specific models,
experiments, and acceptance tests stay within their project directory.

## Dependency shape

```text
gl-sdm ───────> csdm
   │
   └──────────> urm (flagship workload)

shared <────── all projects
```

The arrow into CSDM is a semantic dependency: CSDM assumes the global address
space and transactional memory behavior established by GL-SDM. URM remains a
separate systems project so it can succeed or fail independently of the GL-SDM
architecture.

## Program sequence

1. Prove dense/small GL-SDM semantics on controlled tasks.
2. Introduce sparse addressing and measure the memory/depth trade-off.
3. Add CSDM overlays and consolidation after single-tier behavior is understood.
4. Capture real routing traces before fixing URM's kernel API.
5. Scale only after each project passes its own acceptance gate.

## Repository rule

Cross-project code should be promoted into `shared/` only when at least two
projects use the same stable contract. This keeps the proposals independently
testable and prevents an early shared abstraction from coupling their results.
