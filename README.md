# Global Liquid SDM Research Program

This repository contains Global Liquid SDM and Consolidated SDM, two of the
three proposals in the research program. The URM compiler and kernels live in
the separate [kreasof-ai/urm repository](https://github.com/kreasof-ai/urm).

## Documentation

- [Research program - Markdown edition](docs/research-program.md)
- [Research program - archival PDF](archive/Global_Liquid_SDM_Research_Program.pdf)

## Projects

| Project | Research focus | Relationship |
| --- | --- | --- |
| [Global Liquid SDM](projects/gl-sdm/README.md) | Global sparse memory, tied recurrent reasoning, adaptive depth, and snapshot-and-commit writes | Establishes the core memory semantics |
| [Consolidated SDM](projects/csdm/README.md) | Fast/slow overlays, wake/sleep consolidation, stability, provenance, and rollback | Builds on GL-SDM semantics |

Shared interfaces, benchmark definitions, experimental controls, and result
schemas belong in [`shared/`](shared/README.md). Project-specific models,
experiments, and acceptance tests stay within their project directory.

## Frozen URM dependency

The [GL-SDM baseline suite](projects/gl-sdm/README.md#models-and-experiments)
uses ordinary PyTorch SDPA, upstream CUDA SDM, and FLA GDN2, independently of URM.
The [GL-SDM model](projects/gl-sdm/src/gl_sdm/global_model.py) has a tied reasoner,
adaptive or fixed depth and one global memory bank with snapshot-and-commit
writes. It uses the same experiment interfaces as the three baselines.
GL-SDM defines its own [transactional memory operator](projects/gl-sdm/src/gl_sdm/memory.py);
its chunk path compiles routing and snapshot reads, including backward, with
frozen URM. PyTorch compiles the tied reasoner and proposal arithmetic; GL-SDM
owns deterministic buffered commits. Token transactions remain a control. CSDM consumes the GL-SDM
memory contract and the same URM dependency.
URM source is maintained in its own repository.

The accepted baseline is pinned to commit
[`604bfdf`](https://github.com/kreasof-ai/urm/tree/604bfdf5d2c827266a32ef142ca996cc712d70f0),
also tagged `frozen-2026-09-27`. Install the core package from this repository root:

```sh
python -m pip install -r shared/requirements-urm.txt
```

The pin is in [shared/requirements-urm.txt](shared/requirements-urm.txt).
Torch/CUDA dependencies are installed separately for the chosen machine.
Changes to URM require a separate URM task and an explicit update of this pin.
The preserved benchmark report is in
[URM results](https://github.com/kreasof-ai/urm/blob/604bfdf5d2c827266a32ef142ca996cc712d70f0/results/report.md).

## Program sequence

1. Prove dense/small GL-SDM semantics on controlled tasks.
2. Introduce sparse addressing and measure the memory/depth trade-off.
3. Add CSDM overlays and consolidation after single-tier behavior is understood.
4. Capture real routing traces to guide project-owned kernels against frozen URM.
5. Scale only after each project passes its own acceptance gate.

## Repository rule

Cross-project code should be promoted into `shared/` only when at least two
projects use the same stable contract. This keeps the proposals independently
testable and prevents an early shared abstraction from coupling their results.
