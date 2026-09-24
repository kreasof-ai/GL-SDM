# URM

URM is a semantic-to-execution compiler for routed sequence models. Its core aims to represent and lower three mixer execution families: K1 streamed reduction, K2 compact fixed-address state, and K3 indexed mutable state. Model architectures compose public kernel calls with projections, convolution, FFT, MLPs and cache logic **outside** `src/urm`. Three families are an organizing hypothesis, not a claim of three universal GPU binaries or current coverage of every combination.

Start with the [documentation index](docs/README.md). The [compiler charter](docs/compiler/compiler-charter.md) defines core boundaries; the [roadmap](docs/planning/roadmap.md) gives the implementation order; the [76-architecture ledger](docs/planning/architecture-composition.md) gives proposed external call graphs and explicit blockers. [Evidence rules](docs/validation/evidence.md) distinguish representation, reference execution, native execution, performance and complete source-model parity.

## Current boundary

The public graph catalog has 14 dense-softmax K1 fragments, one route-to-state K3 fragment and no executable K2 graph recipe. `architectures/` and `inference/` are not yet complete model applications. Preserved kernel-slice comparisons and the old generic decoder table are historical evidence, not present source-model coverage. The machine-readable [architecture register](benchmarks/architecture-coverage.json) tracks 80 named rows, 76 of them mixer-relevant.

## Code map

| Location | Responsibility |
|---|---|
| `src/urm/frontend/`, `ir/` | Name-agnostic typed fragments and semantic graph |
| `src/urm/compiler/` | Verified rewrites, partition, constraints, placement, cost, provider and schedule selection |
| `src/urm/runtime/` | Serialized plan binding and state sessions |
| `src/urm/backends/` | Independent NumPy/Torch references and admitted native K1/K2/K3 implementations |
| `recipes/kernels/` | External declarative kernel-call fragments |
| `architectures/`, `train/`, `inference/` | Model modules and applications, currently incomplete |
| `benchmarks/`, `results/` | Pinned comparators, generated indexes and preserved measurements |

## CPU verification

From `projects/urm`, with the test extra installed:

```sh
python -m pip install -e '.[test]'
python -m pytest tests
```

GPU and pinned upstream comparisons require their optional dependencies and supported hardware. A passing CPU suite does not qualify native performance or full source-model behavior.
