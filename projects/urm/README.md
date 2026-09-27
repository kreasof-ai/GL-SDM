# URM

URM is a semantic-to-execution compiler for routed sequence models. Its core represents and lowers four mixer execution families: K1 streamed reduction, K2 compact fixed-address state, K3 indexed mutable state, and K4 exact feedback substitution over an op's own history. Model architectures compose public kernel calls with projections, convolution, FFT, MLPs and cache logic **outside** `src/urm`. The taxonomy is an organizing hypothesis, not a claim of universal GPU binaries or coverage of every combination.

Start with the [documentation index](docs/README.md). The [compiler charter](docs/charter.md) defines core boundaries; the [pipeline](docs/compiler.md) and [backend inventory](docs/backends.md) mirror the code; the [catalog](docs/catalog.md) covers the 52-row architecture registry at its HF-modeling granularities. The [benchmark page](docs/benchmark.md) documents the training-harness campaign and the [evidence policy](docs/evidence.md) distinguishes representation, reference execution, native execution, performance and complete source-model parity.

## Current boundary

The [architecture registry](train/registry.py) carries 52 rows — 51 native-tier plus `mamba1` (reference-tier charter debt). The training-harness benchmark (`train/sweep.py`, `train/upstream.py`) has measured all 51 native rows (10-step 100M-class training, A10G) with checkpoint parity, and joined 49 upstream baselines at matched granularity — see the generated [training-harness report](results_report.md). This is harness-level measurement, not complete source-model qualification; [evidence.md](docs/evidence.md) states the exact claim boundaries. The machine-readable [architecture register](benchmarks/architecture-coverage.json) tracks 80 named rows, 76 of them mixer-relevant.

## Code map

| Location | Responsibility |
|---|---|
| `src/urm/frontend/`, `ir/` | Name-agnostic typed fragments and semantic graph |
| `src/urm/compiler/` | Verified rewrites, partition, constraints, placement, cost, provider and schedule selection |
| `src/urm/runtime/` | Serialized plan binding and state sessions |
| `src/urm/backends/` | Independent NumPy/Torch references and admitted native K1/K2/K3/K4 implementations |
| `recipes/kernels/` | External declarative kernel-call fragments |
| `architectures/`, `train/` | The 52-row catalog modules and the benchmark harness |
| `benchmarks/`, `results/` | Pinned comparators, generated indexes and preserved measurements |

## CPU verification

From `projects/urm`, with the test extra installed:

```sh
python -m pip install -e '.[test]'
python -m pytest tests
```

GPU and pinned upstream comparisons require their optional dependencies and supported hardware. A passing CPU suite does not qualify native performance or full source-model behavior.
