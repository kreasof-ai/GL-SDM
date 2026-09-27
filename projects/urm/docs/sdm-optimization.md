# SDM: external optimization and actual CUDA baseline

SDM now keeps the public native product-key route operators and moves its
decayed-delta state schedule into
[`architectures/sdm_chunked.py`](../architectures/sdm_chunked.py).
The benchmark selects compiled PyTorch with chunk size 128. The shared-frontend
baseline uses the original pinned Meta CUDA/Triton implementation with chunk
size 64. This work changes no file in `src/urm/`; the existing complete native
K3 path remains available as `SparseDeltaMemoryLayer(execution="native")`.
Accepted non-SDM campaign records retain their original source fingerprints.
Each production pair still has identical fingerprints, configs and environment.

## What the historical 40–50% measured

[`historical.json`](../results/sdm-optimization/historical.json) reproduces the
clean historical implementation at
`dd45e66a42fbaf4e570638e175ccac0599aeddfc` from `sdm-reparam`. Its shape is
P=12, T=1024, S=4096, D=64, reads=writes=64, C=256, BF16. Timing includes
forward and reading-loss backward with initial-memory and learned operands
requiring gradients; five warmups and twenty measured iterations exclude
compilation. The machine is the same A10G, now running Torch 2.14.0+cu130.

| Implementation | Forward + backward ms | Historical 522-GFLOP proxy |
|---|---:|---:|
| Historical PyTorch eager | 31.45 | 25.1% |
| Historical PyTorch compiled | 18.64 | 42.3% |
| Original pinned CUDA, C=256 | 15.65 | 50.4% |
| Corrected PyTorch compiled, C=256 | 22.25 | 35.5% |

The old formula is `522 GFLOPs / latency / 66.166 TFLOPS`. It retains a
full-T=1024 dense-work estimate when C=256 chunking reduces the actual work.
The profiler counts **159.015 GFLOPs of GEMMs** in the historical chunked
forward/backward. At the compiled latency, that is a **12.9% GEMM-work estimate**,
not 42.3% physical utilization. The corrected implementation counts 211.109
GFLOPs of GEMMs (14.3% at its latency). These profiler counts exclude solves,
reductions, and fused CUDA/Triton operations; they are not total hardware counters
or an estimate of the CUDA implementation's FLOPs. The historical percentage
also differs from the current decoder's parameter/state estimate and 70-TFLOPS
denominator. Neither percentage can be carried between these workloads.

There is also a correctness difference. The historical chunked path ignores
`log_decay` completely; autograd reports an unused decay input. With nonzero
decay, its maximum BF16 reading/state differences from actual CUDA are
0.02246/0.24023. The corrected path reduces these to 0.000488/0.005859 on the
same inputs and produces a nonzero decay gradient. Earlier single-chunk
analytical backward code also returns zero decay gradients. Porting those
implementations unchanged would sacrifice the memory law used by GL-SDM/CSDM.

Reproduce from the repository root:

```sh
git worktree add --detach /tmp/urm-sdm-history dd45e66a42fbaf4e570638e175ccac0599aeddfc
cd projects/urm
PYTHONPATH=src:. python extra/provision_sdm_cuda.py
python extra/benchmark_sdm.py --historical-root /tmp/urm-sdm-history \
  --out results/sdm-optimization/historical.json
```

Run GPU benchmarks alone. Routes are sorted and unique, normalized read/write
weights and beta are contractive, decay is negative, and the seed is recorded.
Timing variation is expected; these are short microbenchmarks, not convergence
or original software-stack replication.

## Correct schedule and snapshot interface

The operator exposes the same partition-local sparse operands as the recurrence:

```python
from architectures.sdm_chunked import chunked_sparse_delta_memory

readings, next_snapshot = chunked_sparse_delta_memory(
    snapshot, read_indices, read_weights,
    write_indices=write_indices, write_weights=write_weights,
    values=values, beta=beta, log_decay=log_decay,
    chunk_size=128, compiled=True,
)
```

The snapshot is read-only. Both readings and the new snapshot are differentiable,
including gradients to initial memory and decay. Callers own commit/detach timing;
this is a recurrence operator, not the full GL-SDM transaction or CSDM lifecycle.
Write addresses must be unique within each token, valid and partition-local;
cross-token collisions remain ordered. Use nonpositive decay and normalized
weights/contractive gates as supplied by the SDM frontend.

Slot-local cumulative decay factors convert the recurrence into causal chunk
systems. BF16 Tensor Core products construct their coupling/read matrices;
FP32 unit-lower-triangular solves compute deltas directly, avoiding an explicit
inverse. Exponentials and saved decay derivatives stay sparse; only matmul
operands expand to slot vectors. Decay factors are centered to limit magnitude.
An explicit zero-snapshot specialization skips the first initial-memory products
only when lifecycle guarantees zero state without initial-memory gradients.

Strong decay reduces the effective chunk size to bound exponential factors; the
size-one limiting schedule uses a stable token recurrence. This decision incurs
one scalar device synchronization per attempted size outside the compiled graph.
The mathematical path retains all operand gradients. FP64 tests compare exact
readings, terminal state, and all six differentiable operand groups to an
independent serial recurrence, including decay -1000 and two linked transactions.
BF16 eager/compiled tests compare against actual pinned CUDA, including explicit
terminal-state cotangents. The pin's default TF32 matmuls have a separate numerical
budget. BF16 chunking does not reproduce storage rounding at every token; this
is a numerical schedule change, not bitwise equivalence.

Dense chunk-slot tensors still scale with P*T*S. This implementation has been
measured at S=256 and S=4096; it does not qualify arbitrarily large global banks,
adaptive routing, overlays, read-only transactions, or contention policies.
Those GL-SDM/CSDM workloads need their own capacity/bandwidth benchmarks before
choosing this schedule over the original sparse CUDA path.

## Production kernel and machine isolation

The baseline verifies the clean upstream checkout at
`183e7df809131b80ad4393741029d0f20fc3640b`. It calls
`GatedSparseMemoryWriteRead` from Meta's repository, including its real
`sparse_ip_sorted` and `warp_cooperative_gather` CUDA extensions and Triton WY
kernels. Result JSONs record source revision, loaded extension paths/hashes,
nvcc version, and absence of reference fallback. Missing prerequisites fail
explicitly; they never select a reference implementation.

The system nvcc 12.9 conflicts with the installed Torch cu130 headers.
`extra/provision_sdm_cuda.py` installs pinned CUDA 13.0 compiler/runtime/CRT/NVVM/
CCCL packages under `~/.cache/urm/sdm-cuda13` and builds the untouched upstream
sources under `~/.cache/urm/sdm-extensions`. The base Python packages, system CUDA
and driver are unchanged. `URM_SDM_CUDA_HOME` and `URM_SDM_UPSTREAM_ROOT` can select
explicit matching toolkit/pin paths.

The external adapter pads each partition separately with identity transitions,
clones the input snapshot because the pin mutates it, and disables autocast around
the pin's FP32 WY operations. The pin returns an empty second output and reconstructs
earlier memory in its workspace during backward; the adapter therefore saves a
terminal snapshot before backward. That CUDA snapshot is detached: terminal-state
cotangents use the pin's explicit `grad_final_memory` API. The PyTorch API above
supports ordinary autograd through the returned snapshot.

## Decoder campaign

The current row has P=192, T=511 next-token positions, S=256, reads=writes=8:
a different geometry from the historical microbenchmark. Both arms use the
same 94,952,448 parameters, routes/projections, nine-layer decoder, data,
8192-token batch/microbatch, optimizer, BF16, compilation policy and checkpoint
gate. The canonical ten-step measurements and memory traces live in
[`results/report.md`](../results/report.md); the accepted pre-optimization record
is retained in [`accepted-native.json`](../results/sdm-optimization/accepted-native.json).

| Decoder schedule | MFU | Tokens/s | Peak GiB |
|---|---:|---:|---:|
| Accepted native K3 | 16.1% | 19,780 | 7.04 |
| Corrected compiled PyTorch | 19.7% | 24,215 | 11.17 |
| Actual upstream CUDA/Triton | 16.9% | 20,747 | 7.80 |

The optimized decoder is 22.4% faster than the accepted native row and 16.7%
faster than the matched CUDA baseline. Both new runs have finite losses, passing
checkpoint gates, no batch fallback, and exactly flat step-end allocations over
all ten measured steps. The compiled schedule uses more peak memory; throughput
improvement does not imply an improvement in peak activation demand.

Five-step tuning found PyTorch C=128 faster than C=64; actual CUDA C=64 beat
C=128 and C=256 on this small-bank geometry. These are diagnostics, separate
from the canonical final pair; their JSONs retain their individual source hashes.
For a new diagnostic, run one candidate per process:

```sh
python extra/tune_sdm.py --execution torch-chunked --chunk-size 128 \
  --out /tmp/sdm-torch-128.json
python extra/tune_sdm.py --execution upstream-cuda --chunk-size 64 \
  --out /tmp/sdm-cuda-64.json
PYTHONPATH=src:. python -m train.sweep --rows sdm --out-dir /tmp/sdm-sweep
PYTHONPATH=src:. python -m train.upstream --rows sdm --subprocess --out-dir /tmp/sdm-upstream
```

Peak activation allocation can increase with the compiled chunk schedule; flat
step-end traces distinguish that demand from a memory leak. Neither these short
runs nor the historical proxy establishes 40–50% full-decoder MFU or long-run
training equivalence.
