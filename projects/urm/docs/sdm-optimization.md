# SDM: native K3 optimization and actual CUDA baseline

`architectures/sdm_memory.py` now executes its complete public URM graph:
two native route operators and the existing native K3 decayed-delta state
operator. `architectures/sdm_chunked.py` has been deleted. Chunking is a physical
schedule of that operator, selected by the compiler from typed properties and
recorded in the executable plan. There is no SDM-specific Torch backend branch;
the Torch recurrence remains an independent reference.

The exception to the backend freeze is limited to this generic K3 integration
and its compiler/runtime plumbing. Other accepted campaign rows and their
artifacts retain their original source fingerprints. Each production pair still
requires identical fingerprints, configs and environment.

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

## Native schedule and state interface

The compiler registers `chunked_decayed_delta_state` as a floating-point physical
reparameterization, with BF16 forward and all-state/operand backward evidence.
It preserves the existing semantic node, ordered collisions, route certification,
state layout, and read timing. The serialized plan names the rule, chunk cap,
workspace limit, runtime guards, and scan base. Runtime binding rejects missing
or altered schedule metadata before dispatch.

A supplied-route client can use the public interface without an SDM frontend:

```python
from urm.compiler.pipeline import CompilationIntent, ScheduleParams, compile_graph
from urm.ir.program import DType, SparseStateExecutionMode, sparse_state_mixer_program

program = sparse_state_mixer_program(
    name="supplied_route_cache", parallel=2, sequence=41,
    slots_per_partition=67, value_dim=37, writes=3, reads=2,
    dtype=DType.BFLOAT16, mode=SparseStateExecutionMode.TRAINING,
)
plan = compile_graph(program, target="native", intent=CompilationIntent.TRAINING)
result = plan.execute(
    memory=snapshot, read_addresses=read_indices, read_weights=read_weights,
    write_addresses=write_indices, write_weights=write_weights,
    values=values, beta=beta, log_decay=log_decay,
)
readings, next_snapshot = result["readings"], result["updated_memory"]

# A reviewable choice of the original scan through the same public API:
scan_plan = compile_graph(
    program, target="native", intent=CompilationIntent.TRAINING,
    schedule_params=ScheduleParams(sparse_state_schedule="scan"),
)
```

During autograd, the input snapshot remains read-only and both outputs are
differentiable, including initial-memory and decay gradients. Callers own
commit/detach timing. Without gradients, the existing persistent in-place state
ABI and preallocated-output behavior are preserved. Both before-update and
after-update reads are supported. Write addresses remain certified, unique
within a token, valid and partition-local; cross-token collisions stay ordered.

Slot-local cumulative decay factors convert the recurrence into causal chunk
systems. BF16 Tensor Core products construct coupling/read matrices; FP32
unit-lower-triangular solves compute deltas directly. Exponentials and their
saved derivatives stay sparse; only matmul operands expand to slot vectors.
Centering bounds exponential magnitude. The zero-state specialization requires
an actual zero snapshot with no initial-memory gradient; the native provider
checks those facts rather than relying on an architecture lifecycle flag.

Automatic chunk selection requires a BF16 training update, at most 4096 slots,
and a workspace estimate no larger than `64 * 1024**2` elements. The compiler caps
chunks at 128 tokens. Runtime P/T dimensions are checked again. Nonpositive
decay and beta in [0,1] are required; actual slot decay sums reduce the chunk
size until exponential factors are bounded. The decay-bound checks incur scalar
device synchronizations outside the compiled numerical graph. Chunks shorter
than 32 tokens, oversized banks/workspace, inference, read-only access, FP32,
unsafe gate/decay values, and exhaustion of the bounded compiled-shape cache use
the existing ordered scan/read kernels. The cache has a per-function budget on
Torch versions that expose it; no process-global compiler limit is changed.
A forced `chunked` hint cannot bypass the typed eligibility check.

FP64 algebra tests compare exact readings, terminal state and all six operand
groups against an independent serial recurrence, including decay -1000 and
linked transactions. Native BF16 public-plan tests compare both read timings,
terminal-only losses and continuation against the unchanged Torch oracle.
An independent supplied-route cache client uses nonsquare banks, unequal route
widths and a read-only terminal probe. Actual pinned CUDA tests cover autocast,
partition-local padding, all operand gradients and terminal-state lifetime.
The registered BF16 forward budget is atol .02 / rtol .03, and relative gradient
norm error is below .04 in these gates. Chunking changes per-token BF16 storage
rounding; this is floating-point equivalence, not bitwise identity.

Dense chunk-slot tensors scale with P*T*S. The bounded admission policy retains
sparse scans for large global banks. These gates qualify the recurrence and
its physical schedule; GL-SDM/CSDM capacity, adaptive routing, transaction,
overlay and contention policies still require their own workload measurements.

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
cotangents use the pin's explicit `grad_final_memory` API. The native autograd API above
supports ordinary autograd through the returned snapshot.

## Decoder campaign

The current row has P=192, T=511 next-token positions, S=256, reads=writes=8:
a different geometry from the historical microbenchmark. The unchanged model
FLOP estimate is `6 * parameter_count * tokens` plus the state-work term, divided
by full-step time and 70 TFLOPS; it is not the historical 522-GFLOP proxy. Both arms use the
same 94,952,448 parameters, routes/projections, nine-layer decoder, data,
8192-token batch/microbatch, optimizer, BF16, compilation policy and checkpoint
gate. The canonical ten-step measurements and memory traces live in
[`results/report.md`](../results/report.md); the accepted pre-optimization record
is retained in [`accepted-native.json`](../results/sdm-optimization/accepted-native.json).

| Decoder schedule | MFU | Tokens/s | Peak GiB |
|---|---:|---:|---:|
| Accepted native K3 before optimization | 16.1% | 19,780 | 7.04 |
| Current public native scan (control) | 16.6% | 20,408 | 7.04 |
| Current public native chunks | 19.4% | 23,768 | 11.16 |
| Actual upstream CUDA/Triton | 16.9% | 20,726 | 7.80 |

The chunked native decoder is 16.5% faster than the current public native scan,
20.2% faster than the accepted pre-optimization row, and 14.7% faster than the
matched CUDA baseline. The scan control uses the same current projections,
public graph, data, configuration and FLOP numerator; only the compiler's state
schedule preference changes. Its result is a separate
[`native-scan.json`](../results/sdm-optimization/native-scan.json) diagnostic,
not a replacement production arm. The native/CUDA canonical pair and scan
control share source fingerprint `487a6ddbd7320d01401f83a87c998c6ac2b779a9e65dad87f7114042d3cdac8f`.

All three runs have finite losses, passing checkpoint gates, no batch fallback,
and exactly flat step-end allocations over ten measured steps. Chunking uses
more peak memory; throughput improvement does not imply lower activation demand.
The current native loss trace equals the preceding external chunk implementation's
trace on this dataset/seed. That preceding 19.7% result is archived in
[`external-before-integration.json`](../results/sdm-optimization/external-before-integration.json);
it is historical evidence and no longer feeds the production report.
[`native-plans.json`](../results/sdm-optimization/native-plans.json) records the
complete three-op SDM plan and a supplied-route client's native plan, both using
the same physical schedule contract. The validation log records 193 passed
checks and one skip.

Earlier five-step tuning found PyTorch C=128 faster than C=64; actual CUDA C=64 beat
C=128 and C=256 on this small-bank geometry. These are diagnostics, separate
from the canonical final pair; their JSONs retain their individual source hashes.
For a new diagnostic, run one candidate per process:

```sh
python extra/tune_sdm.py --execution native --state-schedule auto \
  --out /tmp/sdm-native.json
python extra/tune_sdm.py --execution native --state-schedule scan \
  --out /tmp/sdm-scan.json
python extra/tune_sdm.py --execution upstream-cuda --chunk-size 64 \
  --out /tmp/sdm-cuda-64.json
PYTHONPATH=src:. python -m train.sweep --rows sdm --out-dir /tmp/sdm-sweep
PYTHONPATH=src:. python -m train.upstream --rows sdm --subprocess --out-dir /tmp/sdm-upstream
```

Peak activation allocation can increase with the compiled chunk schedule; flat
step-end traces distinguish that demand from a memory leak. Neither these short
runs nor the historical proxy establishes 40–50% full-decoder MFU or long-run
training equivalence.
