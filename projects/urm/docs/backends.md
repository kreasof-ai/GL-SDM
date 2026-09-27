# Backend inventory

Mirrors `src/urm/backends/`: `<tier>/<family>/<op>.py`. Tiers: `numpy` and `torch`
are independent reference implementations; `triton` holds the admitted native
kernels. The op inventory equals the IR `SemanticOp` inventory (charter, backend
admission).

## Ops by family

| Family | numpy | torch | triton (native) |
|---|---|---|---|
| K1 streamed reduction | `softmax` | `softmax`, `merge` | `online_softmax`, `indexed`, `channel_decay`, `map_normalize`, `squared_sum`, `threshold_relu` |
| K2 compact fixed-address state | `linear_delta`, `dyadic_banks` | `linear_delta`, `dyadic_banks` | `diagonal`, `dyadic_banks`, `matrix_scan` |
| K3 indexed mutable state | `sparse_state` | `sparse_state` | `sparse_state`, `route_generation` |
| K4 exact feedback substitution | `triangular_solve` | `triangular_solve` | `triangular_solve` |

Every triton op has a certified forward; training-tier ops additionally have a
certified backward. Reference tiers (numpy/torch) are the parity oracles the
`tests/test_architectures_*.py` gates compare native output against.

## Numerical policies

- **Relaxed-atomic backward.** The native indexed-K1 backward
  (`triton/k1/indexed.py`) and the K3 backward reduce dK/dV through relaxed
  `tl.atomic_add`; summation order is scheduler-dependent *by design* (serializing
  the scatter would destroy the parallelism the kernel exists for). Rows on these
  kernels carry `MixerSpec.atomic_backward=True` and the checkpoint gate certifies
  them by loss-trajectory match + bitwise round-trip structure rather than bitwise
  parameter equality (see [benchmark](benchmark.md#gates)).
- **fp32 accumulation.** Native kernels accumulate in fp32 regardless of operand
  dtype; the harness trains under bf16 autocast.

## Hardware envelopes measured on this A10G (22 GiB, 101376 B shared memory)

- The K1 `map_normalize` native kernel (pattention/tokenformer) fits slot counts
  ≤ 128 at width 768; 256 exceeds the 101 KB shared-memory limit. The tokenformer
  block's slot counts are set inside the envelope (`train/registry.py`).
- The chunked-K2 `matrix_scan` state history scales as `tokens × heads × 2145 × 64`
  fp32 elements; `based_attention` (12× score recompute at DV=768) fits only at
  the 1024-token microbatch rung (see the sweep fallback ladder).
- Upstream comparison kernels that exceed this GPU: fla's `log_linear_attn` chunk
  (122 KB) and `log_linear_mamba2` chunk (196 KB) shared memory, fla's RWKV7 chunk
  (131 KB), the pinned Tucker fused kernel (294 KB) — all recorded as
  environment-blocked or run as reference implementations in the upstream arm
  ([catalog](catalog.md), [report](../results_report.md)).
- The lingua SDM sparse inner-product core is a `load_inline` CUDA extension;
  this environment's toolchain is mismatched (nvcc 12.9 vs cu13 headers, no toolkit
  include path) and source builds are excluded by policy — the sdm upstream baseline
  runs the pinned law in torch.

## Compile behavior

Native rows compile through the opaque-op boundary. The upstream comparison arm
runs eager: the fla chunk kernels fail `torch.compile`/Inductor in this
environment (measured), so the upstream side pays only the unfused-surround
penalty — its mixer kernels are already fused Triton.
