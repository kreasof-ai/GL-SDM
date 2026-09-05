# Selected-slot-decay numerical experiment

This experiment is isolated from all production backends and compiler anchors.
The previously referenced formulation was not present in the supplied checkout;
the implementation states the selected-slot-decay triangular equations explicitly
so that the algebra and cast policy are reviewable. It does not alter the frozen
model, the existing per-token-cast oracle, or any acceptance tolerances.

For a token `t`, let `k_t` be its sparse write-weight vector and let `E_t` be a
diagonal matrix containing `exp(g_t)` at selected write slots and one elsewhere.
The selection mask is independent of whether a selected weight happens to be
zero. Without state rounding, the recurrence is

```
M_t = E_t M_(t-1) + k_t d_t
d_t = beta_t (v_t - k_t^T E_t M_(t-1))
```

Within a chunk starting from memory `M_0`, define
`G_i = E_i ... E_1`, and `H_ij = E_i ... E_(j+1)` for `j < i`.
The strictly lower triangular matrix has entries
`A_ij = k_i^T H_ij k_j`. Solve

```
(I + diag(beta) A) d = beta * (v - [k_i^T G_i M_0]_i).
```

Reconstruct any after-update state using
`M_i = G_i M_0 + sum_(j<=i) H_ij k_j d_j`, with `H_ii = I`.
Before-update queries use `M_(i-1)`. Final memory uses the same reconstruction
at the last token. Chunk boundaries propagate in order, including partial chunks.

The implementation computes interval products from differences of cumulative
per-slot **log** decay. It never divides by cumulative decay or constructs
`exp(-prefix)`, which could overflow under strong decay. It uses transparent
PyTorch scatter, reductions, matrix multiplies, and a triangular solve.

The essential numerical limitation is explicit: the candidate casts readings
and chunk boundary memory, but has no per-token state cast inside a chunk.
The oracle still performs `cast(E_t M_(t-1) + k_t d_t)` every token. A rounding
residual `e_t = cast(M_t_unrounded) - M_t_unrounded` would contribute to all
later states and to the right-hand side of the solve. Those residuals depend
on the preceding rounded states; omitting them is a numerical hypothesis to
test, not an exact implementation of the frozen recurrence. The counterexample
test keeps the discrepancy visible rather than changing the oracle or tolerances.

Validation covers chunks 16/32/64, FP32 and BF16, repeated collisions, disjoint
routes, mixed overlaps, and a small-update rounding stress case. Update cases
retain 4096 slots, D=64, and 64/64 route widths. A partition can have at most
64 fully disjoint width-64 tokens in 4096 slots, so those cases stop at 64;
repeated/mixed cases exercise longer sequences and ordered chunk boundaries.
Every update fixture starts with nonzero memory and uses a loss depending on
both readings and final memory. Each of initial memory, write weights, values,
beta, log decay, and read weights has its VJP checked separately against each
reference. Read-only CURRENT_STATE cases check their two differentiable inputs.
FP64 CPU tests independently check the unrounded algebra and all six VJPs.

The pinned upstream comparator keeps the model's chunk size 16 regardless of
candidate chunk size. Calls are made separately for each partition to avoid
its flattened-P*T assumption when a partition has a partial chunk. The pinned
implementation also rejects an unpadded partial chunk even for P=1 (T=35 fails
its last output view into `[1,16,64]`). Comparator-only padding extends calls to
a multiple of 16 with zero write weights, beta, log decay, values, and query
weights; these updates are identity operations on memory. Dummy readings are
trimmed, original losses retain their original normalization, and final-state
cotangents are preserved. Real candidate/oracle sequences are never padded.
The artifact records each comparator's physical length and this policy. Its final
state is snapshotted before backward restores its mutable working memory, and
the final-memory cotangent is injected through its explicit API. Before-update
readings use shifted queries (query `t` after write `t-1`) and a direct initial
memory read for token zero. This is comparator glue, not an upstream kernel edit.
The artifact reports candidate/oracle and candidate/upstream errors separately,
and also records upstream/oracle differences as diagnostics. A failure against
either required reference rejects kernel integration.

The CPU counterexample already rejects all three chunk sizes at the frozen BF16
forward gate. With 64 equal write/read weights, selected memory initially one,
zero values, beta 0.001, log decay -0.001, and 129 repeated tokens, every oracle
update rounds back to one. Candidate final values for chunks 16/32/64 are
0.875, 0.87890625, and 0.87890625. Absolute errors 0.125 and 0.12109375 exceed
`atol + rtol * abs(reference) = 0.04`. The CPU algebra/gradient suite passes
13 tests, including identity-padding output/VJP checks and retained rejection
tests for these three counterexamples;
this is not numerical acceptance. Kernel integration is blocked by this result.

The current implementation is not a cost claim. Its dense slot intermediates
require O(P C^2 S) arithmetic/storage in autograd, in addition to memory/value
reconstruction and the triangular solve. A future accepted GPU formulation
would need to precompute chunk-local overlap/decay work across both partitions
and chunks, account for backward, scratch and orchestration, and retain ordered
boundary propagation. In real arithmetic, a chunk boundary transform is a
diagonal operator plus a rank-at-most-C correction over slots. Composition can
grow that rank with the number of chunks, eventually requiring a dense S-by-S
representation; state rounding additionally makes the transform nonlinear.
No bounded-cost cross-chunk scan representation is demonstrated here.

Any GPU successor must first pass these numerical checks, then demonstrate a
complete-stage speedup giving a credible corrected Amdahl ratio at or below
1.05, and finally pass the unchanged full-model acceptance grid. SDPA remains
an architectural comparison. Route-dimension vectorization of the ordered,
per-token-cast recurrence would be a distinct experiment; no such kernel is
included in this numerical prototype.

## Measured numerical result

`results/pretraining-step/triangular-numerical-v1.json` is schema-valid and records
102 comparisons on the A10G: 96 update cases and six read-only cases. Pinned
upstream passes **102/102** candidate comparisons; the per-token-cast oracle
passes **94/102**. All FP32 cases pass. The eight failures are BF16 rounding-stress
cases: T=129 for chunks 16 and 32, and T=64/129 for chunk 64, at both before- and
after-update read timings. The decision is `rejected_before_kernel_integration`.

For BF16 T=129 after-update readings, the separate forward discrepancies are:

| Chunk | Readings vs oracle | Final memory vs oracle | Readings vs upstream | Final memory vs upstream |
| --- | ---: | ---: | ---: | ---: |
| 16 | 0.125 | 0.125 | 0.00390625 | 0 |
| 32 | 0.12109375 | 0.12109375 | 0.00390625 | 0.00390625 |
| 64 | 0.12109375 | 0.12109375 | 0.00390625 | 0.00390625 |

Every differentiable-input VJP passes its existing frozen gradient tolerance
against each reference. Maximum absolute errors across the full grid are:

| Differentiable input | Against per-token-cast oracle | Against pinned upstream |
| --- | ---: | ---: |
| Initial memory | 4.053116e-6 | 3.814697e-6 |
| Write weights | 5.066395e-7 | 2.980232e-7 |
| Values | 1.192093e-7 | 1.192093e-7 |
| Beta | 6.675720e-6 | 9.536743e-7 |
| Log decay | 4.272461e-4 | 1.220703e-4 |
| Read weights | 1.602173e-4 | 3.051758e-5 |

The artifact also retains upstream-versus-oracle discrepancies. In the T=129
stress case, upstream final memory is 0.875 while the per-token-cast oracle
remains 1.0; the frozen BF16 acceptance intervals around those two references
do not overlap. Agreement with upstream therefore cannot establish the required
per-token-cast contract. This discrepancy is retained rather than resolved by
changing casts, inputs, or tolerances.

No custom GPU prototype, kernel integration, or model integration was performed.
The existing recurrence, route widths, context, model/data configuration, frozen
gates, and previous negative artifacts are unchanged. Reproduce from `projects/urm`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest tests/test_sparse_state_triangular.py
PYTHONPATH=src:/tmp/urm-sdm-upstream.wKxbSo LIBRARY_PATH=/opt/conda/lib \
python benchmarks/sparse_state_triangular.py \
  --output results/pretraining-step/triangular-numerical-v1.json
```

The numerical runner exits nonzero after saving the complete rejected artifact.
