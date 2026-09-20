# Softmax attention family

K1 construction contract and current backend boundary.
This is family 1 of the [coverage matrix](../planning/coverage.md).

## Operation

Inputs: queries `[B,Hq,Tq,Dk]`, keys `[B,Hkv,Tk,Dk]`, values
`[B,Hkv,Tk,Dv]`, explicit query-to-KV head mapping, scale and visibility mask.
For each query row, over visible keys:

```text
scores = scale * Q K^T + bias
P = softmax(scores)
Y = P V
```

Specify causal position offsets independently of tensor lengths, including cached
decode and cross-attention. Define fully masked rows to return zero with zero
input gradients. A provider with different behavior must adapt or decline.
The initial native contract excludes dropout; supporting it later requires an
explicit probability, RNG/replay contract and validated backward.

## Coverage and specialization

MHA is equal query/KV head counts; MQA shares one KV head; GQA uses an explicit
many-to-one mapping. Causal, noncausal, sliding-window and block-sparse visibility
are specializations of the mask contract. Sparse mask expressibility alone does
not establish efficient sparse execution. Cross-attention permits distinct query
and key lengths. Position transforms and compressed projection schemes are frontend
compositions only when their resulting operands match this contract exactly.

## Lowering

Start with the existing dense-attention adapter for its declared envelope. Build
an independent dense reference. The native prefill implementation tiles queries
and keys, maintaining row maximum `m`, exponential sum `l`, and weighted sum `a`:

```text
m_new = max(m, max(scores_tile))
l_new = exp(m-m_new)*l + sum(exp(scores_tile-m_new))
a_new = exp(m-m_new)*a + exp(scores_tile-m_new) @ V_tile
Y = a/l
```

Treat empty/all-masked tiles explicitly to avoid `-inf - -inf`. Accumulate softmax
statistics in declared accumulation precision. Do not materialize the full score
matrix in the optimized path. Backward can recompute probabilities from saved row
statistics and uses `dV=P^T dY`, `dP=dY V^T`,
`dScores=P*(dP-row_sum(P*dP))`, then the Q/K contractions, scale and mask rules.
Shared KV heads require correctly reduced gradients.

Decode needs a separate KV-cache ABI: ownership, valid lengths, position offsets,
append semantics and layout. Stateless attention support does not imply cache
management support. Sparse masks and short decode may need separate schedules.

## Parity route

Compare native kernels with a pinned compatible SDPA/attention implementation,
and compare URM-wrapped library dispatch with that same direct library call.
Validate Q/K/V gradients, supported bias gradients, all-masked rows, head sharing,
unequal lengths and precision extremes. Benchmark training, prefill and cached
decode separately. First match the existing adapter envelope; expand only after
the new capability passes [parity gates](../validation/parity.md).

Current native status: Triton tiled online softmax with recomputed backward is
implemented for FP32/FP16/BF16. GPU differential checks cover MHA-style and
shared-KV attention, unequal query/key lengths, boolean and additive masks,
empty rows, and score-bias gradients. On A10G BF16 B1/T64/Hq=4/D=V=32, its
CUDA-graph kernel path passes pinned FlashAttention output/gradient parity and
the 10% performance gate for MHA/MQA/GQA. Per-call Python dispatch is slower
than the upstream callable on these short cases. Cache ownership, decode
positions, dropout, sparse traversal efficiency, larger dimensions and
end-to-end layer training/inference are unqualified; see the
[measured profile](../planning/coverage.md).
