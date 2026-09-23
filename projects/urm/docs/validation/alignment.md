# Gradient alignment and inference-decoding KL divergence

Status: evidence record. Gradient-alignment rows regenerate from the
committed qualification artifacts; the decoding KL divergence is a live
single-token decode-step measurement. Regenerate with
`PYTHONPATH=src:benchmarks python benchmarks/alignment_report.py`.

## Gradient alignment (native vs upstream/oracle backward)

Per-operand max-abs gradient error recorded in the committed qualification
artifacts. The mixer kernels are operations (no learned weights of their
own), so the alignment is over the input/state operand gradients - the
quantities a model's weights depend on through the chain rule. The state
gradient (the recurrent-state input) is the state-continuation path.

| Workload | Case | dtype | Operand | gradient max abs err | kind |
|---|---|---|---|---|---|
| k1-mha | latency_small | bfloat16 | `query` | 1.19e-07 | input |
| k1-mha | latency_small | bfloat16 | `key` | 4.77e-07 | input |
| k1-mha | latency_small | bfloat16 | `value` | 2.38e-07 | input |
| k1-mha | latency_small | float16 | `query` | 5.96e-08 | input |
| k1-mha | latency_small | float16 | `key` | 5.96e-08 | input |
| k1-mha | latency_small | float16 | `value` | 5.96e-08 | input |
| k1-mha | throughput_medium | bfloat16 | `query` | 3.73e-09 | input |
| k1-mha | throughput_medium | bfloat16 | `key` | 3.73e-09 | input |
| k1-mha | throughput_medium | bfloat16 | `value` | 7.45e-09 | input |
| k1-mha | throughput_medium | float16 | `query` | 5.96e-08 | input |
| k1-mha | throughput_medium | float16 | `key` | 5.96e-08 | input |
| k1-mha | throughput_medium | float16 | `value` | 5.96e-08 | input |
| k1-mha | throughput_long | bfloat16 | `query` | 9.31e-10 | input |
| k1-mha | throughput_long | bfloat16 | `key` | 1.86e-09 | input |
| k1-mha | throughput_long | bfloat16 | `value` | 1.86e-09 | input |
| k1-mha | throughput_long | float16 | `query` | 5.96e-08 | input |
| k1-mha | throughput_long | float16 | `key` | 5.96e-08 | input |
| k1-mha | throughput_long | float16 | `value` | 5.96e-08 | input |
| k1-mha | decode_step | bfloat16 | `query` | 2.38e-07 | input |
| k1-mha | decode_step | bfloat16 | `key` | 1.19e-07 | input |
| k1-mha | decode_step | bfloat16 | `value` | 2.98e-08 | input |
| k1-mha | decode_step | float16 | `query` | 5.96e-08 | input |
| k1-mha | decode_step | float16 | `key` | 5.96e-08 | input |
| k1-mha | decode_step | float16 | `value` | 5.96e-08 | input |
| k1-gqa | latency_small | bfloat16 | `query` | 1.19e-07 | input |
| k1-gqa | latency_small | bfloat16 | `key` | 9.54e-07 | input |
| k1-gqa | latency_small | bfloat16 | `value` | 3.81e-06 | input |
| k1-gqa | latency_small | float16 | `query` | 5.96e-08 | input |
| k1-gqa | latency_small | float16 | `key` | 2.38e-07 | input |
| k1-gqa | latency_small | float16 | `value` | 4.77e-07 | input |
| k1-gqa | throughput_long | bfloat16 | `query` | 4.66e-10 | input |
| k1-gqa | throughput_long | bfloat16 | `key` | 3.73e-09 | input |
| k1-gqa | throughput_long | bfloat16 | `value` | 7.45e-09 | input |
| k1-gqa | throughput_long | float16 | `query` | 0.00e+00 | input |
| k1-gqa | throughput_long | float16 | `key` | 1.19e-07 | input |
| k1-gqa | throughput_long | float16 | `value` | 1.19e-07 | input |
| k1-gqa | decode_step | bfloat16 | `query` | 2.38e-07 | input |
| k1-gqa | decode_step | bfloat16 | `key` | 4.77e-07 | input |
| k1-gqa | decode_step | bfloat16 | `value` | 5.96e-08 | input |
| k1-gqa | decode_step | float16 | `query` | 1.19e-07 | input |
| k1-gqa | decode_step | float16 | `key` | 1.19e-07 | input |
| k1-gqa | decode_step | float16 | `value` | 1.19e-07 | input |
| k1-masked-variant | block_sparse_medium | bfloat16 | `query` | 2.98e-08 | input |
| k1-masked-variant | block_sparse_medium | bfloat16 | `key` | 2.98e-08 | input |
| k1-masked-variant | block_sparse_medium | bfloat16 | `value` | 1.86e-09 | input |
| k1-masked-variant | fully_masked_rows | bfloat16 | `query` | 1.19e-07 | input |
| k1-masked-variant | fully_masked_rows | bfloat16 | `key` | 2.38e-07 | input |
| k1-masked-variant | fully_masked_rows | bfloat16 | `value` | 5.96e-08 | input |
| k2-diagonal-recurrence | latency_short | float32 | `x` | 1.49e-08 | input |
| k2-diagonal-recurrence | latency_short | float32 | `log_decay` | 2.98e-08 | input |
| k2-diagonal-recurrence | latency_short | float32 | `initial_state` | 9.31e-09 | state |
| k2-diagonal-recurrence | latency_short | bfloat16 | `x` | 2.44e-04 | input |
| k2-diagonal-recurrence | latency_short | bfloat16 | `log_decay` | 4.88e-04 | input |
| k2-diagonal-recurrence | latency_short | bfloat16 | `initial_state` | 2.40e-05 | state |
| k2-diagonal-recurrence | throughput_medium | float32 | `x` | 6.98e-10 | input |
| k2-diagonal-recurrence | throughput_medium | float32 | `log_decay` | 1.86e-09 | input |
| k2-diagonal-recurrence | throughput_medium | float32 | `initial_state` | 1.27e-11 | state |
| k2-diagonal-recurrence | throughput_medium | bfloat16 | `x` | 7.63e-06 | input |
| k2-diagonal-recurrence | throughput_medium | bfloat16 | `log_decay` | 3.05e-05 | input |
| k2-diagonal-recurrence | throughput_medium | bfloat16 | `initial_state` | 5.03e-10 | state |
| k2-diagonal-recurrence | continuation_nonzero_state | float32 | `x` | 2.79e-09 | input |
| k2-diagonal-recurrence | continuation_nonzero_state | float32 | `log_decay` | 7.45e-09 | input |
| k2-diagonal-recurrence | continuation_nonzero_state | float32 | `initial_state` | 2.33e-10 | state |
| k2-diagonal-recurrence | continuation_nonzero_state | bfloat16 | `x` | 3.05e-05 | input |
| k2-diagonal-recurrence | continuation_nonzero_state | bfloat16 | `log_decay` | 1.22e-04 | input |
| k2-diagonal-recurrence | continuation_nonzero_state | bfloat16 | `initial_state` | 2.73e-08 | state |
| k2-diagonal-recurrence | decode_step | float32 | `x` | 0.00e+00 | input |
| k2-diagonal-recurrence | decode_step | float32 | `log_decay` | 0.00e+00 | input |
| k2-diagonal-recurrence | decode_step | float32 | `initial_state` | 0.00e+00 | state |
| k2-diagonal-recurrence | decode_step | bfloat16 | `x` | 0.00e+00 | input |
| k2-diagonal-recurrence | decode_step | bfloat16 | `log_decay` | 3.81e-06 | input |
| k2-diagonal-recurrence | decode_step | bfloat16 | `initial_state` | 7.03e-06 | state |
| k2-gated-delta-recurrence | latency_short | float32 | `q` | 9.31e-10 | input |
| k2-gated-delta-recurrence | latency_short | float32 | `k` | 7.45e-09 | input |
| k2-gated-delta-recurrence | latency_short | float32 | `v` | 1.16e-09 | input |
| k2-gated-delta-recurrence | latency_short | float32 | `g` | 3.35e-08 | input |
| k2-gated-delta-recurrence | latency_short | float32 | `beta` | 1.12e-08 | input |
| k2-gated-delta-recurrence | latency_short | float32 | `initial_state` | 5.82e-10 | state |
| k2-gated-delta-recurrence | latency_short | bfloat16 | `q` | 3.05e-05 | input |
| k2-gated-delta-recurrence | latency_short | bfloat16 | `k` | 2.44e-04 | input |
| k2-gated-delta-recurrence | latency_short | bfloat16 | `v` | 3.05e-05 | input |
| k2-gated-delta-recurrence | latency_short | bfloat16 | `g` | 2.44e-04 | input |
| k2-gated-delta-recurrence | latency_short | bfloat16 | `beta` | 4.88e-04 | input |
| k2-gated-delta-recurrence | latency_short | bfloat16 | `initial_state` | 6.25e-06 | state |
| k2-gated-delta-recurrence | throughput_medium | float32 | `q` | 1.09e-11 | input |
| k2-gated-delta-recurrence | throughput_medium | float32 | `k` | 1.16e-10 | input |
| k2-gated-delta-recurrence | throughput_medium | float32 | `v` | 8.19e-12 | input |
| k2-gated-delta-recurrence | throughput_medium | float32 | `g` | 5.82e-10 | input |
| k2-gated-delta-recurrence | throughput_medium | float32 | `beta` | 2.33e-10 | input |
| k2-gated-delta-recurrence | throughput_medium | float32 | `initial_state` | 2.27e-12 | state |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | `q` | 2.38e-07 | input |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | `k` | 1.91e-06 | input |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | `v` | 1.19e-07 | input |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | `g` | 3.81e-06 | input |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | `beta` | 3.81e-06 | input |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | `initial_state` | 2.04e-08 | state |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | `q` | 2.04e-10 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | `k` | 1.40e-09 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | `v` | 8.73e-11 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | `g` | 3.73e-09 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | `beta` | 1.86e-09 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | `initial_state` | 5.09e-11 | state |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | `q` | 1.91e-06 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | `k` | 3.05e-05 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | `v` | 1.91e-06 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | `g` | 3.05e-05 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | `beta` | 3.05e-05 | input |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | `initial_state` | 3.18e-07 | state |
| k2-gated-delta-recurrence | decode_step | float32 | `q` | 7.45e-09 | input |
| k2-gated-delta-recurrence | decode_step | float32 | `k` | 5.96e-08 | input |
| k2-gated-delta-recurrence | decode_step | float32 | `v` | 2.79e-09 | input |
| k2-gated-delta-recurrence | decode_step | float32 | `g` | 4.47e-08 | input |
| k2-gated-delta-recurrence | decode_step | float32 | `beta` | 2.98e-08 | input |
| k2-gated-delta-recurrence | decode_step | float32 | `initial_state` | 7.45e-09 | state |
| k2-gated-delta-recurrence | decode_step | bfloat16 | `q` | 1.22e-04 | input |
| k2-gated-delta-recurrence | decode_step | bfloat16 | `k` | 1.95e-03 | input |
| k2-gated-delta-recurrence | decode_step | bfloat16 | `v` | 3.05e-05 | input |
| k2-gated-delta-recurrence | decode_step | bfloat16 | `g` | 0.00e+00 | input |
| k2-gated-delta-recurrence | decode_step | bfloat16 | `beta` | 5.72e-05 | input |
| k2-gated-delta-recurrence | decode_step | bfloat16 | `initial_state` | 5.47e-05 | state |

## Inference decoding KL divergence (native vs upstream)

A single-token decode step with the native kernel and the upstream
comparator on the same operands; the outputs are softmaxed over the
feature dimension to a per-token distribution, and the KL divergence is
reported in both directions. Near-zero KL means the native decode path
produces the same output distribution as upstream (the quantity that
matters for sampling / generation quality).

| Workload | dtype | KL(native \|\| upstream) | KL(upstream \|\| native) |
|---|---|---|---|
| k1-mha | bfloat16 | -1.388e-07 | 1.444e-07 |
| k1-mha | float16 | 1.871e-08 | -1.858e-08 |
| k2-gated-delta-recurrence | float32 | 2.056e-08 | -2.056e-08 |
| k2-gated-delta-recurrence | bfloat16 | 2.444e-06 | 2.454e-06 |
| k2-diagonal-recurrence | float32 | 0.000e+00 | 0.000e+00 |
| k2-diagonal-recurrence | bfloat16 | 1.784e-07 | -2.039e-08 |
