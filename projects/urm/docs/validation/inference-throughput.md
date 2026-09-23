# Inference throughput and MFU: URM-native vs upstream

Status: evidence record, regenerated from the committed release-gate
artifacts by `benchmarks/inference_report.py`. This is the product-facing
serving comparison the acceptance contract calls for: absolute inference
throughput (tokens/second) and model FLOPs utilization (MFU) for the native
kernel and the upstream comparator on the same operands, per case, dtype,
and mode.

MFU uses the **measured** hardware peak (`results/device-limits.json`), never
the vendor datasheet, per the acceptance requirement to use measured hardware
denominators. The FLOP model counts the operation's useful model FLOPs (the
matmul / state work), documented in `benchmarks/inference_report.py`.

## Prefill throughput (tokens/second)

| Workload | Case | dtype | upstream tok/s | native tok/s | native overhead | gate |
|---|---|---|---|---|---|---|
| k1-mha | latency_small | bfloat16 | 2014289 | 459068 | +311.4% | FAIL |
| k1-mha | latency_small | float16 | 1923034 | 454763 | +322.9% | FAIL |
| k1-mha | throughput_medium | bfloat16 | 19691929 | 14596121 | +40.6% | FAIL |
| k1-mha | throughput_medium | float16 | 19512521 | 14491408 | +37.4% | FAIL |
| k1-mha | throughput_long | bfloat16 | 6772782 | 5849232 | +15.7% | FAIL |
| k1-mha | throughput_long | float16 | 6708539 | 5877274 | +14.1% | FAIL |
| k1-mha | decode_step | bfloat16 | 14062 | 3548 | +324.7% | FAIL |
| k1-mha | decode_step | float16 | 14103 | 3885 | +256.9% | FAIL |
| k1-gqa | latency_small | bfloat16 | 2087700 | 480345 | +304.0% | FAIL |
| k1-gqa | latency_small | float16 | 1836811 | 478916 | +291.2% | FAIL |
| k1-gqa | throughput_long | bfloat16 | 3536766 | 3201135 | +10.3% | FAIL |
| k1-gqa | throughput_long | float16 | 3546005 | 3192822 | +10.9% | FAIL |
| k1-gqa | decode_step | bfloat16 | 11992 | 3741 | +196.8% | FAIL |
| k1-gqa | decode_step | float16 | 12696 | 3756 | +224.5% | FAIL |
| k1-masked-variant | block_sparse_medium | bfloat16 | 12785596 | 12774212 | +0.2% | PASS |
| k1-masked-variant | fully_masked_rows | bfloat16 | 7959426 | 3869846 | +109.8% | FAIL |
| k2-diagonal-recurrence | latency_short | float32 | 417292 | 425833 | -2.6% | PASS |
| k2-diagonal-recurrence | latency_short | bfloat16 | 393198 | 434149 | -11.3% | PASS |
| k2-diagonal-recurrence | throughput_medium | float32 | 32189516 | 21931187 | +46.1% | FAIL |
| k2-diagonal-recurrence | throughput_medium | bfloat16 | 29294368 | 22616670 | +29.3% | FAIL |
| k2-diagonal-recurrence | continuation_nonzero_state | float32 | 3258824 | 2982551 | +9.2% | PASS |
| k2-diagonal-recurrence | continuation_nonzero_state | bfloat16 | 3095131 | 3411973 | -9.3% | PASS |
| k2-diagonal-recurrence | decode_step | float32 | 6607 | 6624 | -3.1% | PASS |
| k2-diagonal-recurrence | decode_step | bfloat16 | 6071 | 6718 | -9.6% | PASS |
| k2-gated-delta-recurrence | latency_short | float32 | 113417 | 442721 | -74.3% | PASS |
| k2-gated-delta-recurrence | latency_short | bfloat16 | 114073 | 445122 | -74.3% | PASS |
| k2-gated-delta-recurrence | throughput_medium | float32 | 9433134 | 3048775 | +209.2% | FAIL |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | 13786070 | 3063386 | +349.7% | FAIL |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | 878016 | 1222077 | -28.2% | PASS |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | 894383 | 1186913 | -25.5% | PASS |
| k2-gated-delta-recurrence | decode_step | float32 | 1774 | 6605 | -73.3% | PASS |
| k2-gated-delta-recurrence | decode_step | bfloat16 | 1780 | 7107 | -74.2% | PASS |

## Decode throughput (tokens/second)

| Workload | Case | dtype | upstream tok/s | native tok/s | native overhead | gate |
|---|---|---|---|---|---|---|
| k1-mha | latency_small | bfloat16 | 21609 | 16169 | +31.8% | FAIL |
| k1-mha | latency_small | float16 | 20766 | 12769 | +34.0% | FAIL |
| k1-mha | throughput_medium | bfloat16 | 64006 | 72094 | -13.3% | PASS |
| k1-mha | throughput_medium | float16 | 64755 | 81510 | -21.2% | PASS |
| k1-mha | throughput_long | bfloat16 | 23018 | 22425 | +0.9% | PASS |
| k1-mha | throughput_long | float16 | 23477 | 21575 | +5.8% | FAIL |
| k1-mha | decode_step | bfloat16 | 17968 | 12591 | +33.4% | FAIL |
| k1-mha | decode_step | float16 | 18348 | 12731 | +35.3% | FAIL |
| k1-gqa | latency_small | bfloat16 | 20237 | 15796 | +23.0% | FAIL |
| k1-gqa | latency_small | float16 | 19335 | 15232 | +23.1% | FAIL |
| k1-gqa | throughput_long | bfloat16 | 33462 | 23672 | +43.4% | FAIL |
| k1-gqa | throughput_long | float16 | 37445 | 25090 | +45.9% | FAIL |
| k1-gqa | decode_step | bfloat16 | 16942 | 12791 | +27.7% | FAIL |
| k1-gqa | decode_step | float16 | 12561 | 12089 | +11.9% | FAIL |
| k2-diagonal-recurrence | latency_short | float32 | 9038 | 11361 | -20.4% | PASS |
| k2-diagonal-recurrence | latency_short | bfloat16 | 9375 | 11237 | -18.0% | PASS |
| k2-diagonal-recurrence | throughput_medium | float32 | 73199 | 90513 | -16.7% | PASS |
| k2-diagonal-recurrence | throughput_medium | bfloat16 | 75242 | 87882 | -14.7% | PASS |
| k2-diagonal-recurrence | continuation_nonzero_state | float32 | 17715 | 22301 | -20.6% | PASS |
| k2-diagonal-recurrence | continuation_nonzero_state | bfloat16 | 18272 | 22425 | -16.3% | PASS |
| k2-diagonal-recurrence | decode_step | float32 | 9083 | 10709 | -14.6% | PASS |
| k2-diagonal-recurrence | decode_step | bfloat16 | 9367 | 11444 | -17.6% | PASS |
| k2-gated-delta-recurrence | latency_short | float32 | 7603 | 17284 | -55.9% | PASS |
| k2-gated-delta-recurrence | latency_short | bfloat16 | 7715 | 17867 | -56.1% | PASS |
| k2-gated-delta-recurrence | throughput_medium | float32 | 59604 | 130531 | -53.8% | PASS |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | 61030 | 133366 | -55.4% | PASS |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | 13728 | 33370 | -58.9% | PASS |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | 14718 | 33631 | -56.4% | PASS |
| k2-gated-delta-recurrence | decode_step | float32 | 7614 | 16826 | -56.2% | PASS |
| k2-gated-delta-recurrence | decode_step | bfloat16 | 7559 | 16647 | -54.6% | PASS |
| k3-sparse-state | ordered_collisions | float32 | 4355 | 3435 | +26.8% | FAIL |
| k3-sparse-state | ordered_collisions | bfloat16 | 4505 | 4311 | -2.5% | PASS |
| k3-sparse-state | imbalanced_routes | float32 | 10518 | 8102 | +11.5% | FAIL |
| k3-sparse-state | imbalanced_routes | bfloat16 | 8333 | 9264 | -10.1% | PASS |
| k3-sparse-state | overlapping_reads | float32 | 5070 | 4534 | +8.0% | FAIL |
| k3-sparse-state | overlapping_reads | bfloat16 | 4673 | 4594 | -4.5% | PASS |

## MFU: model FLOPs utilization (native vs upstream)

MFU = useful model FLOPs / measured wall time / measured hardware peak.
Reported for the forward and decode modes (the serving paths).

| Workload | Case | dtype | mode | upstream MFU | native MFU | measured peak (TFLOP/s) |
|---|---|---|---|---|---|---|
| k1-mha | latency_small | bfloat16 | forward | 0.3% | 0.1% | 66.2 |
| k1-mha | latency_small | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k1-mha | latency_small | float16 | forward | 0.4% | 0.1% | 66.2 |
| k1-mha | latency_small | float16 | decode | 0.0% | 0.0% | 66.2 |
| k1-mha | throughput_medium | bfloat16 | forward | 63.0% | 45.8% | 66.2 |
| k1-mha | throughput_medium | bfloat16 | decode | 0.4% | 0.5% | 66.2 |
| k1-mha | throughput_medium | float16 | forward | 63.4% | 45.8% | 66.2 |
| k1-mha | throughput_medium | float16 | decode | 0.4% | 0.5% | 66.2 |
| k1-mha | throughput_long | bfloat16 | forward | 85.2% | 74.5% | 66.2 |
| k1-mha | throughput_long | bfloat16 | decode | 0.6% | 0.6% | 66.2 |
| k1-mha | throughput_long | float16 | forward | 84.3% | 74.4% | 66.2 |
| k1-mha | throughput_long | float16 | decode | 0.6% | 0.5% | 66.2 |
| k1-mha | decode_step | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k1-mha | decode_step | bfloat16 | decode | 0.1% | 0.1% | 66.2 |
| k1-mha | decode_step | float16 | forward | 0.0% | 0.0% | 66.2 |
| k1-mha | decode_step | float16 | decode | 0.1% | 0.1% | 66.2 |
| k1-gqa | latency_small | bfloat16 | forward | 0.4% | 0.1% | 66.2 |
| k1-gqa | latency_small | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k1-gqa | latency_small | float16 | forward | 0.4% | 0.1% | 66.2 |
| k1-gqa | latency_small | float16 | decode | 0.0% | 0.0% | 66.2 |
| k1-gqa | throughput_long | bfloat16 | forward | 89.0% | 80.7% | 66.2 |
| k1-gqa | throughput_long | bfloat16 | decode | 1.7% | 1.2% | 66.2 |
| k1-gqa | throughput_long | float16 | forward | 88.8% | 79.9% | 66.2 |
| k1-gqa | throughput_long | float16 | decode | 1.9% | 1.3% | 66.2 |
| k1-gqa | decode_step | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k1-gqa | decode_step | bfloat16 | decode | 0.1% | 0.1% | 66.2 |
| k1-gqa | decode_step | float16 | forward | 0.0% | 0.0% | 66.2 |
| k1-gqa | decode_step | float16 | decode | 0.1% | 0.1% | 66.2 |
| k1-masked-variant | block_sparse_medium | bfloat16 | forward | 20.3% | 19.4% | 66.2 |
| k1-masked-variant | fully_masked_rows | bfloat16 | forward | 6.2% | 3.0% | 66.2 |
| k2-diagonal-recurrence | latency_short | float32 | forward | 0.0% | 0.0% | 23.1 |
| k2-diagonal-recurrence | latency_short | float32 | decode | 0.0% | 0.0% | 23.1 |
| k2-diagonal-recurrence | latency_short | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k2-diagonal-recurrence | latency_short | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k2-diagonal-recurrence | throughput_medium | float32 | forward | 0.1% | 0.1% | 23.1 |
| k2-diagonal-recurrence | throughput_medium | float32 | decode | 0.0% | 0.0% | 23.1 |
| k2-diagonal-recurrence | throughput_medium | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k2-diagonal-recurrence | throughput_medium | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k2-diagonal-recurrence | continuation_nonzero_state | float32 | forward | 0.0% | 0.0% | 23.1 |
| k2-diagonal-recurrence | continuation_nonzero_state | float32 | decode | 0.0% | 0.0% | 23.1 |
| k2-diagonal-recurrence | continuation_nonzero_state | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k2-diagonal-recurrence | continuation_nonzero_state | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k2-diagonal-recurrence | decode_step | float32 | forward | 0.0% | 0.0% | 23.1 |
| k2-diagonal-recurrence | decode_step | float32 | decode | 0.0% | 0.0% | 23.1 |
| k2-diagonal-recurrence | decode_step | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k2-diagonal-recurrence | decode_step | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k2-gated-delta-recurrence | latency_short | float32 | forward | 0.0% | 0.0% | 23.1 |
| k2-gated-delta-recurrence | latency_short | float32 | decode | 0.0% | 0.0% | 23.1 |
| k2-gated-delta-recurrence | latency_short | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k2-gated-delta-recurrence | latency_short | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k2-gated-delta-recurrence | throughput_medium | float32 | forward | 6.7% | 2.2% | 23.1 |
| k2-gated-delta-recurrence | throughput_medium | float32 | decode | 0.0% | 0.1% | 23.1 |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | forward | 3.2% | 0.8% | 66.2 |
| k2-gated-delta-recurrence | throughput_medium | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | forward | 0.6% | 0.9% | 23.1 |
| k2-gated-delta-recurrence | continuation_nonzero_state | float32 | decode | 0.0% | 0.0% | 23.1 |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | forward | 0.2% | 0.3% | 66.2 |
| k2-gated-delta-recurrence | continuation_nonzero_state | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k2-gated-delta-recurrence | decode_step | float32 | forward | 0.0% | 0.0% | 23.1 |
| k2-gated-delta-recurrence | decode_step | float32 | decode | 0.0% | 0.0% | 23.1 |
| k2-gated-delta-recurrence | decode_step | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k2-gated-delta-recurrence | decode_step | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k3-sparse-state | ordered_collisions | float32 | forward | 0.0% | 0.0% | 23.1 |
| k3-sparse-state | ordered_collisions | float32 | decode | 0.0% | 0.0% | 23.1 |
| k3-sparse-state | ordered_collisions | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k3-sparse-state | ordered_collisions | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k3-sparse-state | imbalanced_routes | float32 | forward | 0.0% | 0.0% | 23.1 |
| k3-sparse-state | imbalanced_routes | float32 | decode | 0.0% | 0.0% | 23.1 |
| k3-sparse-state | imbalanced_routes | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k3-sparse-state | imbalanced_routes | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
| k3-sparse-state | overlapping_reads | float32 | forward | 0.0% | 0.0% | 23.1 |
| k3-sparse-state | overlapping_reads | float32 | decode | 0.0% | 0.0% | 23.1 |
| k3-sparse-state | overlapping_reads | bfloat16 | forward | 0.0% | 0.0% | 66.2 |
| k3-sparse-state | overlapping_reads | bfloat16 | decode | 0.0% | 0.0% | 66.2 |
