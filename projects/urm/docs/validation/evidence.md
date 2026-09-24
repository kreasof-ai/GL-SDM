# Evidence and claim policy

This page is the single human-readable status source. The machine catalog is [architecture-coverage.json](../../benchmarks/architecture-coverage.json); raw benchmark records remain under `results/`. The [composition ledger](../planning/architecture-composition.md) describes target graphs, not measured support.

## Current status at HEAD `4cb35c5`

| Scope | Evidence now | Claim allowed |
|---|---|---|
| Named source catalog | 80 IDs: 76 mixer-relevant, four outside mixer scope | Source identity and construction backlog |
| Current public graph recipes | 14 K1 dense-softmax fragments, one K3 route-to-state fragment, no K2 graph recipe | Fragment representation/execution only within each tested envelope |
| Native public-path profiles | Four catalog rows have retained URM-native K1 fragment profiles (MHA/MQA/GQA/BitAttention) | Those exact fragment/device/mode comparisons, not source-model performance |
| Complete faithful source models through public graph path | None qualified | No 76-model or three-family performance coverage claim |
| Preserved earlier generic decoder | 62 old recipe rows in `results/validation/master-table.json`; generator is absent from the current tree | Historical engineering evidence, not current compiler or source-model qualification |

The 76 registered kernel-upstream comparisons validate pinned **slices** through the retired comparison path. Their “pass” fields do not certify a current native provider, full architecture, cache or end-to-end model. `pending_graph_migration` is a historical label, not a promise that all those equations fit an existing K2 descriptor.

## Five independent verdicts

Every ID/mode/shape/dtype/device reports these separately:

1. **Represented:** a closed semantic descriptor preserves the exact equation, effects and state/cache contract; unsupported fields decline.
2. **Reference executable:** the public graph executes through independent NumPy/Torch references with output, parameter, route and state-gradient checks.
3. **Native qualified:** a URM-owned provider selected by the public compiler runs forward/backward/decode as claimed and matches the reference. An upstream adapter is a separate `library` tier.
4. **Performance qualified:** the complete timed boundary meets a predeclared paired throughput, latency and memory gate, with raw samples and uncertainty. Training, prefill and decode are separate.
5. **Source-model qualified:** an external complete model module matches the pinned source unit's layer arrangement, parameters, logits, gradients and cache continuation. Fragment parity cannot set this verdict.

Do not collapse these verdicts into one “coverage” number. A result must state the timed boundary (`kernel fragment`, `generic decoder`, `complete source model`), provider tier (`native`, `library`, `reference`), revision, hardware/software environment, precision policy, intent mode, shape and state initialization. Missing upstream modes are `upstream_unavailable`, not passes.

## Historical performance arithmetic, with limits

The retired generic decoder has 61 successful paired native training measurements: K1 22 (median native/upstream throughput ratio 0.955; 18 within the arithmetic 5% slowdown proxy), K2 38 (median 0.377; 4 within the proxy; 16 more than 10× slower), K3 one (ratio 2.033). For 49 recipes with comparable train, prefill-4096 and decode-256 values, 13 meet that arithmetic proxy in all three modes (10 K1, three K2). Two of the three K2 cases are H3/Hyena paths using ordinary Torch FFT; their speed does not qualify a generic Triton recurrence. These are **fragment mappings on a frozen decoder**, not model coverage. Reproduce the arithmetic with `python benchmarks/analysis/coverage_recovery.py` from `projects/urm`.

The five focused preserved native qualification artifacts under `results/qualification/native-*.json` report `correct_below_target` under their stricter frozen gates. They are not overridden by the arithmetic proxy. A fast decode result does not repair the poor long-prefill/training K2 median.

## Acceptance procedure

1. Freeze the source revision, source function/model, workload, allowed precision, tolerance, mandatory modes, performance and memory gate **before** tuning. Keep run provenance and raw samples.
2. Compare independent equation reference, direct upstream, public URM reference, native and explicit library tiers at the same input and state boundary. Test all operand and parameter VJPs, final-state cotangents, nonzero state, cache offsets and repeated decode where applicable.
3. Time the **entire claimed unit**: route/index generation, projections/ordinary operators, kernel calls, intermediate materialization, state/cache traffic, launch and backward. Use paired randomized runs, guard device drift, report median/p95 and confidence interval. An analytical solver estimate is not measurement.
4. Publish the per-ID closeout record described in the [composition ledger](../planning/architecture-composition.md#required-closeout-record-for-each-id), with exact provider/plan step hashes, declines and fallback tier. A new backend branch also needs the [two-client admission record](../compiler/compiler-charter.md#backend-branch-admission).

The [alignment](alignment.md) and [inference-throughput](inference-throughput.md) pages are mechanically rendered from preserved results. They retain their original boundaries; do not interpret them as current source-model serving results.
