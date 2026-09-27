# The architecture catalog

Mirrors `train/registry.py` (52 rows) and `architectures/` (52 modules). Every row
is a `MixerSpec`; each is registered at its faithful **HF-modeling granularity** —
the level at which the pinned upstream defines the architecture.

## Granularity taxonomy

| Granularity | Meaning | Rows |
|---|---|---|
| `mixer` (default) | The row is a [B,T,C]→[B,T,C] mixer module in the shared decoder block | 49 rows |
| `schedule` | The row owns the per-layer mixer *choice* (interleaved hybrid) | `samba_attention` (mamba2-K2 even layers / sliding-window attention odd) |
| `block` | The row owns the full decoder block (norms/residual/MLP) | `pattention` (tokenformer: Q/K/V/O *and* the MLP are all parameter-token attention) |
| `residual` | The row owns the cross-block residual aggregation; base blocks are dense attention | `attnres` (depth-domain softmax over block summaries) |

## Row flags

- **reference-tier** — `mamba1` only: the accepted charter debt (the K2 elementwise
  gate has a single client; the backend admission's two-client rule blocks a native
  branch). mamba1 is not in the 51-row native sweep.
- **atomic-bwd** — the native indexed-K1 backward reduces through relaxed
  `tl.atomic_add` (by design): `dsa`, `longformer`, `sparse_transformer`, `nsa`.
  These certify checkpoint parity by loss trajectory + round-trip structure.
- **stateful** — `sdm` carries a persistent memory bank across microbatches
  (harness reset/detach lifecycle).
- **external-composition** — `mom`, `raven` are plain-torch compositions that do not
  route through a URM plan (`public_path=False`); trained as-is so the report never
  claims compiler coverage it didn't exercise. `sdm` uses `public_path=True`:
  the complete route/update/read graph selects the native K3 provider and its
  generic guarded chunk schedule.
- The registry `upstream` field names the KL-gate parity oracle, where wired — it
  is *not* a statement that no upstream kernel exists (the benchmark's upstream arm
  covers 50/51 native rows; see [benchmark.md](benchmark.md)).

## Per-row map

| Row | KL oracle (registry `upstream`) | Flags |
|---|---|---|
| abc_gsa | fla.ops.abc | |
| attnres | fla.ops.attnres | residual |
| based_attention | fla.ops.based | |
| bit_attention | — | |
| cat_attention | — | |
| comba | fla.ops.comba | |
| conformer_attention | — | |
| deltaformer | — | |
| deltanet | fla.ops.delta_rule.naive.delta_rule_recurrence | |
| dense_attention | torch SDPA | |
| differential_attention | — | |
| dplr | fla.ops.generalized_delta_rule.dplr | |
| dsa | — | atomic-bwd |
| forgetting_attention | fla.ops.forgetting_attn.naive | |
| gated_delta_product | fla.ops.gated_delta_product | |
| gated_deltanet | fla.ops.gated_delta_rule.naive | |
| gdn2 | fla.ops.gdn2.naive | |
| gla | fla.ops.gla.naive | |
| gsa | fla.ops.gsa | |
| hgrn2 | fla.ops.gla.naive | |
| hla | — | no upstream exists (empty pin, paper-only) |
| hopfield_association | — | |
| iplr | — | |
| kata | — | |
| kda | fla.ops.kda.naive | |
| lightnet | — | |
| lightning_attention | fla.ops.simple_gla.naive | |
| linear_attention | fla.ops.linear_attn.naive | |
| log_linear_attention | — | |
| log_linear_mamba2 | — | upstream kernel environment-blocked (SMEM) |
| longformer | — | atomic-bwd |
| mamba1 | mamba_ssm | reference-tier (charter debt) |
| mamba2 | mamba_ssm | |
| mla_attention | — | |
| moba | — | |
| mom | — | external-composition |
| nsa | fla.ops.nsa | atomic-bwd |
| path_attention | — | |
| patention | — | block |
| raven | — | external-composition |
| retnet | fla.ops.retention.naive | |
| rodimus | — | |
| rwkv7 | fla.ops.rwkv7 | |
| samba_attention | fla.models.samba | schedule |
| sdm | — | stateful; complete public K3 graph; native chunk schedule; actual pinned CUDA baseline |
| simple_gla | fla.ops.simple_gla.naive | |
| sparse_transformer | — | atomic-bwd |
| tda | — | |
| tpa_attention | — | |
| tucker_attention | — | |
| wall_attention | — | |
| yoco | — | |

The pinned upstream checkouts live in `/tmp/urm-comparator-pins/` (reprovisioned by
`extra/provision_comparators.py`); the verified adapters live in
`extra/comparators/`.
