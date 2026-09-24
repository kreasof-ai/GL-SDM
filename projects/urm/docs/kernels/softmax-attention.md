# K1: streamed attention reduction

**Written equation contract, not a universal score/reducer implementation.** For query `q_i`, key `k_j`, value `v_j`, explicit query-to-KV head map `h(i)`, visibility `V(i,j)`, scale and optional bias:

```text
s_ij = scale * dot(q_i, k_j) + bias_ij
p_ij = exp(s_ij - m_i) / sum_{j:V(i,j)} exp(s_ij - m_i)
y_i  = sum_{j:V(i,j)} p_ij v_j
```

`m_i` is a stable row maximum. A fully masked row returns zero and has zero input gradients. Causal position offsets are operands independent of tensor lengths, including cached decode and cross-attention. MHA, MQA and GQA differ by `h(i)`, not by architecture-name branches. Dropout requires an additional explicit RNG/replay and backward contract; it is not silently enabled.

Dense tiled online softmax is a physical schedule of this equation. Indexed sparse execution must traverse only selected keys and preserve route order, ties, masking and gradients; a dense masked implementation proves values but **does not** prove sparse-work performance. Alternative scores or reductions (FoX, Wall, POLAR, TDA, KATA, etc.) require a new closed descriptor, independent reference and backend admission. They do not automatically inherit this softmax equation.

For a tiled native reduction maintain `(m,l,a)`, where `m` is the row maximum, `l` the exponent sum and `a` the weighted-value sum. For a new score tile with maximum `m_b`, set `m'=max(m,m_b)`, `l'=exp(m-m')l+Σ_b exp(s_b-m')`, and `a'=exp(m-m')a+Σ_b exp(s_b-m')v_b`; return `a'/l'`. Empty/all-masked tiles need explicit neutral handling to avoid `-∞-(-∞)`. Backward may recompute probabilities from saved row statistics: `dV=PᵀdY`, `dP=dYVᵀ`, `dS=P⊙(dP-row_sum(P⊙dP))`, then `dQ,dK,dscale,dbias` follow the score law. Shared KV heads reduce their gradients across mapped query heads. Save/recompute and accumulation dtype are part of the provider envelope.

The present Triton K1 path has a bounded measured MHA/MQA/GQA fragment envelope, including recomputed backward. It has no claim here for full model layers, general sparse traversal, all cache layouts, dropout or arbitrary reducer combinations. The [charter](../compiler/compiler-charter.md) governs promotion and the [evidence protocol](../validation/evidence.md) governs claims.
