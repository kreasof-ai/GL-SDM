# Unification audit and decisions

Audit date: 2026-09-19. Scope: archived proposals, current compiler boundaries,
FLA inventory and selected equation/reference inspections. This is not an
exhaustive equivalence proof or GPU run. The [register](coverage.md) preserves
the remaining mappings as explicit construction work.

## Corrections to the archived plan

| Archived assertion | Audit finding | Construction decision |
|---|---|---|
| RWKV-7 loses chunked execution because of its value residual | Upstream exports chunk and recurrent implementations; a residual alone is not a barrier | Include RWKV-7 and derive its generalized transition |
| All delta variants share scalar-beta rank-one semantics | Independent factors, multiple updates, preconditioners and extra state need distinct contracts | Add typed transitions and composed state rather than architecture flags |
| ABC is a bounded attention mask; GSA/Raven are equivalent to SDM | ABC/GSA references maintain two recurrent summaries with an intermediate softmax; Raven calls GSA | Explicit state bundles and normalization composition |
| PaTH only changes q/k projections | Its reference builds a triangular transformation before attention | Typed transformation operation with its own VJP |
| Preconditioning is a row scale | PGDN reference maintains additional preconditioner state | Preserve that state, gradients and update order |
| Mamba-2 is a block-diagonal attention mask | Chunk boundaries still communicate state | Retain semiseparable transition and boundary terms |
| Mamba-3 is a minor diagonal variant | Its wrapper has SISO/MIMO paths and multiple cache components | Separate mode qualification and state bundle ABI |
| A callback establishes TTT coverage | A callback does not establish effects, differentiation or serialization | Typed inner-update regions or explicit external anchors |
| Pattention means arbitrary MLP/MoE is already one kernel | Additional projections, gates and dispatch still need representation | Parameter/expert-axis graph composition |
| Axes are one-line changes; three sources cover nearly everything | Expressibility does not establish efficient equivalent execution | Keep the axes, with proof obligations and measured gates |

## Inspected primary sources

FLA audit revision: `864a87f6ce5be8828bef81eb22baafd41937cdf2`.
The findings above are this audit's interpretations of the following code:

- [RWKV-7 exports](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rwkv7/__init__.py): chunk and recurrent entry points exist.
- [Generalized IPLR reference](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/generalized_delta_rule/iplr/naive.py): separate transition factors, recurrence and chunk formulations.
- [ABC](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/abc/naive.py) and [GSA](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gsa/naive.py): two summaries with softmax between them.
- [Raven layer](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/raven.py): GSA calls and router/decay choices.
- [PaTH reference](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/path_attn/naive.py): transformation construction followed by attention.
- [PGDN reference](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/precond_gated_delta_rule/naive.py): accumulated preconditioner and memory updates.
- [Mamba-3 wrapper](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/mamba3.py): separate implementations and phase/SSM/prior-key/prior-value cache.
- [Gated-delta reference](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_delta_rule/naive.py): decay precedes retrieval at this revision. This does not automatically resolve the older frozen 0.5.2 comparator.

## Generalized transition rather than architecture exclusions

An extension target is `M_t = F_t M_(t-1) + U_t`, with input-precomputable
`F_t = D_t + L_t R_t^T`. Affine composition is associative in real arithmetic:
`(F2,U2) compose (F1,U1) = (F2 F1, F2 U1 + U2)`.
This gives an algebraic route, not an efficient compact scan: products can grow
in rank or density, and reassociation needs a numerical policy. Track rank,
shapes and cost before selecting a lowering. Nonlinear state-dependent
coefficients cannot be treated as precomputed input projections without proof.

## Remaining audit debt

Inventory is not an equation audit. Ambiguous CAT, TDA, TPA/Tucker, KATA, HLA and
composite targets need exact identity and source resolution. Identified sources
still need license checks, callable binding, supported-mode probes and pinned
runtime environments. Performance and novelty claims in archived proposals remain
unverified. No novelty claim is made for the proposed generality axes.
