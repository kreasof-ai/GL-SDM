"""External model module: modern Hopfield association layer (arch-078).

Verified composition-now row: each association step is exactly U1.S over the
stored-pattern domain — softmax(β·q·kᵀ + mask)·v with per-head scaling — and
the iteration loop, stopping criterion, query-refresh edge (q ← ξ·k),
projections and norms are external ordinary/control-flow structure (sweep
row-078, verified against hflayers/functional.py @ f56f929c).

Layer shape (mirroring the pinned ``Hopfield`` module): per-pattern LayerNorms
on stored/state/projection patterns, external in-projections (q/k/v), the
iterated typed association, and a disabled-by-default output projection.

Composition notes:

- The pinned per-head scaling multiplies the *query* by β before the
  association (functional.py:339-343). URM's K1 contract scales scores by the
  key-dim rule, and ``softmax((βq)kᵀ) == softmax(β(qkᵀ))`` — so the learnable
  per-head β lives here as an external stage (q ← β·q with the descriptor's
  key-dim scale folded out), never inside the typed call.
- The typed K1 contract is closed around one output ``softmax(·)·value``, so
  each iteration runs two typed calls sharing weights: the query refresh over
  the keys (q ← ξ·K) and the final readout over the pattern values (ξ·V).
- The pinned per-head early deactivation is flattened to the whole-batch
  stopping rule over the max head update norm — the same fixed point for the
  full-batch case.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class HopfieldAssociationLayer(torch.nn.Module):
    """One Hopfield association layer: norms + projections + typed association.

    Parameters
    ----------
    embed_dim:
        Full pattern width (``num_heads * head_dim``); the pinned layer's
        in-projections map embed_dim → embed_dim (q/k/v) over the full width.
    num_heads:
        Association heads (H).
    scaling:
        Initial per-head association scale β; default ``1/sqrt(head_dim)``
        matches the pinned source. Learnable, as in the pinned source.
    update_steps_max / update_steps_eps:
        Iteration bound / fixed-point tolerance (0 = single association, the
        fragment case; <0 = iterate to convergence).
    normalize_patterns:
        Whether to apply the pinned per-pattern LayerNorms (default True,
        matching the pinned module's default configuration).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        scaling: float | None = None,
        update_steps_max: int = 0,
        update_steps_eps: float = 1e-4,
        normalize_patterns: bool = True,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.update_steps_max = update_steps_max
        self.update_steps_eps = update_steps_eps

        # External stages: per-pattern norms and full-width q/k/v projections
        # (the pinned layer's E[patterns/projections/scaling]).
        self.normalize_patterns = normalize_patterns
        if normalize_patterns:
            self.norm_stored_pattern = torch.nn.LayerNorm(embed_dim)
            self.norm_state_pattern = torch.nn.LayerNorm(embed_dim)
            self.norm_pattern_projection = torch.nn.LayerNorm(embed_dim)
        self.in_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.scaling = torch.nn.Parameter(
            torch.full(
                (num_heads,), scaling if scaling is not None else self.head_dim ** -0.5
            )
        )

        document = {
            "schema_version": 2,
            "name": "hopfield_association_step",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32",
                     "shape": ["B", "T", num_heads, self.head_dim]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", num_heads, self.head_dim]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", num_heads, self.head_dim]},
                ],
                "nodes": [
                    {
                        "id": "associate",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value"],
                        "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence",
                            "source_domain": "sequence",
                            "selection": "dense",
                            "normalization": "softmax",
                            "capacity_policy": "dropless",
                            "deterministic": True,
                            "causal": False,
                            "head_map": "equal",
                        },
                    }
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(
            program, target=target, intent=CompilationIntent(intent)
        )

    def _association_step(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """One typed U1.S call with the per-head β folded into q.

        The K1 node applies the key-dim scale (1/√D); the learnable per-head β
        is applied to the query first, pre-divided so the composed scale equals
        the pinned source's per-head β.
        """
        # The native K1 kernel requires one floating-point dtype across q/k/v.
        # Under autocast the fp32 `scaling` parameter promotes the query to fp32 —
        # cast the scaled query back to the projection dtype.
        dtype = query.dtype
        scaled_q = (query * (self.scaling * (self.head_dim ** 0.5)).view(1, 1, -1, 1)).to(dtype)
        return self._plan.execute(
            query=scaled_q, key=keys.to(dtype), value=values.to(dtype),
        )["output"]

    def forward(
        self,
        state_patterns: torch.Tensor,
        stored_patterns: torch.Tensor,
        pattern_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, T, E]`` state patterns, ``[B, S, E]`` stored patterns → readout.

        ``pattern_values`` defaults to the stored patterns (V = Y, the standard
        Hopfield lookup).
        """
        if pattern_values is None:
            pattern_values = stored_patterns
        if self.normalize_patterns:
            stored_patterns = self.norm_stored_pattern(stored_patterns)
            state_patterns = self.norm_state_pattern(state_patterns)
            pattern_values = self.norm_pattern_projection(pattern_values)

        B, T, E = state_patterns.shape
        S = stored_patterns.shape[1]
        qkv = self.in_proj  # full-width external projections
        q = qkv(state_patterns)[..., :E]
        k = qkv(stored_patterns)[..., E : 2 * E]
        v = qkv(pattern_values)[..., 2 * E :]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)

        # The pinned loop computes ξ_n = softmax(β·q_n·kᵀ) and refreshes
        # q_{n+1} ← ξ_n·k; the association returned for the readout is the one
        # computed from the *current* query. With update_steps_max=0 the query
        # is never refreshed, so the readout uses the original q. Each
        # iteration is one typed U1.S call over the keys (ξ·K, the refresh);
        # the final readout is one typed call over the values (ξ·V) with the
        # query that produced the last ξ.
        xi_old: torch.Tensor | None = None
        update_step = 0
        while True:
            refresh = self._association_step(q, k, k)
            if xi_old is not None:
                delta = (xi_old - refresh).norm(p=2, dim=(-2, -1)).max()
                if delta <= self.update_steps_eps:
                    break
            xi_old = refresh
            # The readout belongs to the query that produced this ξ; refresh q
            # only when another iteration will run.
            if 0 <= self.update_steps_max <= update_step:
                break
            q = refresh
            update_step += 1
        # Final readout against the pattern values with the converged weights.
        out = self._association_step(q, k, v)
        return out.reshape(B, T, E)


__all__ = ["HopfieldAssociationLayer"]
