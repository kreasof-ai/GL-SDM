"""External model module: AttnRes depth-domain residual aggregation (arch-054).

Verified composition-now row as a typed depth-domain U1.S instance (sweep
row-054, against fla/ops/attnres/naive.py @ 864a87f6): each residual source is
RMSNorm-ed into a key, logits are the pseudo-query dotted against the keys
(scaled), softmax is over the DEPTH/source axis (dim=0, L = layer index), and
the output is the weighted sum of the UNNORMALIZED residual sources. Depth is
the source domain — there is no sequence state, recurrence or causal mask
(sweep blocker: "do not treat depth as sequence state").

Composition: the per-source RMSNorm key construction and the optional output
RMSNorm (fusing the following prenorm) are external; the mixer is one typed K1
``weighted_reduce`` call per flattened position with ``source_domain="depth"``
— q = query·scale [1,1,1,D], k = RMSNorm-ed sources [1,1,L,D], v = raw
residuals [1,1,L,D], non-causal, equal head map, explicit-operand scale of 1.0
(the module applies the pinned ``scale`` to the query externally, so the typed
call does not re-scale by the key-dim rule).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class AttnResLayer(torch.nn.Module):
    """One AttnRes aggregation: external RMSNorm keys + typed depth-domain K1.

    ``query`` is the per-layer pseudo-query ``[D]``; ``residuals`` are the
    previous-layer residual sources ``[L][..., D]``. The mixer runs per
    flattened position over the depth axis.
    """

    def __init__(
        self,
        num_sources: int,
        *,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError("AttnRes requires at least one residual source")
        self.num_sources = num_sources

        document = {
            "schema_version": 2,
            "name": "attnres_depth_mixer",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["N", 1, 1, "D"]},
                    {"name": "key", "dtype": "float32", "shape": ["N", 1, num_sources, "D"]},
                    {"name": "value", "dtype": "float32", "shape": ["N", 1, num_sources, "D"]},
                    {"name": "scale", "dtype": "float32", "shape": []},
                ],
                "nodes": [
                    {
                        "id": "aggregate",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "scale"],
                        "outputs": ["output"],
                        "params": {
                            "query_domain": "depth",
                            "source_domain": "depth",
                            "selection": "dense",
                            "normalization": "softmax",
                            "capacity_policy": "dropless",
                            "deterministic": True,
                            "causal": False,
                            "head_map": "equal",
                            "scale_rule": "explicit_operand",
                            "roles": {
                                "query": "query",
                                "key": "key",
                                "value": "value",
                                "scale": "scale",
                            },
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

    def forward(
        self,
        query: torch.Tensor,
        residuals: list[torch.Tensor],
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor | None = None,
        rms_eps: float = 1e-6,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Pinned signature: query ``[D]`` or ``[D, 1]``, residuals ``[L][..., D]``."""
        if len(residuals) != self.num_sources:
            raise ValueError(f"expected {self.num_sources} residual sources")
        output_shape = residuals[0].shape
        D = output_shape[-1]
        stacked = torch.stack(tuple(r.view(-1, D) for r in residuals), dim=0)  # [L, N, D]

        # External: RMSNorm keys; unnormalized residuals are the values.
        v = stacked.float()
        k = F.rms_norm(v, (D,), rms_weight.flatten().float(), rms_eps)
        q = (query.flatten().float() * scale)

        L, N, _ = stacked.shape
        # Faithful K1 layout: each flattened position is one BATCH element and
        # the depth sources form the sequence axis — q [N, 1, 1, D] (one query,
        # one head), k/v [N, L, 1, D] (L depth sources). The pinned scale is
        # already applied to q, so the typed call uses an explicit scale of 1.
        out = self._plan.execute(
            query=q.view(1, 1, 1, D).expand(N, 1, 1, D).contiguous(),
            key=k.permute(1, 0, 2).unsqueeze(2).contiguous(),   # [N, L, 1, D]
            value=v.permute(1, 0, 2).unsqueeze(2).contiguous(),  # [N, L, 1, D]
            scale=torch.ones((), dtype=torch.float32, device=q.device),
        )["output"]  # [N, 1, 1, D]
        o = out.view(N, D).view(output_shape)

        if output_rms_weight is not None:
            o = F.rms_norm(o, (D,), output_rms_weight.float(), rms_eps)
        return o.to(stacked.dtype)


__all__ = ["AttnResLayer"]
