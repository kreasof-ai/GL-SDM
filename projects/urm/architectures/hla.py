"""External model module: HLA (arch-074, higher-order linear attention).

Verified combinator row (flagship Nest), masked second-order UNNORMALIZED
causal case (paper Eq. 3.3 + Algorithm 1, the default operator) — against the
pinned paper (HLA.pdf, arXiv 2510.27258, @ 484fef2b).

The Nest combinator composes through the public graph as TWO DEPENDENT typed K2
calls (the exact factorization of the masked unnormalized law):

    S_t = S_(t-1) + k_t k_tᵀ ;  r_t = S_t q_t      (K2 call 1: query=q, key=k, value=k)
    F_t = F_(t-1) + r_t v_tᵀ ;  o_t = q_tᵀ F_t      (K2 call 2: query=q, key=r, value=v)

Both are the plain additive K2 law (``delta=False``, ``gate_scope=none``,
``scale_rule=one``, after-update read). The paper's correction form satisfies
``F_t = S_t C_t − G_t``; expanding it gives ``F_t = Σ_(j≤t) (S_j q_j) v_jᵀ``,
which is exactly the second call's state. The first call's output orientation is
exact because ``S_t`` is symmetric. The two states (``S`` and ``F``) are explicit
producer-carried edges; the intermediate ``r`` is materialized between the two
calls. This stays UNFUSED (two launches); fusing the pair is a later proven
rewrite, not a semantic requirement.

The normalized variant (Eq. 3.4) appends a constant ``1`` to each value in the
second call to produce the denominator, with an explicit ``epsilon`` division
outside K2 — residual here. Decay γ (Eq. 4.1) and ridge λ (Alg. 1 line 7) are
explicitly NOT claimed (sweep verdict). VJP through the pair is by torch autograd
through the two K2 reference calls.

``hla_serial_reference`` is retained as the independent comparator (the direct
transcription of Algorithm 1's serial recurrence) that the parity gate checks the
public two-K2 graph against.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def hla_serial_reference(query, key, value):
    """Independent comparator: the pinned Algorithm 1 serial recurrence.

    ``query``/``key`` ``[B,H,T,K]``, ``value`` ``[B,H,T,V]`` → ``[B,H,T,V]``.
    Per-token ΔS = k kᵀ, ΔC = q vᵀ; exclusive-prefix G correction
    ``G_t += ΔS_t·C_{t-1}``; ``o_t = q_tᵀ(S_t C_t − G_t)`` (γ=1, no ridge/normalize).
    """
    B, H, T, K = key.shape
    V = value.shape[-1]
    device, dtype = key.device, torch.float32
    q = query.to(dtype)
    k = key.to(dtype)
    v = value.to(dtype)
    S = torch.zeros(B, H, K, K, device=device, dtype=dtype)
    C = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    G = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    outs = []
    for t in range(T):
        dS = torch.einsum("bhk,bhl->bhkl", k[:, :, t], k[:, :, t])
        dC = torch.einsum("bhk,bhv->bhkv", q[:, :, t], v[:, :, t])
        G = G + torch.einsum("bhkl,bhlv->bhkv", dS, C)   # exclusive C_{t-1}
        S = S + dS
        C = C + dC
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S @ C - G))
    return torch.stack(outs, dim=2)


class HLALayer(torch.nn.Module):
    """Masked second-order unnormalized HLA as a two-node public K2 graph (Nest).

    No projections in the pinned Algorithm 1 — q/k/v are the inputs. The layer is
    the two dependent additive K2 calls with explicit S and F state edges.
    """

    def __init__(self, head_dim: int, *, target: str = "reference", intent: str = "inference") -> None:
        super().__init__()
        self.head_dim = head_dim
        K = V = head_dim
        document = {
            "schema_version": 2, "name": "hla_two_k2", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", K]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "H", "T", K]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", V]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "initial_S", "dtype": "float32", "shape": ["B", "H", K, K]},
                    {"name": "initial_F", "dtype": "float32", "shape": ["B", "H", K, V]},
                ],
                "nodes": [
                    {
                        # S_t = S + k kᵀ ; r_t = S_t q_t (symmetric S → exact orientation).
                        "id": "s_state", "op": "linear_delta_state",
                        "inputs": ["query", "key", "key", "beta", "initial_S"],
                        "outputs": ["r", "final_S"],
                        "params": {
                            "delta": False, "gate_scope": "none", "read_timing": "after_update",
                            "scale_rule": "one", "normalized": False,
                            "roles": {"query": "query", "key": "key", "value": "key",
                                      "beta": "beta", "initial_state": "initial_S"},
                        },
                    },
                    {
                        # F_t = F + r vᵀ ; o_t = q_tᵀ F_t.
                        "id": "f_state", "op": "linear_delta_state",
                        "inputs": ["query", "r", "value", "beta", "initial_F"],
                        "outputs": ["output", "final_F"],
                        "params": {
                            "delta": False, "gate_scope": "none", "read_timing": "after_update",
                            "scale_rule": "one", "normalized": False,
                            "roles": {"query": "query", "key": "r", "value": "value",
                                      "beta": "beta", "initial_state": "initial_F"},
                        },
                    },
                ],
                "outputs": ["output", "final_S", "final_F"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        initial_S: torch.Tensor | None = None,
        initial_F: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``query``/``key`` ``[B,H,T,K]``, ``value`` ``[B,H,T,V]`` → output ``[B,H,T,V]``."""
        B, H, T, K = key.shape
        V = value.shape[-1]
        device = key.device
        if initial_S is None:
            initial_S = torch.zeros(B, H, K, K, device=device)
        if initial_F is None:
            initial_F = torch.zeros(B, H, K, V, device=device)
        beta = torch.ones(B, H, T, device=device)  # additive law: no gate, β ≡ 1
        out = self._plan.execute(
            query=query.float(), key=key.float(), value=value.float(), beta=beta,
            initial_S=initial_S.float(), initial_F=initial_F.float(),
        )
        return out["output"]


__all__ = ["HLALayer", "hla_serial_reference"]
