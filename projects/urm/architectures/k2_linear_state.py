"""Shared external base for the K2 composition rows (linear-delta / linear-attention).

Each Batch 2 row composes external projections (Q/K/V, gate, beta) with the
typed K2 ``linear_delta_state`` graph, executed through the public compile path.
The mixer is the closed canonical law (decay → retrieve → write → read) with the
descriptor fields carried in the graph params; only the projections and gate
construction are architecture-specific external stages.

The graph operates on ``[B, H, T, K]`` / ``[B, H, T, V]`` operands (the fla
fused-recurrent layout); the module transposes from the conventional
``[B, T, H*D]`` hidden-states boundary.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class K2LinearStateLayer(torch.nn.Module):
    """External projections + typed K2 linear-delta-state mixer.

    Subclasses implement ``_project(hidden)`` returning the operand dict
    (query/key/value/beta/log_decay/initial_state) and ``_postprocess(out)``.
    The descriptor is fixed by the constructor params.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        *,
        delta: bool,
        gate_scope: str,
        scale_rule: str,
        normalized: bool = False,
        read_timing: str = "after_update",
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.delta = delta
        self.gate_scope = gate_scope
        self.scale_rule = scale_rule
        self.normalized = normalized

        decay_shape = (
            ["B", "H", "T", "K", "V"] if gate_scope == "elementwise"
            else ["B", "H", "T", "K"] if gate_scope == "channel"
            else ["B", "H", "T"]
        )
        graph_inputs = [
            {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
            {"name": "key", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
            {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
            {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
            {"name": "log_decay", "dtype": "float32", "shape": decay_shape},
            {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
        ]
        node_inputs = ["query", "key", "value", "beta", "log_decay", "initial_state"]
        roles = {
            "query": "query", "key": "key", "value": "value",
            "beta": "beta", "log_decay": "log_decay", "initial_state": "initial_state",
        }
        if scale_rule == "explicit_operand":
            graph_inputs.append({"name": "scale", "dtype": "float32", "shape": []})
            node_inputs.append("scale")
            roles["scale"] = "scale"
        document = {
            "schema_version": 2,
            "name": f"k2_state:{gate_scope}",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": graph_inputs,
                "nodes": [
                    {
                        "id": "state",
                        "op": "linear_delta_state",
                        "inputs": node_inputs,
                        "outputs": ["output", "final_state"],
                        "params": {
                            "delta": delta,
                            "gate_scope": gate_scope,
                            "read_timing": read_timing,
                            "scale_rule": scale_rule,
                            "normalized": normalized,
                            "roles": roles,
                        },
                    }
                ],
                "outputs": ["output", "final_state"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(
            program, target=target, intent=CompilationIntent(intent)
        )
        # On the native tier, the plan dispatches to a fused Triton kernel through a Python
        # loop dynamo cannot trace. Make the mixer call an opaque graph break so a
        # torch.compile'd model fuses the surround while the public-path provider runs
        # eagerly — the plan, provider and equation are unchanged. We rebind the instance
        # attribute so subclass forwards that call self._run_mixer hit the disabled form.
        if target == "native":
            self._run_mixer = torch._dynamo.disable(self._run_mixer)

    def _run_mixer(self, operands: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._plan.execute(**operands)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses implement _project/_postprocess")


__all__ = ["K2LinearStateLayer"]
