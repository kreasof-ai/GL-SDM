"""External model module: SDM sparse delta memory layer (arch-047).

Verified composition-now row **with a recorded caveat** (sweep row-047): the
update/read law is exactly the closed U3.D decayed-delta contract with a
certified external route trace, and the product-key router is external by
design — verified against lingua/sparse_delta_memory/layer.py @ 183e7df8. The
caveat: the pinned router's tie policy is backend-dependent ``torch.topk``
(not the R.PK HIGHEST_ADDRESS tie rule) and route weights use a scaled
softmax; an exact R.PK claim additionally requires a highest-address tie
realization. This module therefore claims the update/read law (U3.D) plus the
external route/trace composition, not the router tie-policy identity.

Composition (matching the pinned layer's external call graph):
product-key score projections (W_score + read/write biases, value/gate
projection) are external; the K3 route→update→read graph (two
``sparse_route_generation`` nodes + one ``sparse_state_mixer`` node, decayed-
delta update, after-update read) executes through the public compile path; the
output projection is external.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class SparseDeltaMemoryLayer(torch.nn.Module):
    """One SDM sparse-memory layer: external projections + typed K3 graph.

    Parameters mirror the pinned layer contract: ``width = heads * value_dim``,
    product-key scores of width ``2 * factor_extent`` per head, ``reads`` /
    ``writes`` route widths over ``slots_per_partition`` slots. The persistent
    memory buffer is external state carried across calls.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        value_dim: int,
        slots_per_partition: int,
        reads: int,
        writes: int,
        batch_size: int,
        *,
        bias: bool = False,
        target: str = "native",
        intent: str = "training",
    ) -> None:
        super().__init__()
        if width != heads * value_dim:
            raise ValueError("width must equal heads * value_dim")
        factor = round(slots_per_partition ** 0.5)
        if factor * factor != slots_per_partition:
            raise ValueError("slots_per_partition must be a perfect square (product key)")
        self.width = width
        self.heads = heads
        self.value_dim = value_dim
        self.factor_extent = factor
        self.slots_per_partition = slots_per_partition
        self.reads = reads
        self.writes = writes
        self.batch_size = batch_size

        self.score = torch.nn.Linear(width, heads * 2 * factor, bias=bias)
        self.read_score_bias = torch.nn.Parameter(torch.zeros(heads, 2 * factor))
        self.write_score_bias = torch.nn.Parameter(torch.empty(heads, 2 * factor))
        torch.nn.init.normal_(self.write_score_bias, std=0.002)
        self.value_gate = torch.nn.Linear(width, heads * (value_dim + 2), bias=bias)
        self.output = torch.nn.Linear(width, width, bias=bias)
        # One memory bank per (batch, head) — the graph's parallel dim P.
        self.register_buffer(
            "persistent_memory",
            torch.zeros(batch_size * heads, slots_per_partition, value_dim),
            persistent=False,
        )
        self._pending_state: torch.Tensor | None = None

        document = {
            "schema_version": 2,
            "name": "sdm_layer",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "read_scores", "dtype": "bfloat16", "shape": ["P", "T", 2 * factor]},
                    {"name": "write_scores", "dtype": "bfloat16", "shape": ["P", "T", 2 * factor]},
                    {"name": "values", "dtype": "bfloat16", "shape": ["P", "T", value_dim]},
                    {"name": "beta", "dtype": "float32", "shape": ["P", "T", 1]},
                    {"name": "log_decay", "dtype": "float32", "shape": ["P", "T", 1]},
                    {"name": "memory", "dtype": "bfloat16",
                     "shape": ["P", slots_per_partition, value_dim]},
                ],
                "nodes": [
                    {
                        "id": "read_routes",
                        "op": "sparse_route_generation",
                        "inputs": ["read_scores"],
                        "outputs": ["read_addresses", "read_weights"],
                        "params": {
                            "source_extent": slots_per_partition,
                            "route_width": reads,
                        },
                    },
                    {
                        "id": "write_routes",
                        "op": "sparse_route_generation",
                        "inputs": ["write_scores"],
                        "outputs": ["write_addresses", "write_weights"],
                        "params": {
                            "source_extent": slots_per_partition,
                            "route_width": writes,
                        },
                    },
                    {
                        "id": "update_and_read",
                        "op": "sparse_state_mixer",
                        "inputs": [
                            "memory", "read_addresses", "read_weights",
                            "write_addresses", "write_weights",
                            "values", "beta", "log_decay",
                        ],
                        "outputs": ["output", "final_state"],
                        "params": {
                            "slots_per_partition": slots_per_partition,
                            "value_dim": value_dim,
                            "writes": writes,
                            "reads": reads,
                            "operation": "update",
                            "read_timing": "after_update",
                            "update_rule": "decayed_delta",
                            "collision_policy": "ordered",
                            "mode": "training",
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def _project(self, x: torch.Tensor):
        b, t, _ = x.shape
        h, f, d = self.heads, self.factor_extent, self.value_dim
        common = self.score(x).view(b, t, h, 2 * f).permute(0, 2, 1, 3)
        read_scores = (common + self.read_score_bias[None, :, None]).reshape(b * h, t, 2 * f)
        write_scores = (common + self.write_score_bias[None, :, None]).reshape(b * h, t, 2 * f)
        projected = self.value_gate(x).view(b, t, h, d + 2).permute(0, 2, 1, 3)
        values = projected[..., :d].reshape(b * h, t, d).contiguous()
        beta = torch.sigmoid(projected[..., d : d + 1]).reshape(b * h, t, 1)
        log_decay = -F.softplus(projected[..., d + 1 :]).reshape(b * h, t, 1)
        return (
            read_scores.contiguous(), write_scores.contiguous(), values,
            beta.contiguous(), log_decay.contiguous(),
        )

    def detach_state(self) -> None:
        """Persist the last forward's final state (train/loop.py's detach_state)."""
        if self._pending_state is not None:
            with torch.no_grad():
                self.persistent_memory.copy_(self._pending_state.detach())
            self._pending_state = None

    def reset_state(self) -> None:
        self.persistent_memory.zero_()
        self._pending_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        read_scores, write_scores, values, beta, log_decay = self._project(x)
        result = self._plan.execute(
            read_scores=read_scores.to(torch.bfloat16),
            write_scores=write_scores.to(torch.bfloat16),
            values=values.to(torch.bfloat16),
            beta=beta,
            log_decay=log_decay,
            memory=self.persistent_memory.to(torch.bfloat16),
        )
        self._pending_state = result["final_state"]
        readings = result["output"].view(b, self.heads, t, self.value_dim)
        readings = readings.permute(0, 2, 1, 3).reshape(b, t, self.width)
        return self.output(readings.to(self.output.weight.dtype))


__all__ = ["SparseDeltaMemoryLayer"]
