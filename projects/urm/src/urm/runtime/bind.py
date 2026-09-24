"""Runtime binding of compiled plans to their launchers.

:class:`BoundGraphPlan` is the generic, plan-authority executor. It binds a
compiled graph (a :class:`~urm.compiler.pipeline.CompilationResult` carrying
the typed program plus the per-step plan) and executes the plan steps in graph
order. Each step names a typed operation and its selected anchor; operands are
resolved by name from the caller's inputs and earlier step outputs. Execution
never inspects a recipe name, a backend enum, or a retained spec to redispatch —
the serialized plan and the typed node are the only authority. Incomplete or
tampered plans fail before tensor execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from urm.compiler.pipeline import CompilationResult
from urm.ir.program import (
    ScoreNormalization,
    SemanticProgram,
    SparseRouteGeneration,
    SparseStateMixerAccess,
    SparseStateOperation,
    WeightedReduce,
)


class PlanBindingError(RuntimeError):
    """Raised when a serialized plan cannot be bound or executed as written."""


def _torch() -> Any:
    import torch

    return torch


def _execute_k1_attention_node(
    node: WeightedReduce, anchor: str, tensors: dict[str, Any]
) -> Any:
    """Execute one typed K1 attention node against its selected anchor.

    The node's :class:`RouteSpec` carries the equation semantics (softmax
    normalization, causal masking, dense selection); the anchor selects the
    implementation. Operands are bound by name from the tensor table.
    """
    torch = _torch()
    query = tensors["query"]
    key = tensors["key"]
    value = tensors["value"]
    score_bias = tensors.get("score_bias")
    attention_mask = tensors.get("attention_mask")
    causal = bool(node.spec.causal)
    key_dim = query.shape[-1]
    scale = key_dim ** -0.5

    if anchor == "urm_native_k1_online_softmax_v1":
        from urm.backends.triton.k1.online import execute_online_softmax

        return execute_online_softmax(
            query,
            key,
            value,
            attention_mask=attention_mask,
            score_bias=score_bias,
            causal=causal,
            scale=scale,
        )
    if anchor == "torch.nn.functional.scaled_dot_product_attention":
        # Trusted library anchor; semantics identical to the reference equation.
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        def _head_major(mask):
            # B,T,T -> B,1,T,T; B,H,T,T passes through; T,T -> 1,1,T,T.
            if mask.dim() == 2:
                return mask.view(1, 1, *mask.shape)
            if mask.dim() == 3:
                return mask.unsqueeze(1)
            return mask

        attn_mask = None
        if attention_mask is not None:
            attn_mask = _head_major(attention_mask)
        if score_bias is not None:
            bias = _head_major(score_bias)
            attn_mask = bias if attn_mask is None else attn_mask & bias
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal and attn_mask is None, scale=scale
        )
        return out.transpose(1, 2)
    if anchor == "urm.unified.k1.softmax_reference.v1":
        # Transparent eager reference: materialize scores, normalize, reduce.
        q = query.to(torch.float32).transpose(1, 2)
        k = key.to(torch.float32).transpose(1, 2)
        v = value.to(torch.float32).transpose(1, 2)
        if k.shape[1] != q.shape[1]:  # GQA/MQA: expand groups
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if score_bias is not None:
            scores = scores + score_bias.to(torch.float32)
        if causal:
            length_q, length_k = scores.shape[-2], scores.shape[-1]
            mask = torch.ones(length_q, length_k, dtype=torch.bool, device=scores.device).tril_(
                diagonal=length_k - length_q
            )
            scores = scores.masked_fill(~mask, float("-inf"))
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() == 2:
                mask = mask.view(1, 1, *mask.shape)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        return torch.matmul(probs, v).transpose(1, 2).to(value.dtype)
    raise PlanBindingError(f"no graph executor for K1 attention anchor {anchor!r}")


def _execute_sparse_route_node(node: SparseRouteGeneration, anchor: str, tensors: dict[str, Any]) -> tuple[Any, Any]:
    """Execute one sparse route-generation node: scores -> (addresses, weights)."""
    if anchor != "urm_native_sparse_route_selection_v0":
        raise PlanBindingError(
            f"no graph executor for sparse route anchor {anchor!r}"
        )
    from urm.backends.triton.k3.route import sparse_route_selection

    (scores_name,) = node.inputs
    scores = tensors[scores_name]
    spec = node.spec
    addresses, weights = sparse_route_selection(
        scores,
        spec.source_extent,
        spec.route_width,
    )
    return addresses, weights


def _execute_sparse_state_node(node: SparseStateMixerAccess, anchor: str, tensors: dict[str, Any]) -> tuple[Any, Any]:
    """Execute one sparse state-mixer node: ordered update + weighted read.

    The recipe declares the equation (slots, value_dim, widths, read timing);
    parallel/sequence are runtime batch dims re-materialized from the operand
    shapes. The narrow state launcher certifies and executes the routes.
    """
    if anchor not in {
        "urm_native_sparse_state_mixer_v0",
        "urm.unified.k3.sparse_delta_reference.v1",
    }:
        raise PlanBindingError(
            f"no graph executor for sparse state anchor {anchor!r}"
        )
    from dataclasses import replace as _replace

    bound = {name: tensors.get(name) for name in node.inputs}
    memory = bound["memory"]
    values = bound.get("values")
    parallel, sequence = None, None
    if values is not None:
        parallel, sequence = int(values.shape[0]), int(values.shape[1])
    elif bound.get("read_addresses") is not None:
        parallel = int(bound["read_addresses"].shape[0])
        sequence = int(bound["read_addresses"].shape[1])
    spec = _replace(
        node.spec,
        parallel=parallel or node.spec.parallel,
        sequence=sequence or node.spec.sequence,
    )

    if anchor == "urm.unified.k3.sparse_delta_reference.v1":
        # Independent differentiable reference (transparent loop).
        from urm.backends.reference.torch.k3 import torch_sparse_state_mixer

        outputs, state = torch_sparse_state_mixer(
            memory,
            bound["read_addresses"],
            bound["read_weights"],
            write_indices=bound.get("write_addresses"),
            write_weights=bound.get("write_weights"),
            values=values,
            beta=bound.get("beta"),
            log_decay=bound.get("log_decay"),
        )
        return outputs, state

    # Native K3 narrow launcher. The routes are produced by this graph's own
    # route-generation nodes (the native route kernels), so they are trusted
    # well-formed: use the host-side structural certification, not the GPU value
    # scan, which keeps execution torch.compile/fullgraph-traceable.
    from urm.backends.triton.k3.state_launcher import (
        CertifiedSparseStateRoutes,
        SparseState,
        TritonSparseStateMixerBackend,
    )

    routes = CertifiedSparseStateRoutes.certify_trusted(
        spec,
        bound["read_addresses"],
        bound["read_weights"],
        write_indices=bound.get("write_addresses"),
        write_weights=bound.get("write_weights"),
    )
    backend = TritonSparseStateMixerBackend(spec)
    # The operands are produced by this graph's route nodes and the model's
    # projections (trusted), so bind through the cheap generated-routes bridge;
    # the full ``prepare`` re-scans values on GPU, which is not fullgraph-safe.
    prepared = backend._prepare_generated_routes(
        routes,
        values=values,
        beta=bound.get("beta"),
        log_decay=bound.get("log_decay"),
    )
    state = SparseState(memory=memory, sequence_length=0)
    readings, new_state = backend.execute(state, prepared)
    return readings, new_state.memory


@dataclass(frozen=True, slots=True)
class BoundGraphPlan:
    """A compiled graph bound for execution; the plan is the only authority.

    ``compilation.plan.steps`` are executed in order. Each step's ``note`` names
    the typed op in ``compilation.rewritten_program`` it lowers; the step's
    ``anchor`` is the compiler-selected capability. Operands bind by name from
    the caller's inputs and earlier step outputs.
    """

    compilation: CompilationResult

    @property
    def program(self) -> SemanticProgram:
        return self.compilation.rewritten_program

    def _validate(self) -> None:
        plan = self.compilation.plan
        program = self.compilation.rewritten_program
        op_names = set(program.op_names)
        for step in plan.steps:
            if step.kind != "anchor_dispatch":
                raise PlanBindingError(
                    f"plan step {step.step_id}: unsupported kind {step.kind!r}"
                )
            if step.note not in op_names:
                raise PlanBindingError(
                    f"plan step {step.step_id}: names unknown op {step.note!r}"
                )
            if not step.anchor:
                raise PlanBindingError(
                    f"plan step {step.step_id}: no selected anchor"
                )

    def execute(self, **inputs: Any) -> dict[str, Any]:
        """Execute the plan in graph order; return the program's outputs."""
        self._validate()
        program = self.compilation.rewritten_program
        plan = self.compilation.plan

        tensors: dict[str, Any] = {}
        declared_inputs = {handle.name for handle in program.inputs}
        for name, value in inputs.items():
            if name not in declared_inputs:
                raise PlanBindingError(f"unexpected operand {name!r}")
            tensors[name] = value
        missing = declared_inputs - set(tensors)
        # Optional operands may be absent; only enforce that at dispatch.
        tensors.update({name: None for name in missing})

        step_by_note: dict[str, Any] = {step.note: step for step in plan.steps}
        for op in program.ops:
            step = step_by_note.get(op.name)
            if step is None:
                raise PlanBindingError(
                    f"plan has no dispatch step for op {op.name!r}"
                )
            operand_names = [n for n in op.inputs if tensors.get(n) is not None]
            if isinstance(op, WeightedReduce) and (
                op.spec.normalization is ScoreNormalization.SOFTMAX
            ):
                result = _execute_k1_attention_node(op, step.anchor, tensors)
            elif isinstance(op, SparseRouteGeneration):
                result = _execute_sparse_route_node(op, step.anchor, tensors)
            elif isinstance(op, SparseStateMixerAccess):
                result = _execute_sparse_state_node(op, step.anchor, tensors)
            else:
                raise PlanBindingError(
                    f"op {op.name!r} ({type(op).__name__}) has no graph executor "
                    f"for anchor {step.anchor!r}"
                )
            if len(op.outputs) == 1:
                tensors[op.outputs[0]] = result
            else:
                for out_name, out_value in zip(op.outputs, result):
                    tensors[out_name] = out_value

        # The declared graph outputs are authoritative; intermediate node outputs
        # (e.g. a state-mixer's ``final_state``) are also exposed so a stateful
        # consumer can bind the persistent state contract.
        return {name: tensors[name] for name in dict.fromkeys(
            (*program.outputs, *(o for op in program.ops for o in op.outputs))
        )}

    def serialized_plan(self) -> dict[str, object]:
        return self.compilation.plan.to_dict()


__all__ = [
    "BoundGraphPlan",
    "PlanBindingError",
]
