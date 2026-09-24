"""Runtime binding of compiled plans to their launchers.

Two bindings live here:

- :class:`BoundGraphPlan` — the generic, plan-authority executor. It binds a
  compiled graph (a :class:`~urm.compiler.pipeline.CompilationResult` carrying
  the typed program plus the per-step plan) and executes the plan steps in
  graph order. Each step names a typed operation and its selected anchor;
  operands are resolved by name from the caller's inputs and earlier step
  outputs. Execution never inspects a recipe name, a backend enum, or a
  retained spec to redispatch — the serialized plan and the typed node are the
  only authority. Incomplete or tampered plans fail before tensor execution.
- :class:`CompiledSparseMemoryPlan` — the legacy special K3 binder, retained
  until the generic route→update→read graph path replaces it (see
  ``docs/planning/refactor-list.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from urm.compiler.pipeline import CompilationResult, UrmCompiler
from urm.ir.program import (
    ScoreNormalization,
    SemanticNode,
    SemanticProgram,
    SparseMemoryMixerSpec,
    WeightedReduce,
)
from urm.compiler.partition.k3 import plan_sparse_memory


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
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1)
        if score_bias is not None:
            bias = score_bias if score_bias.dim() == 4 else score_bias.unsqueeze(0)
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
            scores = scores.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        return torch.matmul(probs, v).transpose(1, 2).to(value.dtype)
    raise PlanBindingError(f"no graph executor for K1 attention anchor {anchor!r}")


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

        return {name: tensors[name] for name in program.outputs}

    def serialized_plan(self) -> dict[str, object]:
        return self.compilation.plan.to_dict()


@dataclass(frozen=True, slots=True)
class CompiledSparseMemoryPlan:
    """Bound executable whose dispatch is authorized by a serialized plan."""

    compilation: CompilationResult
    backend: object
    launch_config: dict[str, str | int]

    @property
    def spec(self) -> SparseMemoryMixerSpec:
        return self.backend.spec

    @property
    def read_backend(self):
        return self.backend.read_backend

    @property
    def write_backend(self):
        return self.backend.write_backend

    @property
    def state_backend(self):
        return self.backend.state_backend

    @property
    def read_spec(self):
        return self.backend.read_spec

    @property
    def write_spec(self):
        return self.backend.write_spec

    @property
    def state_spec(self):
        return self.backend.state_spec

    def prepare(self, *args, **kwargs):
        return self.backend.prepare(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self.backend.execute(*args, **kwargs)

    def serialized_plan(self) -> dict[str, object]:
        return self.compilation.plan.to_dict()


def compile_sparse_memory_plan(
    spec: SparseMemoryMixerSpec,
    *,
    compiler: UrmCompiler | None = None,
) -> CompiledSparseMemoryPlan:
    """Compile, verify, and bind the exact native Sparse Memory schedule."""
    plan = plan_sparse_memory(spec, compiler=compiler)

    from urm.backends.triton.k3.memory import TritonSparseMemoryBackend

    backend = TritonSparseMemoryBackend(spec)
    read_schedule = backend.read_backend.launch_schedule()
    state_schedule = backend.state_backend.launch_schedule()
    expected = {
        "route_block_half": read_schedule["block_half"],
        "read_route_block": read_schedule["block_route"],
        "write_route_block": (
            backend.write_backend.launch_schedule()["block_route"]
            if backend.write_backend is not None
            else 0
        ),
        "state_block_d": state_schedule["block_d"],
        "state_num_warps": state_schedule["num_warps"],
        "state_num_stages": state_schedule["num_stages"],
        "read_route_num_warps": read_schedule["num_warps"],
        "write_route_num_warps": (
            backend.write_backend.launch_schedule()["num_warps"]
            if backend.write_backend is not None
            else 4
        ),
        "route_backward_num_warps": 4,
        "route_num_stages": read_schedule["num_stages"],
    }
    mismatches = {
        key: {"serialized": plan.launch_config.get(key), "runtime": value}
        for key, value in expected.items()
        if plan.launch_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "serialized Sparse Memory schedule does not match production launch: "
            f"{mismatches}"
        )
    return CompiledSparseMemoryPlan(plan.compilation, backend, plan.launch_config)


__all__ = [
    "BoundGraphPlan",
    "CompiledSparseMemoryPlan",
    "PlanBindingError",
    "compile_sparse_memory_plan",
]
