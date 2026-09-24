"""Runtime binding of compiled plans to their providers.

:class:`BoundGraphPlan` is the generic, plan-authority executor. It binds a
compiled graph (a :class:`~urm.compiler.pipeline.CompilationResult` carrying
the typed program plus the per-step plan) and executes the plan steps in graph
order. Each step names a typed operation and its selected anchor; the anchor
maps to exactly one :class:`~urm.backends.provider.Provider` in the dispatch
table, which accepts or declines the closed request and then executes it.

The runtime owns no equation logic: operands are bound by *role* (never by
global tensor name), the equation is the node's closed descriptor, and the
provider implements it. Incomplete or tampered plans fail before tensor
execution; an unknown anchor name fails the dispatch lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from urm.backends.provider import ProviderFamily, ProviderRequest
from urm.backends.providers import (
    K1NativeTritonProvider,
    K1SdpaLibraryProvider,
    K1TorchReferenceProvider,
    K2NativeDiagonalProvider,
    K2NativeMatrixProvider,
    K2TorchReferenceProvider,
    K3NativeTritonProvider,
    K3RouteNativeTritonProvider,
    K3TorchReferenceProvider,
)
from urm.compiler.pipeline import CompilationResult
from urm.ir.program import (
    LinearDeltaState,
    ScoreNormalization,
    SemanticProgram,
    SparseRouteGeneration,
    SparseStateMixerAccess,
    WeightedReduce,
)


class PlanBindingError(RuntimeError):
    """Raised when a serialized plan cannot be bound or executed as written."""


# The single dispatch table: serialized anchor name → its one provider. An
# anchor name that is not present here is an unknown provider and fails the
# lookup; no name ever maps to two providers.
_PROVIDERS = {
    provider.name: provider
    for provider in (
        K1TorchReferenceProvider(),
        K1SdpaLibraryProvider(),
        K1NativeTritonProvider(),
        K2TorchReferenceProvider(),
        K2NativeDiagonalProvider(),
        K2NativeMatrixProvider(),
        K3TorchReferenceProvider(),
        K3NativeTritonProvider(),
        K3RouteNativeTritonProvider(),
    )
}


_K1_OPTIONAL_ROLES = ("score_bias", "attention_mask", "scale")
_K2_REQUIRED_ROLES = ("query", "key", "value", "beta", "log_decay", "initial_state")


def _bind_roles(
    op: Any,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    tensors: dict[str, Any],
) -> dict[str, Any]:
    """Bind an op's operands by its closed role vocabulary.

    A role-bearing op resolves each role through its mapping; the legacy
    positional form (K1 only) maps the first three inputs to query/key/value.
    A required role that resolves to a missing tensor is a binding error.
    """
    roles = dict(getattr(op, "roles", ()) or ())
    if roles:
        bound = {
            role: tensors.get(edge)
            for role, edge in roles.items()
            if role in (*required, *optional)
        }
    else:
        names = list(op.inputs)
        bound = {}
        for role, name in zip(required, names):
            bound[role] = tensors.get(name)
        for role, name in zip(optional, names[len(required):]):
            bound[role] = tensors.get(name)
    for role in required:
        if bound.get(role) is None:
            raise PlanBindingError(f"op {op.name!r}: role {role!r} is unbound")
    return bound


def _family_and_descriptor(op: SemanticNode) -> tuple[str, Any, dict[str, Any]]:
    """Map a typed node to its provider family, closed descriptor and operands.

    Returns ``(family, descriptor, roles)``; the caller binds roles to tensors.
    """
    if isinstance(op, WeightedReduce) and op.spec.normalization is ScoreNormalization.SOFTMAX:
        if op.k1 is None:
            raise PlanBindingError(f"K1 node {op.name!r} carries no closed descriptor")
        return (
            ProviderFamily.K1,
            op.k1,
            dict(op.roles) if op.roles else {},
        )
    if isinstance(op, LinearDeltaState):
        return (ProviderFamily.K2, op.spec, dict(op.roles))
    if isinstance(op, SparseStateMixerAccess):
        return (ProviderFamily.K3, op.spec, {})
    if isinstance(op, SparseRouteGeneration):
        return (ProviderFamily.K3_ROUTE, op.spec, {})
    raise PlanBindingError(f"op {op.name!r} ({type(op).__name__}) has no provider family")


def _operands_for(
    op: SemanticNode, family: str, tensors: dict[str, Any]
) -> dict[str, Any]:
    """Role-bind (K1/K2) or positionally bind (K3) an op's operand tensors."""
    if family is ProviderFamily.K1:
        return _bind_roles(op, ("query", "key", "value"), _K1_OPTIONAL_ROLES, tensors)
    if family is ProviderFamily.K2:
        return _bind_roles(op, _K2_REQUIRED_ROLES, ("scale",), tensors)
    if family is ProviderFamily.K3:
        # K3 roles are the frozen operand names of the sparse-state contract.
        bound = {name: tensors.get(name) for name in op.inputs}
        if bound.get("memory") is None or bound.get("read_addresses") is None:
            raise PlanBindingError(f"K3 node {op.name!r}: memory/read_addresses unbound")
        return bound
    if family is ProviderFamily.K3_ROUTE:
        (scores_name,) = op.inputs
        scores = tensors.get(scores_name)
        if scores is None:
            raise PlanBindingError(f"K3 route node {op.name!r}: scores unbound")
        return {"scores": scores}
    raise PlanBindingError(f"unknown provider family {family!r}")


def _execute_node(
    op: SemanticNode,
    anchor: str,
    tensors: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Dispatch one typed node to its anchor's provider and execute it."""
    provider = _PROVIDERS.get(anchor)
    if provider is None:
        raise PlanBindingError(f"unknown provider anchor {anchor!r}")
    family, descriptor, _roles = _family_and_descriptor(op)
    if provider.family != family:
        raise PlanBindingError(
            f"anchor {anchor!r} is a {provider.family} provider, not {family}"
        )
    request = ProviderRequest(
        family=family,
        descriptor=descriptor,
        mode=mode,
        accumulation_dtype="float32",
    )
    decline = provider.decline(request)
    if decline is not None:
        raise PlanBindingError(f"provider {anchor!r} declined: {decline}")
    operands = _operands_for(op, family, tensors)
    return provider.execute(request, operands)


@dataclass(frozen=True, slots=True)
class BoundGraphPlan:
    """A compiled graph bound for execution; the plan is the only authority.

    ``compilation.plan.steps`` are executed in order. Each step's ``note`` names
    the typed op in ``compilation.rewritten_program`` it lowers; the step's
    ``anchor`` selects the provider from the dispatch table. Operands bind by
    role from the caller's inputs and earlier step outputs.
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
            if step.anchor not in _PROVIDERS:
                raise PlanBindingError(
                    f"plan step {step.step_id}: unknown provider anchor {step.anchor!r}"
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

        mode = "inference"

        step_by_note: dict[str, Any] = {step.note: step for step in plan.steps}
        for op in program.ops:
            step = step_by_note.get(op.name)
            if step is None:
                raise PlanBindingError(
                    f"plan has no dispatch step for op {op.name!r}"
                )
            outputs = _execute_node(op, step.anchor, tensors, mode)
            for out_name, out_value in zip(op.outputs, outputs.values()):
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
