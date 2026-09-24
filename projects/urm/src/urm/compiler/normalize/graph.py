"""Normalization: a versioned JSON graph document becomes a typed IR program.

This is the single entry point that turns a recipe document into a
:class:`~urm.ir.program.SemanticProgram` of typed operations. It owns no
equation semantics itself; it maps each declared node onto the registered typed
IR operation, resolves operand bindings (graph inputs, declared state, and
earlier node outputs), and lets :meth:`SemanticProgram.build` validate the
resulting graph. Unknown operations, dangling edges, wrong operand arity, and
invalid state order are rejected here or by the program validator — never
silently defaulted.

The document schema is ``recipes/schema/graph-recipe.schema.json``; the loader
in :mod:`urm.frontend.recipes` parses and validates the document before this
normalizer runs.
"""

from __future__ import annotations

from typing import Any

from urm.ir.program import (
    CapacityPolicy,
    DType,
    EpilogueSpec,
    LogicalDomain,
    Matmul,
    MergePolicy,
    OrderedRecurrence,
    RouteSpec,
    Score,
    Select,
    Gather,
    SelectionKind,
    ScoreNormalization,
    SemanticNode,
    SemanticProgram,
    SparseRouteGeneration,
    SparseRouteSelectionSpec,
    SparseStateExecutionMode,
    SparseStateMixerAccess,
    SparseStateMixerSpec,
    SparseStateOperation,
    SparseReadTiming,
    StateRead,
    StateUpdate,
    TensorHandle,
    Transform,
    TransformKind,
    WeightedReduce,
)


class NormalizeError(ValueError):
    """Raised when a recipe document cannot be normalized into typed IR."""


def _enum(enum_type: Any, value: Any, *, field: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as error:
        legal = ", ".join(member.value for member in enum_type)
        raise NormalizeError(f"invalid {field} {value!r}; expected one of: {legal}") from error


def _route_spec(params: dict[str, Any]) -> RouteSpec:
    return RouteSpec(
        query_domain=_enum(LogicalDomain, params["query_domain"], field="query_domain"),
        source_domain=_enum(LogicalDomain, params["source_domain"], field="source_domain"),
        selection=_enum(SelectionKind, params["selection"], field="selection"),
        normalization=_enum(ScoreNormalization, params["normalization"], field="normalization"),
        top_k=params.get("top_k"),
        threshold=params.get("threshold"),
        page_size=params.get("page_size"),
        capacity_policy=_enum(
            CapacityPolicy, params.get("capacity_policy", "dropless"), field="capacity_policy"
        ),
        deterministic=bool(params.get("deterministic", True)),
        causal=bool(params.get("causal", False)),
    )


def _build_node(node: dict[str, Any], *, index: int) -> SemanticNode:
    node_id = node["id"]
    op = node["op"]
    inputs = tuple(node["inputs"])
    outputs = tuple(node["outputs"])
    params = node.get("params", {}) or {}
    if not outputs:
        raise NormalizeError(f"node {node_id!r} must declare at least one output")

    if op == "score":
        return Score(name=node_id, inputs=inputs, outputs=outputs, spec=_route_spec(params))
    if op == "select":
        return Select(name=node_id, inputs=inputs, outputs=outputs, spec=_route_spec(params))
    if op == "gather":
        return Gather(name=node_id, inputs=inputs, outputs=outputs, spec=_route_spec(params))
    if op == "weighted_reduce":
        epilogue = params.get("epilogue")
        epilogue_spec = None
        if epilogue is not None:
            kind = _enum(TransformKind, epilogue["kind"], field="epilogue.kind")
            epilogue_spec = EpilogueSpec(kind=kind, scale=epilogue["scale"])
        return WeightedReduce(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            spec=_route_spec(params),
            epilogue=epilogue_spec,
        )
    if op == "matmul":
        return Matmul(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            transpose_rhs=bool(params.get("transpose_rhs", False)),
        )
    if op == "transform":
        return Transform(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            kind=_enum(TransformKind, params["kind"], field="transform.kind"),
        )
    if op == "ordered_recurrence":
        return OrderedRecurrence(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            algorithm=str(params["transition"]),
        )
    if op == "state_read":
        return StateRead(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            state=str(params["state"]),
            versioned=bool(params.get("versioned", True)),
        )
    if op == "state_update":
        return StateUpdate(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            state=str(params["state"]),
            policy=_enum(MergePolicy, params["policy"], field="state_update.policy"),
            commit_boundary=bool(params.get("commit_boundary", False)),
        )
    if op == "sparse_route_generation":
        # parallel/sequence are runtime batch dims; the recipe declares the
        # equation (source_extent, route_width). They are carried as 1 here and
        # re-materialized by the binder from the actual operand shapes.
        return SparseRouteGeneration(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            spec=SparseRouteSelectionSpec(
                parallel=int(params.get("parallel", 1)),
                sequence=int(params.get("sequence", 1)),
                source_extent=int(params["source_extent"]),
                route_width=int(params["route_width"]),
                dtype=DType.BFLOAT16,
            ),
        )
    if op == "sparse_state_mixer":
        return SparseStateMixerAccess(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            spec=SparseStateMixerSpec(
                parallel=int(params.get("parallel", 1)),
                sequence=int(params.get("sequence", 1)),
                slots_per_partition=int(params["slots_per_partition"]),
                value_dim=int(params["value_dim"]),
                writes=int(params["writes"]),
                reads=int(params["reads"]),
                dtype=DType.BFLOAT16,
                operation=_enum(
                    SparseStateOperation,
                    params.get("operation", "update"),
                    field="sparse_state_mixer.operation",
                ),
                read_timing=_enum(
                    SparseReadTiming,
                    params.get("read_timing", "after_update"),
                    field="sparse_state_mixer.read_timing",
                ),
                mode=_enum(
                    SparseStateExecutionMode,
                    params.get("mode", "inference"),
                    field="sparse_state_mixer.mode",
                ),
            ),
        )
    raise NormalizeError(f"node {node_id!r}: unknown operation {op!r}")


def normalize_graph_document(document: dict[str, Any]) -> SemanticProgram:
    """Normalize a validated graph recipe document into a typed IR program.

    ``document`` must already satisfy the graph recipe schema (the loader
    enforces that). This function resolves operand bindings and operation
    semantics; structural validation (unknown tensors, duplicate definitions,
    unbound outputs, multiple commits) is enforced by
    :meth:`SemanticProgram.build`.
    """
    try:
        graph = document["graph"]
        name = document["name"]
    except KeyError as error:
        raise NormalizeError(f"document is missing required key {error}") from error

    inputs = tuple(
        TensorHandle(
            name=entry["name"],
            dtype=_enum(DType, entry["dtype"], field=f"input {entry['name']!r} dtype"),
            shape=tuple(entry.get("shape", ("...",))),
        )
        for entry in graph["inputs"]
    )
    nodes = tuple(
        _build_node(node, index=index) for index, node in enumerate(graph["nodes"])
    )
    outputs = tuple(graph["outputs"])
    try:
        return SemanticProgram.build(
            name=f"graph:{name}", inputs=inputs, ops=nodes, outputs=outputs
        )
    except Exception as error:
        raise NormalizeError(f"graph {name!r} failed validation: {error}") from error


__all__ = ["NormalizeError", "normalize_graph_document"]
