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
    K1Descriptor,
    K1HeadMap,
    K1ReducerLaw,
    K1ScaleRule,
    K1ScoreLaw,
    K2GateScope,
    K2ReadTiming,
    K2ScaleRule,
    LinearDeltaSpec,
    LinearDeltaState,
    LogicalDomain,
    Matmul,
    Merge,
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


_K1_ROLES = frozenset(
    {"query", "key", "value", "score_bias", "attention_mask", "scale", "channel_gate"}
)
_K2_ROLES = frozenset(
    {"query", "key", "value", "beta", "log_decay", "initial_state", "scale",
     "erase_gate", "write_gate", "predict_key", "alpha", "low_rank_beta"}
)


def _roles(
    raw: Any, *, legal: frozenset[str], node_id: str
) -> tuple[tuple[str, str], ...]:
    """Validate a role→edge mapping; unknown roles or non-string maps reject."""
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise NormalizeError(f"node {node_id!r}: roles must be an object")
    roles: list[tuple[str, str]] = []
    for role, edge in raw.items():
        if role not in legal:
            raise NormalizeError(
                f"node {node_id!r}: unknown role {role!r}; legal: {sorted(legal)}"
            )
        if not isinstance(edge, str) or not edge:
            raise NormalizeError(
                f"node {node_id!r}: role {role!r} must name an edge"
            )
        roles.append((role, edge))
    return tuple(sorted(roles))


def _k1_descriptor(params: dict[str, Any], roles: tuple[tuple[str, str], ...]) -> K1Descriptor:
    head_map_raw = params.get("head_map")
    if head_map_raw is None:
        # Derive the coarsest legal map from the declared roles only: an
        # explicit scale/mask/bias role set still defaults to the shared map.
        head_map_raw = "shared"
    return K1Descriptor(
        scale_rule=_enum(K1ScaleRule, params.get("scale_rule", "key_dim_rsqrt"), field="k1.scale_rule"),
        head_map=_enum(K1HeadMap, head_map_raw, field="k1.head_map"),
        group_size=params.get("group_size"),
        causal=bool(params.get("causal", False)),
        score_bias="score_bias" in dict(roles),
        attention_mask="attention_mask" in dict(roles),
        score_law=_enum(K1ScoreLaw, params.get("score_law", "dot"), field="k1.score_law"),
        reducer_law=_enum(K1ReducerLaw, params.get("reducer_law", "softmax"), field="k1.reducer_law"),
        threshold_beta=params.get("threshold_beta"),
        relu_power=params.get("relu_power"),
        squared_sum_groups=params.get("squared_sum_groups"),
        accumulation_dtype=DType.FLOAT32,
    )


_K2_PARAMS = frozenset(
    {"delta", "gate_scope", "read_timing", "scale_rule", "normalized", "epsilon", "roles",
     "erase_gate", "write_gate", "predict_key", "low_rank", "num_deltas"}
)
_K1_PARAMS = frozenset(
    {
        "query_domain", "source_domain", "selection", "normalization", "top_k",
        "threshold", "page_size", "capacity_policy", "deterministic", "causal",
        "scale_rule", "head_map", "group_size", "roles", "epilogue", "score_law",
        "reducer_law", "threshold_beta", "relu_power", "squared_sum_groups",
    }
)


def _reject_unknown_params(params: dict[str, Any], legal: frozenset[str], node_id: str) -> None:
    unknown = set(params) - legal
    if unknown:
        raise NormalizeError(
            f"node {node_id!r}: unknown param(s) {sorted(unknown)}; legal: {sorted(legal)}"
        )


def _linear_delta_spec(params: dict[str, Any], roles: Any) -> LinearDeltaSpec:
    role_names = set(dict(roles))
    return LinearDeltaSpec(
        delta=bool(params.get("delta", True)),
        gate_scope=_enum(K2GateScope, params.get("gate_scope", "none"), field="k2.gate_scope"),
        read_timing=_enum(
            K2ReadTiming, params.get("read_timing", "after_update"), field="k2.read_timing"
        ),
        scale_rule=_enum(K2ScaleRule, params.get("scale_rule", "one"), field="k2.scale_rule"),
        normalized=bool(params.get("normalized", False)),
        epsilon=float(params.get("epsilon", 1e-6)),
        erase_gate="erase_gate" in role_names,
        write_gate="write_gate" in role_names,
        predict_key="predict_key" in role_names,
        low_rank=bool(params.get("low_rank", False)),
        num_deltas=int(params.get("num_deltas", 1)),
    )


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
        _reject_unknown_params(params, _K1_PARAMS, node_id)
        epilogue = params.get("epilogue")
        epilogue_spec = None
        if epilogue is not None:
            kind = _enum(TransformKind, epilogue["kind"], field="epilogue.kind")
            epilogue_spec = EpilogueSpec(kind=kind, scale=epilogue["scale"])
        roles = _roles(params.get("roles"), legal=_K1_ROLES, node_id=node_id)
        normalization = _enum(
            ScoreNormalization, params["normalization"], field="normalization"
        )
        k1 = (
            _k1_descriptor(params, roles)
            if normalization is ScoreNormalization.SOFTMAX
            else None
        )
        return WeightedReduce(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            spec=_route_spec(params),
            epilogue=epilogue_spec,
            roles=roles,
            k1=k1,
        )
    if op == "linear_delta_state":
        _reject_unknown_params(params, _K2_PARAMS, node_id)
        roles = _roles(params.get("roles"), legal=_K2_ROLES, node_id=node_id)
        if not roles:
            raise NormalizeError(
                f"node {node_id!r}: linear_delta_state requires an explicit roles mapping"
            )
        # log_decay is required unless gate_scope is "none" (no decay); the
        # executor synthesizes the zero schedule for that case.
        required = ["query", "key", "value", "beta", "initial_state"]
        if params.get("gate_scope", "none") != "none":
            required.append("log_decay")
        missing = [r for r in required if r not in dict(roles)]
        if missing:
            raise NormalizeError(
                f"node {node_id!r}: linear_delta_state is missing required roles {missing}"
            )
        return LinearDeltaState(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            spec=_linear_delta_spec(params, roles),
            roles=roles,
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
    if op == "merge":
        return Merge(
            name=node_id,
            inputs=inputs,
            outputs=outputs,
            coefficients=tuple(float(c) for c in params.get("coefficients", ())),
            scale_operands=tuple(str(s) for s in params.get("scale_operands", ())),
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
