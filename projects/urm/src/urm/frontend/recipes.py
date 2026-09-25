"""Versioned JSON recipe loaders (kernel fragments and complete model graphs).

The recipe *catalog* is declarative JSON only: ``recipes/kernels/*.json``
holds schema-v2 typed graph documents and ``recipes/architectures/*.json``
holds complete-model graphs. This module contains the loading and validation
machinery and nothing else — no architecture-named content lives in core URM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# ======================================================================
# Versioned JSON recipe loader (kernel fragments and complete model graphs).
# ======================================================================


KERNEL_SCHEMA_VERSION = 1
KERNEL_FRAGMENT = "kernel_fragment"
COMPLETE_MODEL_GRAPH = "complete_model_graph"


class RecipeError(ValueError):
    """Raised when a recipe document fails schema or semantic validation."""


@dataclass(frozen=True, slots=True)
class ArchitectureLayer:
    """One ordered layer in a complete model graph."""

    id: str
    operation: str
    external_component: str | None
    params: tuple[tuple[str, object], ...]
    state: tuple[tuple[str, object], ...]
    cache: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class ArchitectureRecipe:
    """A validated complete-model-graph recipe document."""

    name: str
    layers: tuple[ArchitectureLayer, ...]
    coverage_level: str
    coverage_validated: bool
    source_comparator: str | None


def load_architecture_recipe_document(document: dict) -> ArchitectureRecipe:
    """Validate and load one complete-model-graph recipe document.

    The document must declare ``kind: complete_model_graph`` and a non-empty
    ordered layer graph. Each layer references a registered typed operation or
    kernel-fragment recipe by name, plus any external architecture-specific
    component. The coverage level is recorded, not asserted: this loader does
    not grant source-architecture coverage.
    """
    if not isinstance(document, dict):
        raise RecipeError("architecture recipe document must be a JSON object")
    version = document.get("schema_version")
    if version != KERNEL_SCHEMA_VERSION:
        raise RecipeError(f"unsupported recipe schema_version: {version!r}")
    if document.get("kind") != COMPLETE_MODEL_GRAPH:
        raise RecipeError(
            f"architecture recipe must declare kind={COMPLETE_MODEL_GRAPH!r}"
        )
    name = document.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RecipeError("architecture recipe requires a non-empty name")
    raw_layers = document.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise RecipeError(f"architecture recipe {name!r} requires a non-empty layer graph")
    layers: list[ArchitectureLayer] = []
    seen: set[str] = set()
    for raw in raw_layers:
        if not isinstance(raw, dict):
            raise RecipeError(f"architecture recipe {name!r} layers must be objects")
        layer_id = raw.get("id")
        operation = raw.get("operation")
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise RecipeError(f"architecture recipe {name!r} layer requires an id")
        if layer_id in seen:
            raise RecipeError(f"duplicate layer id {layer_id!r} in {name!r}")
        seen.add(layer_id)
        if not isinstance(operation, str) or not operation.strip():
            raise RecipeError(f"layer {layer_id!r} requires an operation name")
        layers.append(
            ArchitectureLayer(
                id=layer_id,
                operation=operation,
                external_component=raw.get("external_component"),
                params=tuple(sorted((raw.get("params") or {}).items())),
                state=tuple(sorted((raw.get("state") or {}).items())),
                cache=tuple(sorted((raw.get("cache") or {}).items())),
            )
        )
    coverage = document.get("coverage") or {}
    source = document.get("source") or {}
    return ArchitectureRecipe(
        name=name,
        layers=tuple(layers),
        coverage_level=str(coverage.get("level", "kernel_fragment")),
        coverage_validated=bool(coverage.get("validated", False)),
        source_comparator=source.get("comparator"),
    )


def load_architecture_recipe_file(path: str | Path) -> ArchitectureRecipe:
    """Load a complete-model-graph recipe from a JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise RecipeError(f"invalid JSON in {path}: {error}") from error
    return load_architecture_recipe_document(document)


GRAPH_SCHEMA_VERSION = 2

# Closed typed-operation vocabulary for graph documents. This is the schema's
# discriminated ``op`` set; unknown operations are rejected at load.
GRAPH_OPERATIONS: frozenset[str] = frozenset(
    {
        "score",
        "select",
        "gather",
        "weighted_reduce",
        "matmul",
        "transform",
        "ordered_recurrence",
        "state_read",
        "state_update",
        "sparse_route_generation",
        "sparse_state_mixer",
        "linear_delta_state",
        "merge",
    }
)


@dataclass(frozen=True, slots=True)
class GraphRecipe:
    """A validated graph document: typed nodes, edges, and declared state.

    This is provenance-free semantic input to the compiler; coverage metadata is
    carried alongside, never inside, the graph.
    """

    name: str
    kind: str
    component_scope: str
    architecture_ids: tuple[str, ...]
    required_external_stages: tuple[str, ...]
    document: dict


def _validate_graph_document(document: dict) -> None:
    if not isinstance(document, dict):
        raise RecipeError("recipe document must be a JSON object")
    version = document.get("schema_version")
    if version != GRAPH_SCHEMA_VERSION:
        raise RecipeError(f"unsupported graph schema_version: {version!r}")
    kind = document.get("kind")
    if kind not in {KERNEL_FRAGMENT, COMPLETE_MODEL_GRAPH}:
        raise RecipeError(
            f"graph recipe kind must be {KERNEL_FRAGMENT!r} or "
            f"{COMPLETE_MODEL_GRAPH!r}, got {kind!r}"
        )
    graph = document.get("graph")
    if not isinstance(graph, dict):
        raise RecipeError("graph recipe requires a 'graph' object")
    for key in ("inputs", "nodes", "outputs"):
        if key not in graph:
            raise RecipeError(f"graph recipe is missing graph.{key}")

    defined: set[str] = set()
    for entry in graph["inputs"]:
        input_name = entry.get("name")
        if not input_name:
            raise RecipeError("every graph input requires a name")
        if input_name in defined:
            raise RecipeError(f"duplicate graph input {input_name!r}")
        defined.add(input_name)
    for entry in graph.get("state", []):
        state_name = entry.get("name")
        if not state_name:
            raise RecipeError("every state contract requires a name")
        defined.add(state_name)

    for node in graph["nodes"]:
        node_id = node.get("id")
        op = node.get("op")
        if not node_id:
            raise RecipeError("every graph node requires an id")
        if op not in GRAPH_OPERATIONS:
            legal = ", ".join(sorted(GRAPH_OPERATIONS))
            raise RecipeError(f"node {node_id!r}: unknown operation {op!r}; legal: {legal}")
        for operand in node.get("inputs", []):
            if operand not in defined:
                raise RecipeError(
                    f"node {node_id!r}: dangling edge — input {operand!r} is never "
                    "produced by a graph input, state contract, or earlier node"
                )
        for out in node.get("outputs", []):
            if out in defined:
                raise RecipeError(f"node {node_id!r}: duplicate definition of {out!r}")
            defined.add(out)

    for out in graph["outputs"]:
        if out not in defined:
            raise RecipeError(f"graph output {out!r} is never produced")


def load_graph_recipe_document(document: dict) -> GraphRecipe:
    """Validate and load one versioned graph recipe document (schema_version 2).

    The document's ``graph`` section is normalized into typed IR by
    :func:`urm.compiler.normalize.graph.normalize_graph_document`; this loader
    enforces the closed operation vocabulary and structural legality before
    that. Unknown operations, dangling edges, and duplicate definitions are
    rejected here.
    """
    _validate_graph_document(document)
    return GraphRecipe(
        name=document["name"],
        kind=document["kind"],
        component_scope=document.get("component_scope", ""),
        architecture_ids=tuple(document.get("architecture_ids", ())),
        required_external_stages=tuple(document.get("required_external_stages", ())),
        document=document,
    )


def load_graph_recipe_file(path: str | Path) -> GraphRecipe:
    """Load a versioned graph recipe from a JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise RecipeError(f"invalid JSON in {path}: {error}") from error
    return load_graph_recipe_document(document)


__all__ = [
    "COMPLETE_MODEL_GRAPH",
    "GRAPH_OPERATIONS",
    "GRAPH_SCHEMA_VERSION",
    "GraphRecipe",
    "KERNEL_FRAGMENT",
    "KERNEL_SCHEMA_VERSION",
    "ArchitectureLayer",
    "ArchitectureRecipe",
    "RecipeError",
    "load_architecture_recipe_document",
    "load_architecture_recipe_file",
    "load_graph_recipe_document",
    "load_graph_recipe_file",
]
