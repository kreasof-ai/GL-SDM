"""Gates for the versioned JSON recipe documents and their loaders.

- **schema_version 2 (graph)**: typed-operation graph documents — the
  authoritative form. These are tested for JSON authority (a graph field
  changes the normalized IR) in ``tests/test_graph_vertical_slice.py``; here we
  test the loader's structural validation.
- **architecture documents**: complete-model graphs under
  ``recipes/architectures/`` validated by the architecture loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urm.frontend.recipes import (
    RecipeError,
    load_architecture_recipe_document,
    load_graph_recipe_document,
)

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "recipes" / "kernels"
ARCHITECTURES = ROOT / "recipes" / "architectures"
SCHEMA_GRAPH = ROOT / "recipes" / "schema" / "graph-recipe.schema.json"


def _documents() -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(KERNELS.glob("*.json"))
    }


def test_every_on_disk_kernel_document_is_a_v2_graph_that_loads() -> None:
    """The JSON files are the catalog: every document must load."""
    docs = _documents()
    assert docs, "kernel catalog must not be empty"
    for name, doc in docs.items():
        assert doc.get("schema_version") == 2, f"{name} is not a v2 graph document"
        graph = load_graph_recipe_document(doc)
        assert graph.name == doc["name"]


def test_every_graph_document_validates_against_the_declared_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    graph = json.loads(SCHEMA_GRAPH.read_text(encoding="utf-8"))
    for name, doc in _documents().items():
        jsonschema.validate(doc, graph), name


def test_every_architecture_document_loads() -> None:
    for path in sorted(ARCHITECTURES.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        recipe = load_architecture_recipe_document(doc)
        assert recipe.name == doc["name"]
        assert recipe.layers, f"{recipe.name} has no layers"


def test_graph_loader_rejects_wrong_schema_version() -> None:
    doc = next(iter(_documents().values()))
    doc = dict(doc, schema_version=999)
    with pytest.raises(RecipeError, match="schema_version"):
        load_graph_recipe_document(doc)


def test_graph_loader_rejects_unknown_operation() -> None:
    doc = next(iter(_documents().values()))
    doc = json.loads(json.dumps(doc))  # deep copy
    doc["graph"]["nodes"][0]["op"] = "attention"
    with pytest.raises(RecipeError, match="unknown operation"):
        load_graph_recipe_document(doc)
