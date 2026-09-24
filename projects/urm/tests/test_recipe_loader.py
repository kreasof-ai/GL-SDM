"""Gates for the versioned JSON recipe documents and their loaders.

Two schema generations coexist during the cutover:

- **schema_version 2 (graph)**: typed-operation graph documents — the
  authoritative form. These are tested for JSON authority (a graph field
  changes the normalized IR) in ``tests/test_graph_vertical_slice.py``; here we
  test the loader's structural validation.
- **schema_version 1 (spec-dump)**: the legacy per-recipe ``UnifiedMixerSpec``
  dump. These remain loadable until their families are migrated to graph
  documents, at which point the v1 catalog and its roundtrip test are deleted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urm.frontend.recipes import (
    RecipeError,
    load_graph_recipe_document,
    load_kernel_recipe_dir,
    load_kernel_recipe_document,
    load_kernel_recipe_file,
)

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "recipes" / "kernels"
SCHEMA_V1 = ROOT / "recipes" / "schema" / "kernel-recipe.schema.json"
SCHEMA_GRAPH = ROOT / "recipes" / "schema" / "graph-recipe.schema.json"


def _documents() -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(KERNELS.glob("*.json"))
    }


def test_every_on_disk_document_loads_through_its_schema_loader() -> None:
    """The JSON files are the catalog: every document must load."""
    loaded_v1 = load_kernel_recipe_dir(KERNELS)
    for name, doc in _documents().items():
        if doc.get("schema_version") == 2:
            graph = load_graph_recipe_document(doc)
            assert graph.name == doc["name"]
        else:
            assert name in loaded_v1
            assert loaded_v1[doc["name"]].spec.name == doc["name"]


def test_every_document_validates_against_its_declared_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    v1 = json.loads(SCHEMA_V1.read_text(encoding="utf-8"))
    graph = json.loads(SCHEMA_GRAPH.read_text(encoding="utf-8"))
    for name, doc in _documents().items():
        schema = graph if doc.get("schema_version") == 2 else v1
        jsonschema.validate(doc, schema), name


def test_v1_spec_dumps_roundtrip_the_catalog_specs() -> None:
    """v1 loads are deterministic: two loads of the dir agree spec-for-spec."""
    first = load_kernel_recipe_dir(KERNELS)
    second = load_kernel_recipe_dir(KERNELS)
    assert set(first) == set(second)
    for name in first:
        assert first[name].spec.to_dict() == second[name].spec.to_dict(), name


def test_graph_documents_are_not_loadable_as_v1() -> None:
    docs = _documents()
    for name, doc in docs.items():
        if doc.get("schema_version") == 2:
            with pytest.raises(RecipeError):
                load_kernel_recipe_document(doc)


def test_graph_loader_rejects_wrong_schema_version() -> None:
    doc = next(d for d in _documents().values() if d.get("schema_version") == 2)
    doc = dict(doc, schema_version=999)
    with pytest.raises(RecipeError, match="schema_version"):
        load_graph_recipe_document(doc)


def test_graph_loader_rejects_unknown_operation() -> None:
    doc = next(d for d in _documents().values() if d.get("schema_version") == 2)
    doc = json.loads(json.dumps(doc))  # deep copy
    doc["graph"]["nodes"][0]["op"] = "attention"
    with pytest.raises(RecipeError, match="unknown operation"):
        load_graph_recipe_document(doc)


def test_v1_loader_rejects_wrong_schema_version() -> None:
    doc = next(d for d in _documents().values() if d.get("schema_version") != 2)
    doc = dict(doc, schema_version=999)
    with pytest.raises(RecipeError, match="schema_version"):
        load_kernel_recipe_document(doc)
