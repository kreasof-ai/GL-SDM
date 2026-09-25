"""Gates for complete-model-graph (architecture) recipe documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urm.frontend.recipes import (
    RecipeError,
    load_architecture_recipe_document,
    load_architecture_recipe_file,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHS = ROOT / "recipes" / "architectures"
SCHEMA = ROOT / "recipes" / "schema" / "architecture-recipe.schema.json"


def test_architecture_documents_validate_against_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    docs = sorted(ARCHS.glob("*.json"))
    assert docs, "expected at least one architecture recipe"
    for path in docs:
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), schema)


def test_pattention_is_honestly_scoped_fragment() -> None:
    # PAttention's parameter-token module is not yet built; it records fragment scope.
    patention = load_architecture_recipe_file(ARCHS / "pattention.json")
    assert patention.coverage_level == "kernel_fragment"
    assert patention.coverage_validated is False


def test_samba_records_attention_branch_scope() -> None:
    # Batch 1 landed the attention branch (arch-053 composition-now): validated
    # layer parity, with the Mamba-gated configurations explicitly out of scope.
    samba = load_architecture_recipe_file(ARCHS / "samba.json")
    assert samba.coverage_level == "generic_model_integration"
    assert samba.coverage_validated is True


def test_loader_rejects_empty_layer_graph() -> None:
    doc = json.loads((ARCHS / "samba.json").read_text(encoding="utf-8"))
    doc["layers"] = []
    with pytest.raises(RecipeError, match="layer graph"):
        load_architecture_recipe_document(doc)


def test_loader_rejects_duplicate_layer_ids() -> None:
    doc = json.loads((ARCHS / "samba.json").read_text(encoding="utf-8"))
    doc["layers"] = [doc["layers"][0], doc["layers"][0]]
    with pytest.raises(RecipeError, match="duplicate layer id"):
        load_architecture_recipe_document(doc)


def test_loader_rejects_wrong_kind() -> None:
    doc = json.loads((ARCHS / "samba.json").read_text(encoding="utf-8"))
    doc["kind"] = "kernel_fragment"
    with pytest.raises(RecipeError, match="kind"):
        load_architecture_recipe_document(doc)
