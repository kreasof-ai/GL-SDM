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


def test_batch1_modules_record_validated_layer_parity() -> None:
    # Batch 1 landed the composition-now modules: both record validated
    # generic_model_integration with their residual blockers noted in the recipe.
    samba = load_architecture_recipe_file(ARCHS / "samba.json")
    patention = load_architecture_recipe_file(ARCHS / "pattention.json")
    assert samba.coverage_level == "generic_model_integration"
    assert samba.coverage_validated is True
    assert patention.coverage_level == "generic_model_integration"
    assert patention.coverage_validated is True


def test_samba_records_attention_branch_scope() -> None:
    # arch-053 claims the attention branch only; Mamba-gated configs are out.
    samba = load_architecture_recipe_file(ARCHS / "samba.json")
    assert samba.coverage_level == "generic_model_integration"
    assert samba.coverage_validated is True
    (layer,) = samba.layers
    assert layer.external_component == "architectures.samba_attention.SambaAttentionLayer"


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
