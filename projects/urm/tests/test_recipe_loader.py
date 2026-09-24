"""Gates for the versioned JSON kernel-recipe documents and their loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urm.frontend.recipes import (
    RecipeError,
    load_kernel_recipe_dir,
    load_kernel_recipe_document,
    load_kernel_recipe_file,
)
from urm.frontend.recipes import MIXER_RECIPE_NAMES, named_mixer_recipe

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "recipes" / "kernels"
SCHEMA = ROOT / "recipes" / "schema" / "kernel-recipe.schema.json"


def test_every_catalog_recipe_has_a_json_document() -> None:
    on_disk = {p.stem for p in KERNELS.glob("*.json")}
    assert on_disk == set(MIXER_RECIPE_NAMES)


def test_json_recipes_roundtrip_the_catalog_specs() -> None:
    loaded = load_kernel_recipe_dir(KERNELS)
    catalog_specs = {
        tuple(sorted(named_mixer_recipe(name).spec.to_dict().items()))
        for name in MIXER_RECIPE_NAMES
    }
    loaded_specs = {
        tuple(sorted(recipe.spec.to_dict().items())) for recipe in loaded.values()
    }
    assert loaded_specs == catalog_specs


def test_recipes_validate_against_the_versioned_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    for path in sorted(KERNELS.glob("*.json")):
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), schema)


def test_loader_rejects_wrong_schema_version() -> None:
    doc = json.loads((KERNELS / "mha.json").read_text(encoding="utf-8"))
    doc["schema_version"] = 999
    with pytest.raises(RecipeError, match="schema_version"):
        load_kernel_recipe_document(doc)


def test_loader_rejects_non_fragment_kind() -> None:
    doc = json.loads((KERNELS / "mha.json").read_text(encoding="utf-8"))
    doc["kind"] = "complete_model_graph"
    with pytest.raises(RecipeError, match="kind"):
        load_kernel_recipe_document(doc)


def test_loader_rejects_unknown_spec_fields() -> None:
    doc = json.loads((KERNELS / "mha.json").read_text(encoding="utf-8"))
    doc["spec"]["hidden_python_callback"] = "evil"
    with pytest.raises(RecipeError, match="unknown mixer spec fields"):
        load_kernel_recipe_document(doc)


def test_loader_rejects_missing_file() -> None:
    with pytest.raises((RecipeError, FileNotFoundError, OSError)):
        load_kernel_recipe_file(KERNELS / "does_not_exist.json")
