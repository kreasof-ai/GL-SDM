"""Consumer-side access to the declarative recipe catalog.

The catalog itself is JSON only (``recipes/kernels/*.json``): schema-v1 kernel
fragments load as :class:`urm.frontend.recipes.MixerRecipe` and compile through
the transitional family path (:func:`urm.compiler.mixer.compile_mixer`); schema-v2
typed graph documents load as :class:`urm.frontend.recipes.GraphRecipe` and
compile through the public graph path
(:func:`urm.compiler.pipeline.compile_graph`). This module is the single place
benchmarks and tests resolve recipe names; core URM ships only the loaders.
"""

from __future__ import annotations

import json
from pathlib import Path

from urm.frontend.recipes import (
    GRAPH_SCHEMA_VERSION,
    GraphRecipe,
    MixerRecipe,
    RecipeError,
    load_graph_recipe_document,
    load_kernel_recipe_document,
)

KERNEL_RECIPES_DIR = Path(__file__).resolve().parents[1] / "recipes" / "kernels"


def kernel_recipe_names() -> tuple[str, ...]:
    """Every recipe name in the catalog (v1 kernel fragments and v2 graphs)."""
    names = []
    for path in sorted(KERNEL_RECIPES_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        names.append(document["name"])
    return tuple(names)


def is_graph_recipe(name: str) -> bool:
    """Whether ``name`` is a schema-v2 typed graph document."""
    document = json.loads(
        (KERNEL_RECIPES_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    return document.get("schema_version") == GRAPH_SCHEMA_VERSION


def load_recipe(name: str) -> MixerRecipe | GraphRecipe:
    """Load the recipe ``name`` from the JSON catalog, dispatching on schema."""
    path = KERNEL_RECIPES_DIR / f"{name.strip().lower().replace('-', '_').replace(' ', '_')}.json"
    if not path.exists():
        legal = ", ".join(kernel_recipe_names())
        raise RecipeError(f"no recipe named {name!r}; available: {legal}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") == GRAPH_SCHEMA_VERSION:
        return load_graph_recipe_document(document)
    return load_kernel_recipe_document(document)


def load_kernel_recipe(name: str) -> MixerRecipe:
    """Load ``name`` as a schema-v1 kernel fragment; v2 graphs decline here."""
    recipe = load_recipe(name)
    if isinstance(recipe, GraphRecipe):
        raise RecipeError(
            f"recipe {name!r} is a typed graph document (schema_version 2); "
            "compile it through the graph path "
            "(load_graph_recipe_file -> normalize_graph_document -> compile_graph)"
        )
    return recipe


def compile_named_recipe(name: str, *, target: str, intent: str = "inference"):
    """Compile a schema-v2 recipe through the public graph path.

    Returns a :class:`urm.runtime.bind.BoundGraphPlan`; ``execute(**operands)``
    returns a dict of named outputs. ``target`` selects the implementation tier
    ("reference", "library", "native"); semantic legality is enforced by the
    compiler independently of the target.
    """
    from urm.compiler.normalize.graph import normalize_graph_document
    from urm.compiler.pipeline import CompilationIntent, compile_graph
    from urm.frontend.recipes import load_graph_recipe_file

    recipe = load_graph_recipe_file(KERNEL_RECIPES_DIR / f"{name}.json")
    program = normalize_graph_document(recipe.document)
    return compile_graph(
        program, target=target, intent=CompilationIntent(intent)
    )
