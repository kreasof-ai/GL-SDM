"""Consumer-side access to the declarative recipe catalog.

The catalog itself is JSON only: ``recipes/kernels/*.json`` holds schema-v2
typed graph documents (compiled through the public graph path,
:func:`urm.compiler.pipeline.compile_graph`) and ``recipes/architectures/*.json``
holds complete-model graphs. This module is the single place benchmarks and
tests resolve recipe names and documents; core URM ships only the loaders.
"""

from __future__ import annotations

import json
from pathlib import Path

KERNEL_RECIPES_DIR = Path(__file__).resolve().parents[1] / "recipes" / "kernels"


def kernel_recipe_names() -> tuple[str, ...]:
    """Every recipe name in the kernel catalog."""
    names = []
    for path in sorted(KERNEL_RECIPES_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        names.append(document["name"])
    return tuple(names)


def recipe_document(name: str) -> dict:
    """Return the raw parsed JSON document for ``name`` (raises KeyError)."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    path = KERNEL_RECIPES_DIR / f"{normalized}.json"
    if not path.exists():
        raise KeyError(f"no recipe named {name!r}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    return compile_graph(program, target=target, intent=CompilationIntent(intent))
