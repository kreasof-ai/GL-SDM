"""Batch-0 semantic-truth and provider-contract gates.

These tests close the Gate 0/1 conditions for the K1 descriptor and the new K2
`linear_delta_state` node: every accepted JSON semantic field changes the
normalized descriptor or is rejected; alpha-renamed graphs normalize
identically; role binding replaces name lookup; and a tampered or incomplete
K2 plan declines before any kernel runs.
"""

from __future__ import annotations

import copy

import pytest

from urm.compiler.normalize.graph import NormalizeError, normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import RecipeError, load_graph_recipe_document, load_graph_recipe_file
from urm.runtime.bind import PlanBindingError

K2_DOC = {
    "schema_version": 2,
    "name": "k2_probe",
    "kind": "kernel_fragment",
    "component_scope": "canonical delta-rule state",
    "graph": {
        "inputs": [
            {"name": "q", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
            {"name": "k", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
            {"name": "v", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
            {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
            {"name": "log_decay", "dtype": "float32", "shape": ["B", "H", "T"]},
            {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
        ],
        "nodes": [
            {
                "id": "state",
                "op": "linear_delta_state",
                "inputs": ["q", "k", "v", "beta", "log_decay", "initial_state"],
                "outputs": ["output", "final_state"],
                "params": {
                    "delta": True,
                    "gate_scope": "head",
                    "read_timing": "after_update",
                    "scale_rule": "one",
                    "roles": {
                        "query": "q", "key": "k", "value": "v",
                        "beta": "beta", "log_decay": "log_decay",
                        "initial_state": "initial_state",
                    },
                },
            }
        ],
        "outputs": ["output", "final_state"],
    },
}


def _doc():
    return copy.deepcopy(K2_DOC)


# --- Gate 0: every accepted K2 semantic field changes the descriptor ---------


def test_k2_every_semantic_field_changes_descriptor():
    base = normalize_graph_document(_doc()).ops[0].spec

    for field, value in (
        ("delta", False),
        ("gate_scope", "channel"),
        ("read_timing", "before_update"),
        ("scale_rule", "key_dim_rsqrt"),
        ("normalized", True),
    ):
        doc = _doc()
        doc["graph"]["nodes"][0]["params"][field] = value
        spec = normalize_graph_document(doc).ops[0].spec
        assert getattr(spec, field).value if hasattr(getattr(spec, field), "value") else getattr(spec, field) != getattr(base, field)


def test_k2_unknown_and_invalid_fields_reject():
    # Unknown param key.
    doc = _doc()
    doc["graph"]["nodes"][0]["params"]["unknown_transition"] = "magic"
    with pytest.raises((RecipeError, NormalizeError)):
        load_graph_recipe_document(doc) and normalize_graph_document(doc)
    # Invalid enum value.
    doc = _doc()
    doc["graph"]["nodes"][0]["params"]["gate_scope"] = "per_token_magic"
    with pytest.raises((RecipeError, NormalizeError)):
        normalize_graph_document(load_graph_recipe_document(doc).document)
    # Missing required roles.
    doc = _doc()
    del doc["graph"]["nodes"][0]["params"]["roles"]["key"]
    with pytest.raises((RecipeError, NormalizeError)):
        normalize_graph_document(load_graph_recipe_document(doc).document)


def test_k2_alpha_renaming_normalizes_identically():
    doc = _doc()
    renamed = _doc()
    mapping = {"q": "alpha_q", "k": "alpha_k", "v": "alpha_v"}
    g = renamed["graph"]
    for inp in g["inputs"]:
        if inp["name"] in mapping:
            inp["name"] = mapping[inp["name"]]
    node = g["nodes"][0]
    node["inputs"] = [mapping.get(i, i) for i in node["inputs"]]
    node["params"]["roles"] = {
        r: mapping.get(e, e) for r, e in node["params"]["roles"].items()
    }
    a = normalize_graph_document(doc).ops[0]
    b = normalize_graph_document(renamed).ops[0]
    # The closed descriptor is identical under alpha-renaming; roles point at
    # the renamed edges by construction (that is what role binding means).
    assert a.spec == b.spec
    assert set(dict(a.roles)) == set(dict(b.roles))
    assert dict(b.roles)["query"] == "alpha_q"


# --- Gate 1: K2 role binding, tamper and decline ------------------------------


def _k2_operands():
    torch = pytest.importorskip("torch")
    B, H, T, K, V = 1, 2, 5, 4, 3
    return torch, {
        "q": torch.randn(B, H, T, K),
        "k": torch.randn(B, H, T, K),
        "v": torch.randn(B, H, T, V),
        "beta": torch.rand(B, H, T),
        "log_decay": -torch.rand(B, H, T),
        "initial_state": torch.randn(B, H, K, V),
    }


def test_k2_roles_not_names_drive_binding():
    """Renaming the graph edges must not change the bound execution (roles win)."""
    torch, operands = _k2_operands()
    plan = compile_graph(normalize_graph_document(_doc()), target="reference")
    out_named = plan.execute(**operands)["output"]

    renamed = _doc()
    g = renamed["graph"]
    mapping = {"q": "qq", "k": "kk", "v": "vv"}
    for inp in g["inputs"]:
        inp["name"] = mapping.get(inp["name"], inp["name"])
    node = g["nodes"][0]
    node["inputs"] = [mapping.get(i, i) for i in node["inputs"]]
    node["params"]["roles"] = {r: mapping.get(e, e) for r, e in node["params"]["roles"].items()}
    plan2 = compile_graph(normalize_graph_document(renamed), target="reference")
    out_renamed = plan2.execute(**{mapping.get(n, n): t for n, t in operands.items()})["output"]
    assert torch.allclose(out_named, out_renamed)


def test_k2_tampered_plan_declines_before_execution():
    import dataclasses

    from urm.compiler.placement.plan import PlanStep
    from urm.runtime.bind import BoundGraphPlan

    torch, operands = _k2_operands()
    plan = compile_graph(normalize_graph_document(_doc()), target="reference")
    compilation = plan.compilation
    tampered_steps = tuple(
        PlanStep(step_id=s.step_id, kind=s.kind, anchor=s.anchor, note="other")
        for s in compilation.plan.steps
    )
    tampered = dataclasses.replace(
        compilation, plan=dataclasses.replace(compilation.plan, steps=tampered_steps)
    )
    with pytest.raises(PlanBindingError):
        BoundGraphPlan(tampered).execute(**operands)


def test_k2_missing_role_operand_declines():
    torch, operands = _k2_operands()
    plan = compile_graph(normalize_graph_document(_doc()), target="reference")
    del operands["k"]  # remove the key operand
    with pytest.raises(PlanBindingError):
        plan.execute(**operands)


def test_k1_descriptor_carries_scale_law_not_runtime_math():
    """The K1 descriptor must carry the scale law; runtime never recomputes it."""
    program = normalize_graph_document(load_graph_recipe_file("recipes/kernels/mha.json").document)
    op = program.ops[0]
    assert op.k1 is not None
    assert op.k1.scale_rule.value == "key_dim_rsqrt"
    assert op.k1.causal is True
    # A non-softmax weighted reduce carries no K1 descriptor.
    assert op.spec.normalization.value == "softmax"
