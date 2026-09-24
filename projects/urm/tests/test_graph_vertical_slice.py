"""Vertical-slice gates for the graph compile path (refactor-list.md step 1).

Proves, for the MHA kernel fragment:

- JSON authority: a graph document is the source of truth; toggling ``causal``
  in the loaded JSON changes the normalized IR and the observable output.
- Semantic selection: the typed node compiles to an attention anchor by
  equation contract, never by recipe name; a forced incompatible anchor (Polar
  for plain softmax MHA) is rejected before a plan is emitted.
- Reference parity: the reference-tier graph executor agrees with the
  independent float64 NumPy oracle.
- Plan authority: execution runs the serialized plan steps; a plan with a
  missing dispatch step fails to bind.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from urm.compiler.normalize.graph import NormalizeError, normalize_graph_document
from urm.compiler.pipeline import compile_graph
from urm.frontend.recipes import (
    RecipeError,
    load_graph_recipe_document,
    load_graph_recipe_file,
)
from urm.runtime.bind import PlanBindingError

RECIPE = "recipes/kernels/mha.json"


def _mha_document() -> dict:
    return load_graph_recipe_file(RECIPE).document


def _tensors(seed: int = 0):
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    B, T, H, D = 2, 8, 4, 16
    return (
        torch,
        torch.randn(B, T, H, D),
        torch.randn(B, T, H, D),
        torch.randn(B, T, H, D),
    )


def test_graph_document_normalizes_to_typed_ir():
    program = normalize_graph_document(_mha_document())
    assert program.name == "graph:mha"
    (op,) = program.ops
    assert type(op).__name__ == "WeightedReduce"
    assert op.spec.normalization.value == "softmax"
    assert op.spec.selection.value == "dense"
    assert op.spec.causal is True
    assert program.outputs == ("output",)


def test_json_authority_causal_toggle_changes_ir_and_output():
    torch, q, k, v = _tensors()
    program_on = normalize_graph_document(_mha_document())
    out_on = compile_graph(program_on, target="reference").execute(
        query=q, key=k, value=v, score_bias=None, attention_mask=None
    )["output"]

    doc_off = copy.deepcopy(_mha_document())
    doc_off["graph"]["nodes"][0]["params"]["causal"] = False
    program_off = normalize_graph_document(doc_off)
    # The IR itself differs.
    assert program_on.ops[0].spec.causal != program_off.ops[0].spec.causal
    out_off = compile_graph(program_off, target="reference").execute(
        query=q, key=k, value=v, score_bias=None, attention_mask=None
    )["output"]
    assert not torch.allclose(out_on, out_off), "causal toggle had no effect"


def test_reference_parity_against_float64_numpy_oracle():
    torch, q, k, v = _tensors()
    from urm.backends.reference.numpy.k1_attention import attention

    out = compile_graph(
        normalize_graph_document(_mha_document()), target="reference"
    ).execute(query=q, key=k, value=v, score_bias=None, attention_mask=None)["output"]

    # Oracle consumes [Hq, Tq, K] per batch; run per batch item.
    errs = []
    for b in range(q.shape[0]):
        ref = attention(
            q[b].transpose(0, 1).numpy(),  # [H, T, D]
            k[b].transpose(0, 1).numpy(),
            v[b].transpose(0, 1).numpy(),
            causal=True,
        )
        errs.append(
            np.abs(out[b].transpose(0, 1).numpy().astype(np.float64) - ref).max()
        )
    assert max(errs) < 1e-4, max(errs)


def test_forced_incompatible_anchor_declines_before_plan():
    # A Polar-only anchor must not be selectable for a plain softmax MHA node:
    # the equation contract gate rejects it during selection.
    from urm.compiler.select.anchors import (
        AnchorRegistry,
        TRUSTED_ANCHORS,
        make_selector,
    )
    from urm.compiler.pipeline import UrmCompiler

    polar_only = tuple(
        a for a in TRUSTED_ANCHORS if a.name == "atma_polar_triton_adapter"
    )
    registry = AnchorRegistry()
    registry.register(make_selector(polar_only))
    program = normalize_graph_document(_mha_document())
    with pytest.raises(Exception) as excinfo:
        UrmCompiler(anchors=registry).compile(program)
    assert "no_anchor_available" in str(excinfo.value) or "declined" in str(
        excinfo.value
    )


def test_plan_with_missing_dispatch_step_fails_to_bind():
    torch, q, k, v = _tensors()
    from urm.compiler.placement.plan import PlanStep

    import dataclasses

    plan = compile_graph(normalize_graph_document(_mha_document()), target="reference")
    compilation = plan.compilation
    # Tamper: point the dispatch step at a different op name.
    tampered_steps = tuple(
        PlanStep(step_id=s.step_id, kind=s.kind, anchor=s.anchor, note="other")
        for s in compilation.plan.steps
    )
    tampered_plan = dataclasses.replace(compilation.plan, steps=tampered_steps)
    tampered = dataclasses.replace(compilation, plan=tampered_plan)
    from urm.runtime.bind import BoundGraphPlan

    with pytest.raises(PlanBindingError):
        BoundGraphPlan(tampered).execute(
            query=q, key=k, value=v, score_bias=None, attention_mask=None
        )


def test_loader_rejects_unknown_operation_and_dangling_edge():
    doc = _mha_document()
    bad_op = copy.deepcopy(doc)
    bad_op["graph"]["nodes"][0]["op"] = "attention"
    with pytest.raises(RecipeError):
        load_graph_recipe_document(bad_op)

    bad_edge = copy.deepcopy(doc)
    bad_edge["graph"]["nodes"][0]["inputs"][1] = "never_defined"
    with pytest.raises(RecipeError):
        load_graph_recipe_document(bad_edge)

    # Unknown node id caught at normalization when structure passes the loader.
    weird = copy.deepcopy(doc)
    weird["graph"]["nodes"][0]["outputs"] = ["output"]
    weird["graph"]["outputs"] = ["missing"]
    with pytest.raises((RecipeError, NormalizeError)):
        normalize_graph_document(load_graph_recipe_document(weird).document)
