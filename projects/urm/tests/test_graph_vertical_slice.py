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


def test_k3_route_update_read_graph_compiles_and_executes():
    """The K3 recipe compiles as a route→update→read graph through the common path."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("native K3 requires CUDA")
    recipe = load_graph_recipe_file("recipes/kernels/sparse_delta_memory.json")
    program = normalize_graph_document(recipe.document)
    # Two route-generation nodes plus one state-mixer node, no special SDM plan.
    assert [type(op).__name__ for op in program.ops] == [
        "SparseRouteGeneration",
        "SparseRouteGeneration",
        "SparseStateMixerAccess",
    ]
    plan = compile_graph(program, target="native")
    anchors = [step.anchor for step in plan.compilation.plan.steps]
    assert anchors == [
        "urm_native_sparse_route_selection_v0",
        "urm_native_sparse_route_selection_v0",
        "urm_native_sparse_state_mixer_v0",
    ]

    # Match the recipe's declared equation: slots_per_partition=4096 (factor 64),
    # value_dim=128, route width 4.
    P, T, S, D, F, W = 1, 4, 4096, 128, 64, 4
    dev = "cuda"
    ops = {
        "read_scores": torch.randn(P, T, 2 * F, dtype=torch.bfloat16, device=dev),
        "write_scores": torch.randn(P, T, 2 * F, dtype=torch.bfloat16, device=dev),
        "values": torch.randn(P, T, D, dtype=torch.bfloat16, device=dev),
        "beta": torch.rand(P, T, 1, dtype=torch.bfloat16, device=dev),
        "log_decay": -torch.rand(P, T, 1, dtype=torch.bfloat16, device=dev),
        "memory": torch.zeros(P, S, D, dtype=torch.bfloat16, device=dev),
    }
    out = plan.execute(**ops)["output"]
    assert tuple(out.shape) == (P, T, D)


def test_k3_state_mixer_matches_independent_reference():
    """The native K3 state mixer matches the independent differentiable reference."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("native K3 requires CUDA")
    import dataclasses

    from urm.backends.reference.torch.k3 import torch_sparse_state_mixer
    from urm.backends.triton.k3.state_launcher import (
        CertifiedSparseStateRoutes,
        SparseState,
        TritonSparseStateMixerBackend,
    )
    from urm.ir.program import SparseReadTiming

    recipe = load_graph_recipe_file("recipes/kernels/sparse_delta_memory.json")
    program = normalize_graph_document(recipe.document)
    P, T, S, D, W = 1, 4, 4096, 16, 4
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    read_idx = torch.stack(
        [torch.randperm(S, generator=gen, device=dev)[:W].sort().values for _ in range(P * T)]
    ).reshape(P, T, W).to(torch.int64)
    write_idx = torch.stack(
        [torch.randperm(S, generator=gen, device=dev)[:W].sort().values for _ in range(P * T)]
    ).reshape(P, T, W).to(torch.int64)
    rw = torch.softmax(torch.randn(P, T, W, generator=gen, device=dev), -1).to(torch.bfloat16)
    ww = torch.softmax(torch.randn(P, T, W, generator=gen, device=dev), -1).to(torch.bfloat16)
    values = torch.randn(P, T, D, generator=gen, device=dev, dtype=torch.bfloat16)
    beta = torch.rand(P, T, 1, generator=gen, device=dev, dtype=torch.bfloat16)
    log_decay = -torch.rand(P, T, 1, generator=gen, device=dev, dtype=torch.bfloat16)
    memory0 = torch.zeros(P, S, D, dtype=torch.bfloat16, device=dev)

    spec = dataclasses.replace(
        program.ops[2].spec, parallel=P, sequence=T, value_dim=D, reads=W, writes=W
    )
    routes = CertifiedSparseStateRoutes.certify(
        spec, read_idx, rw, write_indices=write_idx, write_weights=ww
    )
    backend = TritonSparseStateMixerBackend(spec)
    prepared = backend.prepare(routes, values=values, beta=beta, log_decay=log_decay)
    native_out, native_state = backend.execute(
        SparseState(memory=memory0.clone(), sequence_length=0), prepared
    )
    ref_out, ref_state = torch_sparse_state_mixer(
        memory0.clone(),
        read_idx,
        rw,
        write_indices=write_idx,
        write_weights=ww,
        values=values,
        beta=beta,
        log_decay=log_decay,
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    assert (native_out.float() - ref_out.float()).abs().max().item() < 2e-2
    assert (native_state.memory.float() - ref_state.float()).abs().max().item() < 2e-2


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
