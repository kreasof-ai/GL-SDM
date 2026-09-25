"""Gate 2: the native K2 tier runs the real Triton scan and declines honestly.

The two native K2 anchors (``urm_native_diagonal_recurrence_v1`` /
``urm_native_matrix_state_recurrence_v1``) execute the fused Triton matrix-state
recurrence — not the reference recurrence. They implement only the canonical K2 law
(delta/additive update, diagonal gate scopes none/scalar/head/channel, before/after
read) faithfully; the features whose native kernel semantics diverge from the pinned
law — the normalized denominator, the elementwise gate scope, and the A8 generalized
transitions (erase/write/predict/low_rank/multi-delta) — are distinct equation
contracts the native anchors decline at *selection* time (via ``semantic_contracts``),
so the public path stays honest rather than executing the wrong equation.

CUDA + Triton are required; the tests skip cleanly when either is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.backends.contract import ProviderFamily, ProviderRequest
from urm.backends.triton.k2 import K2NativeMatrixProvider
from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document
from urm.ir.program import K2GateScope, K2ScaleRule, LinearDeltaSpec

B, H, T, K, V = 2, 2, 24, 8, 8
DEV = "cuda"


def _k2_doc(_extra_roles=None, _extra_inputs=None, **spec_over):
    roles = {"query": "query", "key": "key", "value": "value", "beta": "beta",
             "log_decay": "log_decay", "initial_state": "initial_state"}
    if _extra_roles:
        roles.update(_extra_roles)
    params = {
        "delta": True, "gate_scope": "head", "read_timing": "after_update",
        "scale_rule": "one", "roles": roles,
    }
    params.update(spec_over)
    inputs = [
        {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
        {"name": "key", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
        {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
        {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
        {"name": "log_decay", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
        {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
    ]
    if _extra_inputs:
        inputs = inputs + _extra_inputs
    node_inputs = ["initial_state", "key", "query", "value", "beta", "log_decay"] + (
        list(_extra_roles.values()) if _extra_roles else []
    )
    return {
        "schema_version": 2, "name": "k2_native_test", "kind": "kernel_fragment",
        "graph": {
            "inputs": inputs,
            "nodes": [{
                "id": "mix", "op": "linear_delta_state",
                "inputs": node_inputs,
                "outputs": ["output", "final_state"],
                "params": params,
            }],
            "outputs": ["output", "final_state"],
        },
    }


def _operands(gate: str):
    torch.manual_seed(5)
    q = torch.randn(B, H, T, K, device=DEV)
    k = torch.randn(B, H, T, K, device=DEV)
    v = torch.randn(B, H, T, V, device=DEV)
    beta = torch.rand(B, H, T, device=DEV)
    if gate in ("scalar", "head"):
        g = torch.nn.functional.logsigmoid(torch.randn(B, H, T, device=DEV))
    elif gate == "channel":
        g = torch.nn.functional.logsigmoid(torch.randn(B, H, T, K, device=DEV))
    else:  # none
        g = torch.zeros(B, H, T, K, device=DEV)
    m0 = torch.zeros(B, H, K, V, device=DEV)
    return dict(query=q, key=k, value=v, beta=beta, log_decay=g, initial_state=m0)


@pytest.mark.parametrize("delta", [True, False])
@pytest.mark.parametrize("gate", ["none", "scalar", "head", "channel"])
@pytest.mark.parametrize("timing", ["after_update", "before_update"])
def test_native_k2_canonical_matches_reference(delta, gate, timing):
    """The native Triton scan matches the reference tier across the canonical envelope."""
    doc = _k2_doc(delta=delta, gate_scope=gate, read_timing=timing)
    ops = _operands(gate)
    native = compile_graph(
        normalize_graph_document(load_graph_recipe_document(doc).document),
        target="native", intent=CompilationIntent.INFERENCE,
    )
    reference = compile_graph(
        normalize_graph_document(load_graph_recipe_document(doc).document),
        target="reference", intent=CompilationIntent.INFERENCE,
    )
    out_n = native.execute(**ops)
    out_r = reference.execute(**ops)
    assert (out_n["output"] - out_r["output"]).abs().max().item() < 5e-3
    assert (out_n["final_state"] - out_r["final_state"]).abs().max().item() < 5e-3


@pytest.mark.parametrize("kwargs", [
    {"delta": False, "normalized": True},
    {"delta": False, "gate_scope": "elementwise"},
    # A8 features are role-bound: erase_gate becomes a descriptor flag via the role.
    {"delta": False, "_extra_roles": {"erase_gate": "eg"},
     "_extra_inputs": [{"name": "eg", "dtype": "float32", "shape": ["B", "H", "T", "K"]}]},
    {"delta": True, "num_deltas": 2},
])
def test_native_k2_declines_noncanonical(kwargs):
    """Descriptors outside the canonical envelope are declined by the native anchors."""
    doc = _k2_doc(**kwargs)
    with pytest.raises(Exception):
        compile_graph(
            normalize_graph_document(load_graph_recipe_document(doc).document),
            target="native", intent=CompilationIntent.INFERENCE,
        )


def test_native_k2_normalized_falls_back_to_reference_tier():
    """A normalized descriptor declines native but still executes on the reference tier."""
    doc = _k2_doc(delta=False, normalized=True)
    plan = compile_graph(
        normalize_graph_document(load_graph_recipe_document(doc).document),
        target="reference", intent=CompilationIntent.INFERENCE,
    )
    ops = _operands("head")
    out = plan.execute(**{k: v.cpu() for k, v in ops.items()})
    assert out["output"].shape == (B, H, T, V)


@pytest.mark.parametrize("delta", [True, False])
@pytest.mark.parametrize("gate", ["head", "channel"])
def test_native_k2_cotangents_match_reference(delta, gate):
    """Operand and final-state cotangents pass through the native reverse scan."""
    from urm.backends.torch.k2 import linear_delta_state as torch_lds
    from urm.backends.triton.k2 import linear_delta_state as native_lds
    from urm.ir.program import K2ReadTiming

    torch.manual_seed(5)
    spec = LinearDeltaSpec(
        delta=delta, gate_scope=K2GateScope(gate),
        read_timing=K2ReadTiming.AFTER_UPDATE, scale_rule=K2ScaleRule.ONE,
    )
    m0 = torch.zeros(B, H, K, V, device=DEV)
    base = dict(
        q=torch.randn(B, H, T, K, device=DEV), k=torch.randn(B, H, T, K, device=DEV),
        v=torch.randn(B, H, T, V, device=DEV), beta=torch.rand(B, H, T, device=DEV),
        g=(torch.nn.functional.logsigmoid(torch.randn(B, H, T, device=DEV)) if gate == "head"
           else torch.nn.functional.logsigmoid(torch.randn(B, H, T, K, device=DEV))),
    )
    names = ("q", "k", "v", "beta", "g")

    def run(fn):
        p = {n: base[n].clone().requires_grad_(True) for n in names}
        out, final = fn(m0, p["k"], p["q"], p["v"], p["beta"], p["g"], spec=spec)
        (out.float().sum() + final.float().sum()).backward()
        return {n: p[n].grad for n in names}

    gn = run(native_lds)
    gr = run(torch_lds)
    for n in names:
        if gn[n] is None and gr[n] is None:
            continue
        assert (gn[n] - gr[n]).abs().max().item() < 5e-2, f"cotangent d{n}"


def test_native_k2_provider_decline_is_structured():
    """The provider's structured decline names the divergent feature, never silent."""
    provider = K2NativeMatrixProvider()
    base = dict(delta=False, gate_scope=K2GateScope.HEAD, scale_rule=K2ScaleRule.ONE)

    def req(**over):
        spec = LinearDeltaSpec(**{**base, **over})
        return ProviderRequest(family=ProviderFamily.K2, descriptor=spec, mode="inference")

    assert provider.decline(req()) is None  # canonical: no decline
    assert "normalized" in provider.decline(req(normalized=True))
    assert "elementwise" in provider.decline(req(gate_scope=K2GateScope.ELEMENTWISE))
    assert "transition" in provider.decline(req(erase_gate=True))
    assert "transition" in provider.decline(req(num_deltas=2))
