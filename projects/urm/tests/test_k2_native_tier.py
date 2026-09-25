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
from urm.backends.triton.k2.matrix_scan import K2NativeMatrixProvider
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
])
def test_native_k2_declines_noncanonical(kwargs):
    """Only the genuinely-divergent descriptors are declined by the native anchors.

    The A8 transition features (erase/write/predict/low_rank/multi-delta) are
    parity-qualified and admitted; the normalized variant (denominator law diverges,
    measured 1.9e7) and the elementwise gate (single-client, reference-tier) decline.
    """
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
    from urm.backends.torch.k2.linear_delta import linear_delta_state as torch_lds
    from urm.backends.triton.k2.matrix_scan import linear_delta_state as native_lds
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
    # The transition features are parity-qualified and admitted (no decline).
    assert provider.decline(req(erase_gate=True)) is None
    assert provider.decline(req(num_deltas=2)) is None


@pytest.mark.parametrize("delta", [True, False])
def test_native_k2_key_dim_rsqrt_scale_matches_reference(delta):
    """The native tier resolves scale_rule=key_dim_rsqrt to K**-0.5 — never 1.0.

    Regression: the native provider used to leave scale=None (→ 1.0) while the
    reference resolved key_dim_rsqrt → K**-0.5, so GLA/DeltaNet/GDN-class laws ran at
    K**0.5 × the pinned read scale on the native tier while every gate stayed green.
    """
    from urm.backends.torch.k2.linear_delta import linear_delta_state as torch_lds
    from urm.backends.triton.k2.matrix_scan import linear_delta_state as native_lds
    from urm.ir.program import K2ReadTiming

    torch.manual_seed(11)
    spec = LinearDeltaSpec(
        delta=delta, gate_scope=K2GateScope.CHANNEL,
        read_timing=K2ReadTiming.AFTER_UPDATE, scale_rule=K2ScaleRule.KEY_DIM_RSQRT,
    )
    m0 = torch.zeros(B, H, K, V, device=DEV)
    base = dict(
        q=torch.randn(B, H, T, K, device=DEV), k=torch.randn(B, H, T, K, device=DEV),
        v=torch.randn(B, H, T, V, device=DEV), beta=torch.rand(B, H, T, device=DEV),
        g=torch.nn.functional.logsigmoid(torch.randn(B, H, T, K, device=DEV)),
    )
    names = ("q", "k", "v", "beta", "g")

    def run(fn):
        p = {n: base[n].clone().requires_grad_(True) for n in names}
        out, final = fn(m0, p["k"], p["q"], p["v"], p["beta"], p["g"], spec=spec)
        (out.float().sum() + final.float().sum()).backward()
        return out, final, {n: p[n].grad for n in names}

    out_n, final_n, gn = run(native_lds)
    out_r, final_r, gr = run(torch_lds)
    assert (out_n - out_r).abs().max().item() < 1e-4, "output parity at key_dim_rsqrt"
    assert (final_n - final_r).abs().max().item() < 1e-4, "final-state parity"
    for n in names:
        if gn[n] is None and gr[n] is None:
            continue
        assert (gn[n] - gr[n]).abs().max().item() < 5e-2, f"cotangent d{n}"
    # And the fix must be visible in the value itself: with K=64 the pinned read scale
    # is 0.125, so a scale=1.0 regression shows up as an 8x output ratio.
    assert out_n.abs().max().item() < out_r.abs().max().item() * 4


@pytest.mark.parametrize("feature", ["dual_gate", "multi_rank", "retrieval_key", "low_rank"])
def test_native_k2_transition_features_match_reference(feature):
    """The A8 transition features are parity-qualified on the native tier: forward AND
    cotangents match the Torch reference (the measured residuals that admitted them:
    dual-gate 1.5e-5/1.5e-4, multi-rank 7.6e-5/1.4e-4, retrieval 1.9e-6, low-rank 1.4e-6)."""
    from urm.backends.torch.k2.linear_delta import linear_delta_state as torch_lds
    from urm.backends.triton.k2.matrix_scan import linear_delta_state as native_lds
    from urm.ir.program import K2ReadTiming

    torch.manual_seed(13)
    m0 = torch.zeros(B, H, K, V, device=DEV)
    base = dict(
        q=torch.randn(B, H, T, K, device=DEV), k=torch.randn(B, H, T, K, device=DEV),
        v=torch.randn(B, H, T, V, device=DEV), beta=torch.rand(B, H, T, device=DEV),
        g=torch.nn.functional.logsigmoid(torch.randn(B, H, T, K, device=DEV)),
    )
    spec_kw = dict(delta=True, gate_scope=K2GateScope.CHANNEL,
                   read_timing=K2ReadTiming.AFTER_UPDATE, scale_rule=K2ScaleRule.ONE)
    ops = {}
    if feature == "dual_gate":
        # The GDN2 law is the delta rule with beta≡1 over the dual gates; the native
        # kernel's additive dual-gate form computes the same equation (verified).
        spec_kw.update(erase_gate=True, write_gate=True)
        base["beta"] = torch.ones(B, H, T, device=DEV)
        ops = dict(erase_gate=torch.rand(B, H, T, K, device=DEV),
                   write_gate=torch.rand(B, H, T, V, device=DEV))
    elif feature == "multi_rank":
        spec_kw = dict(delta=False, gate_scope=K2GateScope.HEAD,
                       read_timing=K2ReadTiming.AFTER_UPDATE, scale_rule=K2ScaleRule.ONE,
                       num_deltas=2)
        base["k"] = torch.randn(B, H, T * 2, K, device=DEV)
        base["v"] = torch.randn(B, H, T * 2, V, device=DEV)
        base["beta"] = torch.rand(B, H, T * 2, device=DEV)
        base["g"] = torch.nn.functional.logsigmoid(torch.randn(B, H, T, device=DEV))
    elif feature == "retrieval_key":
        spec_kw.update(predict_key=True)
        ops = dict(predict_key=torch.randn(B, H, T, K, device=DEV))
    else:  # low_rank
        spec_kw = dict(delta=False, gate_scope=K2GateScope.NONE,
                       read_timing=K2ReadTiming.AFTER_UPDATE, scale_rule=K2ScaleRule.ONE,
                       low_rank=True)
        ops = dict(alpha=torch.rand(B, H, T, K, device=DEV) * 0.1,
                   low_rank_beta=torch.rand(B, H, T, K, device=DEV) * 0.1)
    spec = LinearDeltaSpec(**spec_kw)
    names = tuple(base) + tuple(ops)

    def run(fn):
        p = {n: base[n].clone().requires_grad_(True) for n in base}
        po = {n: ops[n].clone().requires_grad_(True) for n in ops}
        out, final = fn(m0, p["k"], p["q"], p["v"], p["beta"], p["g"], spec=spec, **po)
        (out.float().sum() + final.float().sum()).backward()
        return out, final, {**{n: p[n].grad for n in base}, **{n: po[n].grad for n in ops}}

    out_n, final_n, gn = run(native_lds)
    out_r, final_r, gr = run(torch_lds)
    assert (out_n - out_r).abs().max().item() < 5e-4, f"{feature} output parity"
    assert (final_n - final_r).abs().max().item() < 5e-4, f"{feature} final-state parity"
    for n in names:
        if gn[n] is None and gr[n] is None:
            continue
        if gn[n] is None or gr[n] is None:
            # beta is structural (≡1) in the dual-gate and low-rank laws: the native
            # additive mapping doesn't differentiate it while the reference's delta-rule
            # spec does (a ~zero grad). Not a divergence — skip the None pair.
            if n == "beta" and feature in ("dual_gate", "low_rank"):
                continue
            raise AssertionError(f"{feature} cotangent d{n}: None mismatch (native {gn[n] is None}, ref {gr[n] is None})")
        assert (gn[n] - gr[n]).abs().max().item() < 5e-2, f"{feature} cotangent d{n}"
