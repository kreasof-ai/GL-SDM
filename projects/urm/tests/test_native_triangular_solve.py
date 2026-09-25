"""Gate: the native Triton triangular_solve tier runs the real substitution kernels.

The native anchor ``urm_native_triangular_solve_v1`` executes the exact strict-causal
forward substitution ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` — one program per
(batch·head, D-block) walks the token axis in order — and its adjoint backward
substitution for training. This gate pins forward AND cotangent parity against the
Torch reference provider (fp32), the structured decline contract, and the native-tier
binding of both UT clients (arch-013 DeltaFormer, arch-010 PaTH).

CUDA + Triton are required; the tests skip cleanly when either is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.backends.contract import ProviderFamily, ProviderRequest
from urm.backends.torch.k4.triangular_solve import triangular_solve_forward
from urm.backends.triton.k4.triangular_solve import (
    TriangularSolveNativeTritonProvider,
    triangular_solve,
)
from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document
from urm.ir.program import TriangularSolve

B, H, T, D = 2, 3, 33, 24
DEV = "cuda"


def _operands(seed=5, b=B, h=H, t=T, d=D):
    """Realistic operands: strict-lower softmax probabilities, uniform diagonal."""
    torch.manual_seed(seed)
    scores = torch.randn(b, h, t, t, device=DEV)
    strict = torch.triu(torch.ones(t, t, device=DEV, dtype=torch.bool), diagonal=0)
    probs = torch.nan_to_num(
        torch.softmax(scores.masked_fill(strict, float("-inf")), dim=-1), nan=0.0
    )
    beta = torch.rand(b, h, t, device=DEV)
    value = torch.randn(b, h, t, d, device=DEV)
    return probs, beta, value


@pytest.mark.parametrize("shape", [(2, 3, 33, 24), (2, 2, 64, 64), (1, 2, 128, 96)])
def test_native_triangular_solve_forward_matches_reference(shape):
    """The native Triton solve matches the Torch reference forward substitution."""
    probs, beta, value = _operands(seed=5, b=shape[0], h=shape[1], t=shape[2], d=shape[3])
    native = triangular_solve(probs, beta, value)
    reference = triangular_solve_forward(probs, beta, value)
    err = (native - reference).abs().max().item()
    assert err < 1e-4, f"forward parity: max abs err {err}"


@pytest.mark.parametrize("shape", [(2, 3, 33, 24), (1, 2, 128, 96)])
def test_native_triangular_solve_cotangents_match_reference(shape):
    """Operand cotangents (probs/beta/value) pass through the native backward scan."""
    probs, beta, value = _operands(seed=7, b=shape[0], h=shape[1], t=shape[2], d=shape[3])
    cotangent = torch.randn_like(value)

    def run(fn):
        leaves = [t.clone().requires_grad_(True) for t in (probs, beta, value)]
        return torch.autograd.grad(fn(*leaves), leaves, grad_outputs=cotangent)

    grads_native = run(triangular_solve)
    grads_reference = run(triangular_solve_forward)
    for name, native, reference in zip(
        ("probs", "beta", "value"), grads_native, grads_reference
    ):
        err = (native - reference).abs().max().item()
        assert err < 1e-4, f"cotangent d{name}: max abs err {err}"


def test_native_triangular_solve_provider_decline_is_structured():
    """The provider declines honestly: wrong descriptor, non-fp32 accumulation."""
    provider = TriangularSolveNativeTritonProvider()
    op = TriangularSolve(
        name="solve",
        inputs=("probs", "beta", "value"),
        outputs=("u",),
        roles=(("probs", "probs"), ("beta", "beta"), ("value", "value")),
    )
    canonical = ProviderRequest(
        family=ProviderFamily.TRIANGULAR_SOLVE, descriptor=op, mode="training"
    )
    assert provider.decline(canonical) is None
    wrong = ProviderRequest(
        family=ProviderFamily.TRIANGULAR_SOLVE, descriptor=object(), mode="inference"
    )
    assert provider.decline(wrong) is not None
    fp16 = ProviderRequest(
        family=ProviderFamily.TRIANGULAR_SOLVE,
        descriptor=op,
        mode="inference",
        accumulation_dtype="float16",
    )
    assert "float32" in provider.decline(fp16)


def _solve_doc():
    return {
        "schema_version": 2, "name": "trisolve_native_test", "kind": "kernel_fragment",
        "graph": {
            "inputs": [
                {"name": "probs", "dtype": "float32", "shape": ["B", "H", "T", "T"]},
                {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "D"]},
            ],
            "nodes": [{
                "id": "solve", "op": "triangular_solve",
                "inputs": ["probs", "beta", "value"], "outputs": ["u"],
                "params": {"roles": {"probs": "probs", "beta": "beta", "value": "value"}},
            }],
            "outputs": ["u"],
        },
    }


def test_native_triangular_solve_binds_native_anchor():
    """target='native' selects the Triton anchor for the typed triangular_solve node."""
    plan = compile_graph(
        normalize_graph_document(load_graph_recipe_document(_solve_doc()).document),
        target="native", intent=CompilationIntent.TRAINING,
    )
    anchors = [step.anchor for step in plan.compilation.plan.steps]
    assert anchors == ["urm_native_triangular_solve_v1"]
    probs, beta, value = _operands(seed=9)
    out = plan.execute(probs=probs, beta=beta, value=value)["u"]
    reference = triangular_solve_forward(probs, beta, value)
    err = (out - reference).abs().max().item()
    assert err < 1e-4, f"bound-plan parity: max abs err {err}"


# --- UT clients on the native tier (arch-013 DeltaFormer, arch-010 PaTH) ---


def test_deltaformer_native_training_matches_reference_tier():
    """DeltaFormerLayer(target='native', intent='training') binds the native solve
    anchor, produces finite gradients, and matches the reference-tier layer."""
    from architectures.deltaformer import DeltaFormerLayer

    torch.manual_seed(3)
    h, d, t = 2, 8, 8
    q = torch.randn(2, h, t, d, device=DEV)
    k = torch.randn(2, h, t, d, device=DEV)
    v = torch.randn(2, h, t, d, device=DEV)
    beta = torch.rand(2, h, t, device=DEV)

    def run(target):
        layer = DeltaFormerLayer(h, d, target=target, intent="training")
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, beta)]
        out = layer(*leaves)
        out.square().sum().backward()
        return layer, out, [x.grad for x in leaves]

    native_layer, out_n, grads_n = run("native")
    _, out_r, grads_r = run("reference")
    solve_anchors = [s.anchor for s in native_layer._solve.compilation.plan.steps]
    assert solve_anchors == ["urm_native_triangular_solve_v1"]
    err = (out_n - out_r).abs().max().item()
    assert err < 1e-4, f"deltaformer native vs reference tier: max abs err {err}"
    for name, g_n, g_r in zip(("q", "k", "v", "beta"), grads_n, grads_r):
        assert torch.isfinite(g_n).all(), f"deltaformer d{name} not finite"
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"deltaformer cotangent d{name}: max abs err {err}"


def test_path_attention_native_training_matches_reference_tier():
    """PaTHAttentionLayer(target='native', intent='training') binds the native solve
    anchor, produces finite gradients, and matches the reference-tier layer."""
    from architectures.path_attention import PaTHAttentionLayer

    torch.manual_seed(5)
    h, d, t, hq = 2, 8, 8, 4
    scale = d ** -0.5
    q = torch.randn(1, t, hq, d, device=DEV)
    k = torch.randn(1, t, h, d, device=DEV)
    v = torch.randn(1, t, h, d, device=DEV)
    w = torch.randn(1, t, h, d, device=DEV) * 0.3
    beta = torch.rand(1, t, h, device=DEV) * 0.5
    g = torch.nn.functional.logsigmoid(torch.randn(1, t, hq, device=DEV))

    def run(target):
        layer = PaTHAttentionLayer(hq, d, target=target, intent="training")
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, w, beta, g)]
        out = layer(*leaves, scale)
        out.square().sum().backward()
        return layer, out, [x.grad for x in leaves]

    native_layer, out_n, grads_n = run("native")
    _, out_r, grads_r = run("reference")
    solve_anchors = [s.anchor for s in native_layer._solve.compilation.plan.steps]
    assert solve_anchors == ["urm_native_triangular_solve_v1"]
    err = (out_n - out_r).abs().max().item()
    assert err < 1e-4, f"path_attention native vs reference tier: max abs err {err}"
    for name, g_n, g_r in zip(("q", "k", "v", "w", "beta", "g"), grads_n, grads_r):
        if g_n is None and g_r is None:
            continue
        assert torch.isfinite(g_n).all(), f"path_attention d{name} not finite"
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"path_attention cotangent d{name}: max abs err {err}"
