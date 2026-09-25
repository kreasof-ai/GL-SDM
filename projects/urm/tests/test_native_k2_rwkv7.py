"""Native K2 low-rank+decay composition (RWKV-7 DPLR): the fused Triton scan runs the
pinned law on the native tier.

The RWKV-7 DPLR transition COMPOSES a pointwise (channel) decay with an additive
rank-1 transition, per the pinned fla law (fla/ops/rwkv7/fused_recurrent.py @
864a87f6 → fused_recurrent_dplr_delta_rule with gk = w) and the URM Torch reference
low-rank branch::

    lr_read = alpha_tᵀ · M_{t-1}                       (off the PRE-decay state)
    M_t     = exp(w_t) ⊙ M_{t-1} + k_t·v_tᵀ + low_rank_beta_t ⊗ lr_read
    o_t     = scale · q_tᵀ · M_t                       (read after update)

The native kernel takes the ``alpha``/``low_rank_beta`` factors directly (not the
full ``left = I + β αᵀ`` matrix), reads ``alphaᵀ·M`` off the pre-decay state, applies
the decay to the carried state, and adds the rank-1 term UNDECAYED alongside the
``k·vᵀ`` write. These tests pin forward AND cotangent parity against the Torch
reference and the pinned fla law, and verify the public ``RWKV7Layer`` runs natively
with finite, reference-matching gradients.

CUDA + Triton are required; the fla-pin parity tests skip cleanly when the pinned
checkout (``/tmp/urm-comparator-pins/fla``) is not provisioned.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.backends.torch.k2 import linear_delta_state as torch_lds
from urm.backends.triton.k2 import linear_delta_state as native_lds
from urm.backends.triton.k2 import K2NativeMatrixProvider
from urm.backends.contract import ProviderFamily, ProviderRequest
from urm.ir.program import K2GateScope, K2ReadTiming, K2ScaleRule, LinearDeltaSpec

B, H, T, K, V = 2, 2, 24, 8, 8
DEV = "cuda"


def _spec(timing=K2ReadTiming.AFTER_UPDATE, scale=K2ScaleRule.ONE):
    return LinearDeltaSpec(
        delta=False, gate_scope=K2GateScope.CHANNEL, read_timing=timing,
        scale_rule=scale, low_rank=True,
    )


def _operands(seed=13, ab_scale=0.2):
    """Stable operands: the rank-1 factors are scaled so the recurrence is contractive
    (an unscaled lb⊗(alphaᵀ·M) term can amplify the state past fp32 abs tolerances)."""
    torch.manual_seed(seed)
    base = dict(
        q=torch.randn(B, H, T, K, device=DEV),
        k=torch.randn(B, H, T, K, device=DEV),
        v=torch.randn(B, H, T, V, device=DEV),
        beta=torch.rand(B, H, T, device=DEV),
        g=torch.nn.functional.logsigmoid(torch.randn(B, H, T, K, device=DEV)),
        alpha=torch.randn(B, H, T, K, device=DEV) * ab_scale,
        low_rank_beta=torch.randn(B, H, T, K, device=DEV) * ab_scale,
    )
    return base


@pytest.mark.parametrize("timing", [K2ReadTiming.AFTER_UPDATE, K2ReadTiming.BEFORE_UPDATE])
def test_native_rwkv7_composed_matches_torch_reference(timing):
    """Forward + final state: native low_rank+channel-decay matches the Torch reference."""
    ops = _operands()
    spec = _spec(timing=timing)
    m0 = torch.zeros(B, H, K, V, device=DEV)
    out_n, final_n = native_lds(
        m0, ops["k"], ops["q"], ops["v"], ops["beta"], ops["g"],
        spec=spec, alpha=ops["alpha"], low_rank_beta=ops["low_rank_beta"],
    )
    out_r, final_r = torch_lds(
        m0, ops["k"], ops["q"], ops["v"], ops["beta"], ops["g"],
        spec=spec, alpha=ops["alpha"], low_rank_beta=ops["low_rank_beta"],
    )
    assert (out_n - out_r).abs().max().item() < 1e-4, "output parity"
    assert (final_n - final_r).abs().max().item() < 1e-4, "final-state parity"


def test_native_rwkv7_composed_cotangents_match_torch_reference():
    """Operand and final-state cotangents pass through the composed reverse scan."""
    ops = _operands()
    spec = _spec()
    m0 = torch.zeros(B, H, K, V, device=DEV)
    names = ("q", "k", "v", "g", "alpha", "low_rank_beta")

    def run(fn):
        p = {n: ops[n].clone().requires_grad_(True) for n in names}
        beta = ops["beta"].clone()
        out, final = fn(
            m0, p["k"], p["q"], p["v"], beta, p["g"],
            spec=spec, alpha=p["alpha"], low_rank_beta=p["low_rank_beta"],
        )
        (out.float().sum() + final.float().sum()).backward()
        return {n: p[n].grad for n in names}

    gn = run(native_lds)
    gr = run(torch_lds)
    for n in names:
        assert gn[n] is not None and gr[n] is not None, f"cotangent d{n} missing"
        assert (gn[n] - gr[n]).abs().max().item() < 1e-3, f"cotangent d{n}"


def test_native_rwkv7_low_rank_term_is_active():
    """The rank-1 transition must affect the output (not collapse to decay + write)."""
    ops = _operands(seed=9)
    spec = _spec()
    m0 = torch.zeros(B, H, K, V, device=DEV)

    def fwd(alpha):
        out, _ = native_lds(
            m0, ops["k"], ops["q"], ops["v"], ops["beta"], ops["g"],
            spec=spec, alpha=alpha, low_rank_beta=ops["low_rank_beta"],
        )
        return out

    out1 = fwd(ops["alpha"])
    out2 = fwd(torch.zeros_like(ops["alpha"]))  # alpha=0 kills the low-rank term
    assert (out1 - out2).abs().max().item() > 1e-3, "low-rank transition has no effect"


def test_native_rwkv7_matches_pinned_fla_naive():
    """Forward parity against the pinned fla DPLR naive recurrence (no CUDA build needed)."""
    from benchmarks.comparators.fla_k2 import fla_op

    try:
        dplr = fla_op("fla.ops.generalized_delta_rule.dplr.naive.dplr_recurrence")
    except Exception as error:  # noqa: BLE001 - optional pinned dependency
        pytest.skip(f"pinned fla checkout unavailable: {error!r}")
    ops = _operands(seed=7)
    spec = _spec(scale=K2ScaleRule.KEY_DIM_RSQRT)  # the pinned law reads (q·K^-0.5)ᵀ·M
    m0 = torch.zeros(B, H, K, V, device=DEV)
    out_n, final_n = native_lds(
        m0, ops["k"], ops["q"], ops["v"], ops["beta"], ops["g"],
        spec=spec, alpha=ops["alpha"], low_rank_beta=ops["low_rank_beta"],
    )
    # dplr_recurrence takes head-first [b,h,l,d]; alpha reads, beta(=low_rank_beta) writes.
    ref_o, ref_S = dplr(
        ops["q"], ops["k"], ops["v"], ops["alpha"], ops["low_rank_beta"], ops["g"],
        initial_state=None, output_final_state=True,
    )
    assert (out_n - ref_o).abs().max().item() < 1e-4, "pinned fla naive output parity"
    assert (final_n - ref_S).abs().max().item() < 1e-4, "pinned fla naive final-state parity"


def test_native_rwkv7_matches_pinned_fla_fused_recurrent():
    """Forward parity against the pinned fla fused_recurrent_rwkv7 (the CUDA op the
    RWKV7Layer maps to), on identical operands."""
    from benchmarks.comparators.fla_k2 import fla_op

    try:
        fused_rwkv7 = fla_op("fla.ops.rwkv7.fused_recurrent.fused_recurrent_rwkv7")
    except Exception as error:  # noqa: BLE001 - optional pinned dependency
        pytest.skip(f"pinned fla fused op unavailable: {error!r}")
    ops = _operands(seed=7)
    spec = _spec(scale=K2ScaleRule.KEY_DIM_RSQRT)
    m0 = torch.zeros(B, H, K, V, device=DEV)
    out_n, final_n = native_lds(
        m0, ops["k"], ops["q"], ops["v"], ops["beta"], ops["g"],
        spec=spec, alpha=ops["alpha"], low_rank_beta=ops["low_rank_beta"],
    )
    # fused_recurrent_rwkv7 takes [B,T,H,*]: (r, w, k, v, a, b); a=alpha, b=low_rank_beta.
    o_fla, S_fla = fused_rwkv7(
        ops["q"].transpose(1, 2), ops["g"].transpose(1, 2), ops["k"].transpose(1, 2),
        ops["v"].transpose(1, 2), ops["alpha"].transpose(1, 2),
        ops["low_rank_beta"].transpose(1, 2),
        scale=None, initial_state=None, output_final_state=True,
    )
    assert (out_n - o_fla.transpose(1, 2)).abs().max().item() < 1e-4, "fused output parity"
    assert (final_n - S_fla).abs().max().item() < 1e-4, "fused final-state parity"


def test_native_rwkv7_provider_admits_composed_low_rank_decay():
    """The native provider admits low_rank+channel decay (it now runs the composed law);
    it never declines what it can execute, and the descriptor is served natively."""
    provider = K2NativeMatrixProvider()
    spec = _spec()
    request = ProviderRequest(
        family=ProviderFamily.K2, descriptor=spec, mode="inference",
    )
    assert provider.decline(request) is None


def test_rwkv7_layer_native_matches_reference_tier():
    """RWKV7Layer(num_heads, head_k_dim, head_v_dim, target="native", intent="training")
    builds, runs forward+backward on CUDA with finite grads, and matches the
    reference-tier layer on identical operands (weights shared via load_state_dict)."""
    from architectures.rwkv7 import RWKV7Layer

    heads, kd, vd, seq = 4, 64, 64, 8
    torch.manual_seed(0)
    r = torch.randn(2, seq, heads, kd, device=DEV)
    w = -torch.rand(2, seq, heads, kd, device=DEV) * 0.3
    k = torch.randn(2, seq, heads, kd, device=DEV)
    v = torch.randn(2, seq, heads, vd, device=DEV)
    a = torch.randn(2, seq, heads, kd, device=DEV) * 0.2
    b = torch.randn(2, seq, heads, kd, device=DEV) * 0.2

    native = RWKV7Layer(heads, kd, vd, target="native", intent="training")
    reference = RWKV7Layer(heads, kd, vd, target="reference", intent="training")
    reference.load_state_dict(native.state_dict())

    def forward_backward(layer):
        operands = [t.clone().requires_grad_(True) for t in (r, w, k, v, a, b)]
        out = layer(*operands)
        out.square().sum().backward()
        return out, {n: t.grad for n, t in zip(("r", "w", "k", "v", "a", "b"), operands)}

    out_n, grads_n = forward_backward(native)
    out_r, grads_r = forward_backward(reference)
    assert (out_n - out_r).abs().max().item() < 1e-4, "layer output parity"
    for n, g in grads_n.items():
        assert g is not None and torch.isfinite(g).all().item(), f"native d{n} not finite"
        assert g.abs().sum().item() > 0, f"native d{n} is zero"
        assert (g - grads_r[n]).abs().max().item() < 1e-3, f"layer cotangent d{n}"
