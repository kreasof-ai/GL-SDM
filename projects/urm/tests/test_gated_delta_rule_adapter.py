"""Correctness and schema tests for the FLA gated delta-rule comparator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
fla = pytest.importorskip("fla")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for gated delta-rule adapter tests", allow_module_level=True
    )

import torch.nn.functional as F

from urm.adapters.gated_delta_reference import (
    eager_gated_delta_rule,
    oracle_gated_delta_rule,
)
from urm.adapters.gated_delta_rule import (
    GatedDeltaRuleSpec,
    UrmGatedDeltaRuleAdapter,
    fla_version,
)

PROJECT_ROOT = Path(__file__).parents[1]

# Dtype-specific tolerances (docs/benchmarking.md: tolerances live in tests).
OUTPUT_TOL = {
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
    torch.float16: {"atol": 1.5e-2, "rtol": 2e-2},
}
STATE_TOL = {"atol": 6e-2, "rtol": 2e-2}


def _sample(b=2, t=16, h=4, hv=None, k=32, v=48, dtype=torch.bfloat16, seed=5):
    hv = hv or h
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((b, t, h, k), device="cuda", dtype=dtype, generator=generator)
    # Caller-side L2 normalization of k per the frozen contract.
    k_data = torch.randn(
        (b, t, h, k), device="cuda", dtype=torch.float32, generator=generator
    )
    k_data = F.normalize(k_data, dim=-1).to(dtype)
    v_out = torch.randn((b, t, hv, v), device="cuda", dtype=dtype, generator=generator)
    g = F.logsigmoid(
        torch.rand((b, t, hv), device="cuda", generator=generator) * 0.9
    ).float()
    beta = torch.rand((b, t, hv), device="cuda", generator=generator).float()
    return q, k_data.to(dtype), v_out, g, beta


def _assert_close(actual, expected, **tol):
    torch.testing.assert_close(
        actual.float(), expected.float(), check_device=True, **tol
    )


def test_spec_rejects_bad_shapes_layouts_and_dtypes() -> None:
    q, k, v, g, beta = _sample()
    GatedDeltaRuleSpec.from_tensors(q, k, v, g, beta)

    with pytest.raises(ValueError, match="rank-4"):
        GatedDeltaRuleSpec.from_tensors(q[0], k[0], v[0], g, beta)
    with pytest.raises(ValueError, match="share"):
        GatedDeltaRuleSpec.from_tensors(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v,
            g,
            beta,
        )
    # A FULLY self-consistent [B,H,T,K] permutation passes pure shape
    # validation (every cross-tensor check stays aligned). That residual
    # hazard is exactly why oracle-equivalence tests are part of the
    # acceptance gates rather than optional.
    swapped = GatedDeltaRuleSpec.from_tensors(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        g.transpose(1, 2).contiguous(),
        beta.transpose(1, 2).contiguous(),
    )
    assert swapped.sequence == 4 and swapped.heads == 16
    with pytest.raises(ValueError, match="GVA"):
        GatedDeltaRuleSpec.from_tensors(
            q, k, v[:, :, :-1], g[:, :, :-1], beta[:, :, :-1]
        )
    with pytest.raises(ValueError, match="supports"):
        GatedDeltaRuleSpec.from_tensors(q.float(), k.float(), v.float(), g, beta)
    with pytest.raises(ValueError, match="dtypes must match"):
        GatedDeltaRuleSpec.from_tensors(q, k.half(), v.half(), g, beta)
    with pytest.raises(ValueError, match="gate must have shape"):
        GatedDeltaRuleSpec.from_tensors(q, k, v, g[..., :2], beta)
    with pytest.raises(ValueError, match="unknown adapter mode|mode must be"):
        GatedDeltaRuleSpec.from_tensors(q, k, v, g, beta, mode="scan")


def test_flavor_identity_is_recorded_or_not_applicable() -> None:
    identity = fla_version()
    if "package" in identity:
        assert identity["version"]
        assert identity["license"] == "MIT"
        assert "externally" in identity["usage"] or "external" in identity["usage"]
    else:
        assert identity["status"] == "not_applicable"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_matches_fp32_oracle_prefill_and_decode(dtype) -> None:
    q, k, v, g, beta = _sample(dtype=dtype)
    state0 = torch.randn((2, 4, 32, 48), device="cuda", dtype=torch.float32)

    reference, ref_state = oracle_gated_delta_rule(
        q, k, v, g, beta, initial_state=state0.clone(), output_final_state=True
    )
    prefill = UrmGatedDeltaRuleAdapter("prefill")
    out, final = prefill.execute(
        q, k, v, g, beta, initial_state=state0.clone(), output_final_state=True
    )
    _assert_close(out, reference, **OUTPUT_TOL[dtype])
    _assert_close(final, ref_state, **STATE_TOL)

    decode = UrmGatedDeltaRuleAdapter("decode")
    single, single_state = decode.execute(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        g[:, :1],
        beta[:, :1],
        initial_state=state0.clone(),
        output_final_state=True,
    )
    ref_single, ref_single_state = oracle_gated_delta_rule(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        g[:, :1],
        beta[:, :1],
        initial_state=state0.clone(),
        output_final_state=True,
    )
    _assert_close(single, ref_single, **OUTPUT_TOL[dtype])
    _assert_close(single_state, ref_single_state, **STATE_TOL)


def test_initial_and_final_state_semantics() -> None:
    q, k, v, g, beta = _sample(t=8)
    nonzero = torch.randn((2, 4, 32, 48), device="cuda", dtype=torch.float32)
    zeros = torch.zeros_like(nonzero)

    out_nonzero, final_nonzero = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, g, beta, initial_state=nonzero.clone(), output_final_state=True
    )
    out_zero, final_zero = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, g, beta, initial_state=zeros.clone(), output_final_state=True
    )
    # A carried state must change outputs and persist into the final state.
    assert not torch.allclose(out_nonzero.float(), out_zero.float())
    _reference_carry, ref_carry = oracle_gated_delta_rule(
        q, k, v, g, beta, initial_state=nonzero.clone(), output_final_state=True
    )
    _assert_close(final_nonzero, ref_carry, **STATE_TOL)
    _reference_fresh, ref_fresh = oracle_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True
    )
    _assert_close(final_zero, ref_fresh, **STATE_TOL)

    # Final state is returned only when requested.
    _, none_state = UrmGatedDeltaRuleAdapter("prefill").execute(q, k, v, g, beta)
    assert none_state is None


def test_gradients_match_eager_baseline_including_initial_state() -> None:
    q, k, v, g, beta = _sample(b=1, t=12, h=2, k=16, v=24)
    state0 = torch.randn((1, 2, 16, 24), device="cuda", dtype=torch.float32)

    leaves = [
        tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)
    ]
    state_leaf = state0.detach().clone().requires_grad_(True)
    out, final = UrmGatedDeltaRuleAdapter("prefill").execute(
        *leaves, initial_state=state_leaf, output_final_state=True
    )
    grad_out = torch.randn_like(out, dtype=torch.float32)
    grad_state = torch.randn_like(final, dtype=torch.float32)
    torch.autograd.backward([out, final], [grad_out, grad_state])
    adapter_grads = [tensor.grad.clone() for tensor in (*leaves, state_leaf)]

    eager_leaves = [
        tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)
    ]
    eager_state = state0.detach().clone().requires_grad_(True)
    eager_out, eager_final = eager_gated_delta_rule(
        *eager_leaves, initial_state=eager_state, output_final_state=True
    )
    torch.autograd.backward([eager_out, eager_final], [grad_out, grad_state])

    names = ("dq", "dk", "dv", "dg", "dbeta", "dstate0")
    for name, got, want in zip(
        names, adapter_grads, [t.grad for t in (*eager_leaves, eager_state)]
    ):
        assert got is not None and torch.isfinite(got).all(), name
        # bf16 kernel gradients versus an fp32 eager recurrence: tolerances
        # are dtype-scaled per docs/fla-gated-delta-rule.md.
        torch.testing.assert_close(got.float(), want.float(), atol=5e-2, rtol=2e-2)


def test_decode_is_forward_only_upstream() -> None:
    q, k, v, g, beta = _sample(b=1, t=1)
    leaves = [
        tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)
    ]
    out, _ = UrmGatedDeltaRuleAdapter("decode").execute(*leaves)
    loss = out.float().sum()
    with pytest.raises((NotImplementedError, RuntimeError)):
        loss.backward()


def test_sequence_one_decode_matches_chunk_prefill() -> None:
    q, k, v, g, beta = _sample(b=2, t=1, h=4, seed=9)
    state0 = torch.randn((2, 4, 32, 48), device="cuda", dtype=torch.float32)
    decode_out, decode_state = UrmGatedDeltaRuleAdapter("decode").execute(
        q, k, v, g, beta, initial_state=state0.clone(), output_final_state=True
    )
    chunk_out, chunk_state = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, g, beta, initial_state=state0.clone(), output_final_state=True
    )
    _assert_close(decode_out, chunk_out, atol=2e-2, rtol=2e-2)
    _assert_close(decode_state, chunk_state, atol=6e-2, rtol=2e-2)


def test_non_power_of_two_lengths() -> None:
    for t in (1, 7, 37, 100):
        q, k, v, g, beta = _sample(b=2, t=t, seed=t)
        reference, _ref_state = oracle_gated_delta_rule(q, k, v, g, beta)
        out, _final = UrmGatedDeltaRuleAdapter("prefill").execute(q, k, v, g, beta)
        _assert_close(out, reference, atol=2e-2, rtol=2e-2)


def test_gate_limits_and_beta_boundaries() -> None:
    q, k, v, g, beta = _sample(b=1, t=10, h=2, k=16, v=16, seed=13)

    # Zero gate: no decay; strongly negative gate decays everything toward zero
    # before each write.
    out_no_decay, _ = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, torch.zeros_like(g), beta
    )
    ref_no_decay, _ = oracle_gated_delta_rule(q, k, v, torch.zeros_like(g), beta)
    _assert_close(out_no_decay, ref_no_decay, atol=2e-2, rtol=2e-2)

    strong = torch.full_like(g, -20.0)
    out_strong, state_strong = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, strong, beta, output_final_state=True
    )
    ref_strong, ref_state_strong = oracle_gated_delta_rule(
        q, k, v, strong, beta, output_final_state=True
    )
    _assert_close(out_strong, ref_strong, atol=2e-2, rtol=2e-2)
    _assert_close(state_strong, ref_state_strong, atol=6e-2, rtol=2e-2)

    # beta ~ 0 suppresses writes entirely: state is pure decay of the initial
    # value, S_final = exp(sum(g)) * S_0.
    zero_beta = torch.zeros_like(beta)
    init = torch.ones((1, 2, 16, 16), device="cuda", dtype=torch.float32)
    _, state_beta0 = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, g, zero_beta, initial_state=init.clone(), output_final_state=True
    )
    decayed = init * torch.exp(g.cumsum(dim=1)[:, -1]).view(1, 2, 1, 1)
    torch.testing.assert_close(state_beta0.float(), decayed, atol=6e-2, rtol=2e-2)

    # With normalized k and beta == 1, retrieval after the update returns v
    # exactly (the delta rule's exact-write property).
    ones_beta = torch.ones_like(beta)
    out_exact, _state = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, torch.full_like(g, -20.0), ones_beta
    )
    ref_exact, _ = oracle_gated_delta_rule(
        q, k, v, torch.full_like(g, -20.0), ones_beta
    )
    _assert_close(out_exact, ref_exact, atol=2e-2, rtol=2e-2)


def test_multiple_batches_heads_and_gva_grouping() -> None:
    # HV > H grouped-value-attention case against the oracle.
    q, k, v, g, beta = _sample(b=3, t=14, h=2, hv=6, seed=17)
    reference, ref_state = oracle_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True
    )
    out, final = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, g, beta, output_final_state=True
    )
    _assert_close(out, reference, atol=2e-2, rtol=2e-2)
    _assert_close(final, ref_state, atol=6e-2, rtol=2e-2)
    assert out.shape == (3, 14, 6, 48)
    assert final.shape == (3, 6, 32, 48)


def test_deterministic_repeatable_forward() -> None:
    q, k, v, g, beta = _sample(seed=23)
    first, first_state = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, g, beta, output_final_state=True
    )
    second, second_state = UrmGatedDeltaRuleAdapter("prefill").execute(
        q, k, v, g, beta, output_final_state=True
    )
    assert torch.equal(first, second)
    assert torch.equal(first_state, second_state)


def test_adapter_rejects_mode_mismatch() -> None:
    q, k, v, g, beta = _sample(b=1, t=4)
    with pytest.raises(RuntimeError, match="unsupported configuration"):
        UrmGatedDeltaRuleAdapter("decode").execute(q, k, v, g, beta)


def test_committed_gated_delta_rule_artifact_validates_against_schema() -> None:
    schema_path = PROJECT_ROOT / "benchmarks" / "gated-delta-rule-result-schema.json"
    artifact_path = PROJECT_ROOT / "results" / "fla-gated-delta-rule" / "benchmark.json"
    if not artifact_path.exists():
        pytest.skip("gated delta-rule benchmark has not been run yet")
    from jsonschema import validate

    validate(json.loads(artifact_path.read_text()), json.loads(schema_path.read_text()))
