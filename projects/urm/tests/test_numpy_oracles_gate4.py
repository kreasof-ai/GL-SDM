"""Gate 4: the independent NumPy oracle tier for TriangularSolve (UT) and
DyadicBankedState (A4) agrees with the Torch reference and the native Triton tier.

The compiler charter admits a new semantic axis in the order "new semantic axis →
IR + NumPy/Torch references BEFORE native execution". The TriangularSolve (UT
axis, K4 family) and DyadicBankedState (A4 axis, K2 family) ops shipped Torch
references and native Triton tiers but skipped the independent high-precision
equation step — a pure-NumPy float64 recurrence written from the equation, not
transliterated from the Torch code. This module closes that gap and pins it:

for each op, on shared operands, the NumPy oracle
(``urm.reference.numpy.k4.triangular_solve.v1`` /
``urm.reference.numpy.k2.dyadic_banked_state.v1``) is compared against the Torch
reference and the native Triton execution (both reached through the uniform
provider ``execute`` path). The oracle runs in float64 and is cast to float32
for comparison; parity is bounded at ~1e-4 for the float32 accumulation
differences between the fp32 tiers and the fp64 oracle.

CUDA + Triton are required for the native leg; the tests skip cleanly when
either is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.backends.contract import ProviderFamily, ProviderRequest
from urm.backends.numpy.k2.dyadic_banks import K2DyadicBankedStateNumpyProvider
from urm.backends.numpy.k4.triangular_solve import K4NumpyProvider
from urm.backends.registry import discover_providers
from urm.backends.torch.k2.dyadic_banks import DyadicBankedStateTorchReferenceProvider
from urm.backends.torch.k4.triangular_solve import TriangularSolveTorchReferenceProvider
from urm.backends.triton.k2.dyadic_banks import (
    DyadicBankedStateNativeTritonProvider,
)
from urm.backends.triton.k4.triangular_solve import (
    TriangularSolveNativeTritonProvider,
)
from urm.ir.program import DyadicBankedState, TriangularSolve

TOL = 1e-4


def _max_err(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def _triangular_solve_request():
    op = TriangularSolve(
        name="solve",
        inputs=("probs", "beta", "value"),
        outputs=("u",),
        roles=(("probs", "probs"), ("beta", "beta"), ("value", "value")),
    )
    return ProviderRequest(
        family=ProviderFamily.TRIANGULAR_SOLVE, descriptor=op, mode="inference"
    )


def _triangular_solve_operands(seed, b, h, t, d):
    """Strict-lower softmax probabilities, a uniform diagonal and a random RHS."""
    torch.manual_seed(seed)
    scores = torch.randn(b, h, t, t, device="cuda")
    strict = torch.triu(torch.ones(t, t, device="cuda", dtype=torch.bool), diagonal=0)
    probs = torch.nan_to_num(
        torch.softmax(scores.masked_fill(strict, float("-inf")), dim=-1), nan=0.0
    )
    beta = torch.rand(b, h, t, device="cuda")
    value = torch.randn(b, h, t, d, device="cuda")
    return probs, beta, value


@pytest.mark.parametrize("shape", [(2, 3, 33, 24), (2, 2, 64, 64), (1, 2, 128, 96)])
def test_triangular_solve_numpy_oracle_matches_torch_and_native(shape):
    """The float64 NumPy forward substitution matches both fp32 tiers."""
    b, h, t, d = shape
    probs, beta, value = _triangular_solve_operands(5, *shape)
    operands = {"probs": probs, "beta": beta, "value": value}
    request = _triangular_solve_request()

    np_out = K4NumpyProvider().execute(
        request, {k: v.cpu().numpy() for k, v in operands.items()}
    )["output"].astype(np.float32)
    torch_out = (
        TriangularSolveTorchReferenceProvider()
        .execute(request, operands)["output"]
        .cpu()
        .numpy()
    )
    native_out = (
        TriangularSolveNativeTritonProvider()
        .execute(request, operands)["output"]
        .cpu()
        .numpy()
    )

    err_torch = _max_err(np_out, torch_out)
    err_native = _max_err(np_out, native_out)
    assert err_torch < TOL, f"numpy-vs-torch: max abs err {err_torch}"
    assert err_native < TOL, f"numpy-vs-native: max abs err {err_native}"


def _dyadic_request(num_levels):
    op = DyadicBankedState(
        name="bank",
        inputs=("query", "key", "value", "log_decay", "level_scales"),
        outputs=("output",),
        num_levels=num_levels,
    )
    return ProviderRequest(
        family=ProviderFamily.DYADIC_BANKED_STATE, descriptor=op, mode="training"
    )


def _dyadic_operands(seed, b, t, h, d, levels):
    torch.manual_seed(seed)
    q = torch.randn(b, t, h, d, device="cuda")
    k = torch.randn(b, t, h, d, device="cuda")
    v = torch.randn(b, t, h, d, device="cuda")
    g = torch.nn.functional.logsigmoid(torch.randn(b, t, h, device="cuda"))
    ls = torch.randn(b, t, h, levels, device="cuda")
    return {"query": q, "key": k, "value": v, "log_decay": g, "level_scales": ls}


@pytest.mark.parametrize(
    "B, T, H, D, L",
    [
        (2, 8, 2, 8, 4),    # exact dyadic blocks (T = 2^(L-1))
        (2, 7, 3, 16, 4),   # partial dyadic block tail
        (2, 24, 4, 64, 6),  # multi-level hierarchy across level boundaries
    ],
)
def test_dyadic_banked_state_numpy_oracle_matches_torch_and_native(B, T, H, D, L):
    """The float64 NumPy banked dyadic recurrence matches both fp32 tiers."""
    operands = _dyadic_operands(5 + T, B, T, H, D, L)
    request = _dyadic_request(L)

    np_out = (
        K2DyadicBankedStateNumpyProvider()
        .execute(request, {k: v.cpu().numpy() for k, v in operands.items()})["output"]
        .astype(np.float32)
    )
    torch_out = (
        DyadicBankedStateTorchReferenceProvider()
        .execute(request, operands)["output"]
        .cpu()
        .numpy()
    )
    native_out = (
        DyadicBankedStateNativeTritonProvider()
        .execute(request, operands)["output"]
        .cpu()
        .numpy()
    )

    err_torch = _max_err(np_out, torch_out)
    err_native = _max_err(np_out, native_out)
    assert err_torch < TOL, f"numpy-vs-torch: max abs err {err_torch}"
    assert err_native < TOL, f"numpy-vs-native: max abs err {err_native}"


def test_gate4_numpy_providers_are_discovered():
    """Auto-discovery surfaces both new oracle anchors in the dispatch table."""
    providers = discover_providers()
    assert "urm.reference.numpy.k4.triangular_solve.v1" in providers
    assert "urm.reference.numpy.k2.dyadic_banked_state.v1" in providers
    for name in (
        "urm.reference.numpy.k4.triangular_solve.v1",
        "urm.reference.numpy.k2.dyadic_banked_state.v1",
    ):
        provider = providers[name]
        assert provider.tier == "reference"
        wrong = ProviderRequest(
            family=provider.family, descriptor=object(), mode="inference"
        )
        assert provider.decline(wrong) is not None
