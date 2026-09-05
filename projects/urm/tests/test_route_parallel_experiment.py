"""Isolated mechanism and benchmark-authority regressions, no dispatch changes."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
from state_tensor_audit import persistent_audit, tensor_audit


def test_state_audit_finds_cancelled_errors_without_replacing_checksum():
    ref = torch.zeros(1, 2, 4)
    value = ref.clone()
    value[0, 0, 1], value[0, 1, 3] = 1.0, -1.0
    checksum = [[{"mean": 0.0}]]
    result = persistent_audit([[ref]], [[value]], checksum, checksum)[0]
    assert result["historical_checksum_passed"]
    assert not result["passed"]
    assert result["diagnostic_only"]
    assert result["checksum_and_tensor_diagnostic_disagree"]
    assert result["violating_elements"] == 2
    assert result["worst_coordinate"] == [0, 0, 1]
    assert result["worst_candidate"] == 1.0 and result["worst_reference"] == 0.0


def test_state_audit_retains_original_gpu_checksum_not_cpu_reduction():
    value = torch.zeros(1, 2, 4)
    result = persistent_audit(
        [[value]], [[value]], [[{"mean": 0}]], [[{"mean": 3e-6}]]
    )[0]
    assert result["passed"] and not result["historical_checksum_passed"]
    assert result["checksum_and_tensor_diagnostic_disagree"]


def test_state_audit_finiteness_and_tolerance_are_explicit():
    report = tensor_audit(
        torch.tensor([float("nan")]), torch.zeros(1), {"atol": 0.02, "rtol": 0.02}
    )
    assert not report["finite"] and not report["passed"]
    assert report["violating_elements"] == report["candidate_nonfinite"] == 1
    assert report["max_abs"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("d", [4, 64])
def test_nontrivial_joint_vjp_and_selected_zero_weight_decay(
    resident, before, dtype, d
):
    from urm.backends.sparse_state_reference import torch_sparse_state_mixer
    from urm.compiler.semantic import SparseReadTiming
    from urm.experiments.route_parallel import update

    torch.manual_seed(9191)
    p, t, s = 1, 3, 67
    wi = torch.arange(64, device="cuda").expand(p, t, -1).contiguous()
    ri = (wi + 1).contiguous()
    data = [
        torch.randn(p, s, d, device="cuda", dtype=dtype) * 0.05,
        torch.full((p, t, 64), 1 / 64, device="cuda", dtype=dtype),
        torch.randn(p, t, d, device="cuda", dtype=dtype) * 0.05,
        torch.full((p, t, 1), 0.1, device="cuda", dtype=dtype),
        torch.full((p, t, 1), -0.03, device="cuda", dtype=dtype),
        torch.full((p, t, 64), 1 / 64, device="cuda", dtype=dtype),
    ]
    data[1][..., 0] = 0  # selected row must decay even with zero weight
    ct_y = torch.randn(p, t, d, device="cuda")
    ct_f = torch.randn(p, s, d, device="cuda")
    runs = []
    for candidate in (False, True):
        xs = [x.clone().requires_grad_() for x in data]
        m, w, v, b, g, q = xs
        if candidate:
            y, f = update(m, wi, w, v, b, g, ri, q, resident=resident, before=before)
        else:
            y, f = torch_sparse_state_mixer(
                m,
                ri,
                q,
                write_indices=wi,
                write_weights=w,
                values=v,
                beta=b,
                log_decay=g,
                read_timing=SparseReadTiming.BEFORE_UPDATE
                if before
                else SparseReadTiming.AFTER_UPDATE,
            )
        # Deliberately NOT normalized: zero/missing VJPs must not hide below atol.
        gradients = torch.autograd.grad(
            (y.float() * ct_y).sum() + (f.float() * ct_f).sum(), xs
        )
        runs.append((y, f, gradients))
        assert all(float(x.abs().max()) > 1e-4 for x in gradients)
    forward_tol = (
        {"atol": 2e-5, "rtol": 2e-5}
        if dtype == torch.float32
        else {"atol": 0.02, "rtol": 0.02}
    )
    backward_tol = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 0.03, "rtol": 0.03}
    )
    for i in (0, 1):
        torch.testing.assert_close(runs[1][i], runs[0][i], **forward_tol)
    for a, b in zip(runs[1][2], runs[0][2], strict=True):
        torch.testing.assert_close(a, b, **backward_tol)
    torch.testing.assert_close(runs[1][1][:, 64:], data[0][:, 64:], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("resident", [False, True])
def test_backward_read_cotangent_is_cast_before_write_vjp(resident):
    from urm.experiments.route_parallel import backward

    dtype, device = torch.bfloat16, "cuda"
    wi = torch.arange(64, device=device).reshape(1, 1, 64)
    q = torch.full((1, 1, 64), 1 / 64, device=device, dtype=dtype)
    v = torch.zeros(1, 1, 4, device=device, dtype=dtype)
    v[0, 0, 0], v[0, 0, 1] = 1, -1
    dy = torch.zeros_like(v)
    dy[0, 0, 0] = 1 / 16
    df = torch.ones(1, 64, 4, device=device, dtype=dtype)
    histories = torch.zeros(1, 1, 64, 4, device=device, dtype=dtype)
    # +1/1024 in dimension zero rounds back to BF16 one BEFORE the write VJP.
    # Opposite values cancel grad_beta only if this intermediate cast occurs.
    # Omitting it produces grad_beta=1/1024 despite identical final grad_memory.
    grads = backward(
        dy,
        df,
        histories,
        histories,
        wi,
        q,
        v,
        torch.zeros(1, 1, 1, device=device, dtype=dtype),
        torch.zeros(1, 1, 1, device=device, dtype=dtype),
        wi,
        q,
        resident,
        False,
    )
    torch.testing.assert_close(grads[0], df, atol=0, rtol=0)
    torch.testing.assert_close(grads[3], torch.zeros_like(grads[3]), atol=0, rtol=0)
    counterfactual = (
        (df.float() + q[..., None].float() * dy[:, :, None].float())
        * q[..., None].float()
        * v[:, :, None].float()
    ).sum()
    assert float(counterfactual) == 1 / 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("loss_kind", ["readings_only", "final_only"])
def test_unused_output_cotangents(resident, loss_kind):
    from urm.experiments.route_parallel import update

    device = "cuda"
    m = torch.randn(1, 64, 4, device=device, requires_grad=True)
    wi = torch.arange(64, device=device).reshape(1, 1, 64)
    w = torch.full((1, 1, 64), 1 / 64, device=device, requires_grad=True)
    v = torch.ones(1, 1, 4, device=device, requires_grad=True)
    b = torch.ones(1, 1, 1, device=device, requires_grad=True) * 0.1
    g = torch.zeros_like(b, requires_grad=True)
    y, f = update(m, wi, w, v, b, g, wi, w, resident=resident)
    loss = (y if loss_kind == "readings_only" else f).sum()
    gradients = torch.autograd.grad(loss, (m, w, v, b, g))
    assert all(torch.isfinite(x).all() for x in gradients)
