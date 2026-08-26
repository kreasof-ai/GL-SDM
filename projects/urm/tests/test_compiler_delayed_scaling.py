"""Differential proof: CODA-style delayed row scaling through a linear map.

Proves Linear(RowScale(x, r), W) <-> RowScale(Linear(x, W), r) at the tensor
level with explicit PyTorch references - forward equivalence, gradients for
x, W and r, batch/head reshapes, non-contiguous inputs, dtype envelopes, and
the structured rejections that keep the rule honest.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from urm.compiler.rewrite import DELAY_ROW_SCALE_THROUGH_GEMM, RewriteEngine
from urm.compiler.semantic import (
    DType,
    Matmul,
    SemanticProgram,
    TensorHandle,
    Transform,
    TransformKind,
)

# -- reference implementations -------------------------------------------------


def _materialized(x, w, r):
    """RowScale then Linear (two materializations)."""
    return (r[:, None] * x) @ w


def _delayed(x, w, r):
    """Linear then RowScale (scale folded into the GEMM output epilogue)."""
    return r[:, None] * (x @ w)


def _check_grads(dtype, atol, rtol, m=17, k=33, n=29):
    generator = torch.Generator().manual_seed(5)
    x0 = torch.randn((m, k), generator=generator).to(dtype)
    w0 = torch.randn((k, n), generator=generator).to(dtype)
    r0 = torch.randn((m,), generator=generator).to(dtype)
    grad = torch.randn((m, n)).to(dtype)

    xm, wm, rm = (t.clone().requires_grad_(True) for t in (x0, w0, r0))
    _materialized(xm, wm, rm).backward(grad)
    xd, wd, rd = (t.clone().requires_grad_(True) for t in (x0, w0, r0))
    _delayed(xd, wd, rd).backward(grad)

    out_forward_gap = (_materialized(x0, w0, r0) - _delayed(x0, w0, r0)).abs().max()
    assert out_forward_gap <= atol

    for name, a, b in (
        ("dx", xm.grad, xd.grad),
        ("dW", wm.grad, wd.grad),
        ("dr", rm.grad, rd.grad),
    ):
        assert a is not None and b is not None, name
        gap = (a.float() - b.float()).abs().max()
        assert gap <= atol + rtol * b.float().abs().max(), f"{name}: {gap}"


def test_fp32_forward_and_all_gradients_match() -> None:
    _check_grads(torch.float32, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "dtype,atol",
    [(torch.bfloat16, 9e-2), (torch.float16, 4e-2)],
)
def test_half_precision_envelopes(dtype, atol) -> None:
    _check_grads(dtype, atol=atol, rtol=1e-2)


def test_batch_head_reshaped_inputs() -> None:
    """b/h leading dims flatten to one logical row axis; identity still holds."""
    generator = torch.Generator().manual_seed(7)
    b, h, m, k, n = 2, 3, 5, 8, 4
    x = torch.randn((b, h, m, k), generator=generator)
    w = torch.randn((k, n), generator=generator)
    r = torch.randn((b, h, m), generator=generator)
    flat_x = x.reshape(-1, k)
    flat_r = r.reshape(-1)
    materialized = (flat_r[:, None] * flat_x) @ w
    delayed = flat_r[:, None] * (flat_x @ w)
    assert torch.allclose(materialized, delayed, atol=1e-5)
    assert materialized.shape == (b * h * m, n)


def test_non_contiguous_inputs_supported() -> None:
    generator = torch.Generator().manual_seed(9)
    base = torch.randn((64, 40), generator=generator)
    x_t = base.t()[:, :33]  # non-contiguous view [40, 33]
    w = torch.randn((33, 21), generator=generator)
    r = torch.randn((40,), generator=generator)
    assert not x_t.is_contiguous()
    materialized = (r[:, None] * x_t) @ w
    delayed = r[:, None] * (x_t @ w)
    assert torch.allclose(materialized, delayed, atol=1e-5)


def test_zero_scales_are_preserved_exactly() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn((9, 12), generator=generator)
    w = torch.randn((12, 7), generator=generator)
    r = torch.zeros((9,))
    assert torch.equal(_materialized(x, w, r), _delayed(x, w, r))
    assert torch.equal(_delayed(x, w, r), torch.zeros((9, 7)))


def test_repeated_rows_and_non_power_of_two_dims() -> None:
    generator = torch.Generator().manual_seed(13)
    k, n = 37, 19
    x = torch.randn((3, k), generator=generator).repeat(5, 1)  # repeated rows
    w = torch.randn((k, n), generator=generator)
    r = torch.randn((3,), generator=generator).repeat(5)
    assert torch.allclose(_materialized(x, w, r), _delayed(x, w, r), atol=1e-5)


# -- IR-level rejection semantics -----------------------------------------------


def _program_with_transform(kind: TransformKind) -> SemanticProgram:
    return SemanticProgram.build(
        name="reject_case",
        inputs=(
            TensorHandle("x", DType.FLOAT32, ("m", "k")),
            TensorHandle("W", DType.FLOAT32, ("k", "n")),
            TensorHandle("r", DType.FLOAT32, ("m",)),
        ),
        ops=(
            Transform(name="t", inputs=("x", "r"), outputs=("xs",), kind=kind),
            Matmul(name="linear", inputs=("xs", "W"), outputs=("y",)),
        ),
        outputs=("y",),
    )


def test_rule_rejects_nonlinear_intervening_transform() -> None:
    program = _program_with_transform(TransformKind.GELU)
    engine = RewriteEngine([DELAY_ROW_SCALE_THROUGH_GEMM])
    result = engine.apply(program)
    assert not result.changed
    rejection = next(a for a in result.trace.attempts if a.outcome == "rejected")
    assert rejection.reason_code == "rewrite_precondition_failed"


def test_rule_requires_matmul_subject() -> None:
    # Scale feeding an elementwise chain (no Matmul subject): nothing to match.
    program = SemanticProgram.build(
        name="no_gemm",
        inputs=(
            TensorHandle("x", DType.FLOAT32, ("m", "k")),
            TensorHandle("r", DType.FLOAT32, ("m",)),
        ),
        ops=(
            Transform(
                name="t",
                inputs=("x", "r"),
                outputs=("xs",),
                kind=TransformKind.ROW_SCALE,
            ),
        ),
        outputs=("xs",),
    )
    engine = RewriteEngine([DELAY_ROW_SCALE_THROUGH_GEMM])
    assert engine.candidates_for(program) == ()
