"""Algebra checks only; these do not certify the BF16 numerical contract."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from urm.backends.sparse_state_reference import torch_sparse_state_mixer
from urm.backends.sparse_state_triangular import torch_selected_slot_triangular
from urm.compiler.semantic import SparseReadTiming


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_comparator_padding_preserves_real_outputs_final_memory_and_all_vjps(dtype):
    sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
    from sparse_state_triangular import pad_upstream_partition

    generator = torch.Generator().manual_seed(832)
    t, slots, dim = 35, 17, 2
    wi = torch.stack((torch.arange(t) % slots, (torch.arange(t) + 2) % slots), -1)[None]
    ri = wi.flip(-1)
    shapes = ((1, slots, dim), (1, t, 2), (1, t, dim), (1, t, 1), (1, t, 1), (1, t, 2))
    data = [
        torch.rand(shape, generator=generator, dtype=torch.float64).to(dtype) * 0.1
        for shape in shapes
    ]
    data[4] = -data[4]
    ct = torch.randn(1, t, dim, generator=generator).to(dtype)
    mt = torch.randn(1, slots, dim, generator=generator).to(dtype)
    rows = []
    for padded in (False, True):
        leaves = [x.clone().requires_grad_() for x in data]
        m, w, v, b, g, q = leaves
        i, j = wi, ri
        if padded:
            i, w, v, b, g, j, q = pad_upstream_partition(i, w, v, b, g, j, q)
            assert i.shape[1] == 48
        y, final = torch_sparse_state_mixer(
            m,
            j,
            q,
            write_indices=i,
            write_weights=w,
            values=v,
            beta=b,
            log_decay=g,
            read_timing=SparseReadTiming.AFTER_UPDATE,
            accumulation_dtype=torch.float64
            if dtype is torch.float64
            else torch.float32,
        )
        y = y[:, :t]
        gradients = torch.autograd.grad((y * ct).sum() + (final * mt).sum(), leaves)
        rows.append((y, final, *gradients))
    for a, b in zip(*rows, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_selected_slot_decays_even_with_zero_write_weight():
    memory = torch.ones(1, 2, 1, dtype=torch.float64)
    indices = torch.zeros(1, 16, 1, dtype=torch.long)
    scalar = torch.zeros(1, 16, 1, dtype=torch.float64)
    _, final = torch_selected_slot_triangular(
        memory,
        indices,
        scalar + 1,
        write_indices=indices,
        write_weights=scalar,
        values=scalar,
        beta=scalar,
        log_decay=scalar - 0.1,
        accumulation_dtype=torch.float64,
    )
    torch.testing.assert_close(
        final[0, 0, 0], torch.exp(torch.tensor(-1.6, dtype=torch.float64))
    )
    assert final[0, 1, 0] == 1  # an unselected slot is never decayed


def test_all_differentiable_inputs_pass_finite_difference_gradcheck():
    t, slots = 17, 3
    generator = torch.Generator().manual_seed(8701)
    wi = torch.stack((torch.arange(t) % slots, (torch.arange(t) + 1) % slots), -1)[None]
    ri = wi.flip(-1)
    shapes = ((1, slots, 1), (1, t, 2), (1, t, 1), (1, t, 1), (1, t, 1), (1, t, 2))
    leaves = [
        torch.rand(shape, generator=generator, dtype=torch.float64) * 0.1
        for shape in shapes
    ]
    leaves[4] = -leaves[4] - 0.1
    leaves = tuple(x.requires_grad_() for x in leaves)

    def call(m, w, v, b, g, q):
        return torch_selected_slot_triangular(
            m,
            ri,
            q,
            write_indices=wi,
            write_weights=w,
            values=v,
            beta=b,
            log_decay=g,
            accumulation_dtype=torch.float64,
            chunk_size=16,
        )

    assert torch.autograd.gradcheck(call, leaves, fast_mode=True)


@pytest.mark.parametrize("chunk", [16, 32, 64])
@pytest.mark.parametrize(
    "timing", [SparseReadTiming.BEFORE_UPDATE, SparseReadTiming.AFTER_UPDATE]
)
def test_triangular_real_arithmetic_and_all_input_vjps(chunk, timing):
    generator = torch.Generator().manual_seed(561)
    t, s, d = chunk + 3, 11, 3
    wi = torch.stack((torch.arange(t) % s, (torch.arange(t) + 2) % s), -1)[None]
    ri = torch.stack((torch.arange(t) % s, (torch.arange(t) + 1) % s), -1)[None]
    data = [
        torch.randn(1, s, d, dtype=torch.float64, generator=generator) * 0.1,
        torch.rand(1, t, 2, dtype=torch.float64, generator=generator) * 0.4,
        torch.randn(1, t, d, dtype=torch.float64, generator=generator) * 0.1,
        torch.rand(1, t, 1, dtype=torch.float64, generator=generator),
        -torch.rand(1, t, 1, dtype=torch.float64, generator=generator) * 0.3,
        torch.rand(1, t, 2, dtype=torch.float64, generator=generator),
    ]
    ct = torch.randn(1, t, d, dtype=torch.float64, generator=generator)
    mt = torch.randn(1, s, d, dtype=torch.float64, generator=generator)
    rows = []
    for fn in (torch_sparse_state_mixer, torch_selected_slot_triangular):
        leaves = [x.clone().requires_grad_() for x in data]
        m, w, v, b, g, q = leaves
        kwargs = {"chunk_size": chunk} if fn is torch_selected_slot_triangular else {}
        y, final = fn(
            m,
            ri,
            q,
            write_indices=wi,
            write_weights=w,
            values=v,
            beta=b,
            log_decay=g,
            read_timing=timing,
            accumulation_dtype=torch.float64,
            **kwargs,
        )
        gradients = torch.autograd.grad((y * ct).sum() + (final * mt).sum(), leaves)
        rows.append((y, final, *gradients))
    for candidate, oracle in zip(rows[1], rows[0], strict=True):
        torch.testing.assert_close(candidate, oracle, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_bf16_intermediate_cast_counterexample_is_retained(chunk):
    t, s, d, w = 129, 64, 1, 64
    wi = torch.arange(w).expand(1, t, w)
    memory = torch.ones(1, s, d, dtype=torch.bfloat16)
    weights = torch.full((1, t, w), 1 / w, dtype=torch.bfloat16)
    kwargs = {
        "write_indices": wi,
        "write_weights": weights,
        "values": torch.zeros(1, t, d, dtype=torch.bfloat16),
        "beta": torch.full((1, t, 1), 0.001, dtype=torch.bfloat16),
        "log_decay": torch.full((1, t, 1), -0.001, dtype=torch.bfloat16),
        "read_timing": SparseReadTiming.AFTER_UPDATE,
    }
    oracle = torch_sparse_state_mixer(memory, wi, weights, **kwargs)
    candidate = torch_selected_slot_triangular(
        memory, wi, weights, chunk_size=chunk, **kwargs
    )
    assert torch.equal(oracle[1], memory)  # each tiny update rounds back to 1
    # Preserve the failure under the existing BF16 forward atol=rtol=.02.
    assert not torch.allclose(candidate[1], oracle[1], atol=0.02, rtol=0.02)
