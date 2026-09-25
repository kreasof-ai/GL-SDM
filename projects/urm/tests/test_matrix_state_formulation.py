"""Independent algebra and adjoint checks for the K2 matrix-state oracle.

NumPy only, no GPU dependency. Mirrors ``test_sparse_slot_formulation.py``: the
chunked parallel form must match the independent sequential recurrence in reals,
and the analytical adjoint must match finite differences of the chunked form.
"""

import numpy as np
import pytest

from urm.backends.numpy.k2.linear_delta import chunked, recurrent, recurrent_vjp
from urm.backends.numpy.k3.sparse_state import chunked as sparse_chunked
from urm.backends.numpy.k3.sparse_state import recurrent as sparse_recurrent


def sample(seed=3, t=7, k=5, d=3):
    rng = np.random.default_rng(seed)
    return {
        "memory": rng.normal(size=(k, d)),
        "keys": rng.normal(size=(t, k)),
        "queries": rng.normal(size=(t, k)),
        "values": rng.normal(size=(t, d)),
        "beta": rng.uniform(0.1, 0.9, t),
        "log_decay": -rng.uniform(0, 0.6, t),
    }


@pytest.mark.parametrize("seed", [3, 19, 42])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 16])
def test_chunked_matches_independent_recurrence(seed, chunk_size):
    args = sample(seed)
    expected = recurrent(**args)
    actual = chunked(**args, chunk_size=chunk_size)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_nonzero_initial_state_and_continuation(chunk_size):
    # A nonzero boundary state must decay and be read; then the final state of a
    # first segment must continue a second segment exactly.
    args = sample(seed=11, t=6)
    args["memory"] = np.ones_like(args["memory"])
    out_full, state_full = recurrent(**args)
    out_chunk, state_chunk = chunked(**args, chunk_size=chunk_size)
    np.testing.assert_allclose(out_chunk, out_full, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(state_chunk, state_full, rtol=2e-12, atol=2e-12)

    # Continuation: split the sequence, run both halves, and confirm equality
    # with the single full run (the boundary state carries across exactly).
    split = 4
    first = {k: (v[:split] if k in ("keys", "queries", "values", "beta", "log_decay") else v)
             for k, v in args.items()}
    out_a, boundary = recurrent(**first)
    second = {k: (v[split:] if k in ("keys", "queries", "values", "beta", "log_decay") else v)
              for k, v in args.items()}
    second["memory"] = boundary
    out_b, state_b = recurrent(**second)
    np.testing.assert_allclose(
        np.concatenate([out_a, out_b]), out_full, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(state_b, state_full, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_long_decay_does_not_erase_recent_write(chunk_size):
    t = 32
    v = np.zeros((t, 1))
    v[-1] = 1
    args = {
        "memory": np.zeros((1, 1)),
        "keys": np.ones((t, 1)),
        "queries": np.ones((t, 1)),
        "values": v,
        "beta": np.ones(t),
        "log_decay": -np.ones(t),
    }
    y, m = chunked(**args, chunk_size=chunk_size)
    np.testing.assert_allclose(y[-1], 1)
    np.testing.assert_allclose(m, 1)


def test_read_before_update_reads_pre_update_state():
    args = sample(seed=5, t=5)
    out_after, _ = recurrent(**args, read_before_update=False)
    out_before, _ = recurrent(**args, read_before_update=True)
    # With a zero initial state, the before-update read of token 0 sees nothing,
    # while the after-update read sees the first write.
    args_zero = {**args, "memory": np.zeros_like(args["memory"])}
    ob, _ = recurrent(**args_zero, read_before_update=True)
    np.testing.assert_allclose(ob[0], 0.0, atol=1e-15)
    assert not np.allclose(out_before, out_after)


@pytest.mark.parametrize(
    "name", ["memory", "keys", "queries", "values", "beta", "log_decay"]
)
def test_all_adjoints_against_chunked_finite_differences(name):
    args = sample()
    rng = np.random.default_rng(51)
    dy = rng.normal(size=args["values"].shape)
    dm = rng.normal(size=args["memory"].shape)
    analytical = recurrent_vjp(**args, output_cotangent=dy, final_cotangent=dm)[name]
    x = args[name]
    numerical = np.zeros_like(x)
    epsilon = 1e-6
    for index in np.ndindex(x.shape):
        losses = []
        for sign in [1, -1]:
            perturbed = x.copy()
            perturbed[index] += sign * epsilon
            y, m = chunked(**{**args, name: perturbed}, chunk_size=3)
            losses.append(np.sum(y * dy) + np.sum(m * dm))
        numerical[index] = (losses[0] - losses[1]) / (2 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_k2_is_dense_specialization_of_k3(seed):
    """K2 with dense keys is K3 with every slot selected (acceptance section 3).

    The shared chunked formulation specializes K3 by selecting every slot and
    setting the write weights to the dense key vector; scalar head decay then
    matches the per-slot decay of an all-selected K3 partition. Preserve this
    identity as a regression so it is not rediscovered architecture by
    architecture.
    """
    rng = np.random.default_rng(seed)
    t = int(rng.integers(1, 12))
    k = int(rng.integers(1, 6))
    d = int(rng.integers(1, 5))
    chunk_size = int(rng.integers(1, t + 2))
    memory = rng.normal(size=(k, d))
    keys = rng.normal(size=(t, k))
    queries = rng.normal(size=(t, k))
    values = rng.normal(size=(t, d))
    beta = rng.uniform(0.1, 0.9, t)
    log_decay = -rng.uniform(0, 0.6, t)
    selected = np.ones((t, k), dtype=bool)

    out_dense, state_dense = chunked(
        memory, keys, queries, values, beta, log_decay, chunk_size=chunk_size
    )
    out_sparse, state_sparse = sparse_chunked(
        memory, keys, queries, values, beta, log_decay, selected,
        chunk_size=chunk_size,
    )
    np.testing.assert_allclose(out_dense, out_sparse, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(state_dense, state_sparse, rtol=2e-12, atol=2e-12)

    # The sequential forms agree too.
    out_dense_r, state_dense_r = recurrent(
        memory, keys, queries, values, beta, log_decay
    )
    out_sparse_r, state_sparse_r = sparse_recurrent(
        memory, keys, queries, values, beta, log_decay, selected
    )
    np.testing.assert_allclose(out_dense_r, out_sparse_r, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(state_dense_r, state_sparse_r, rtol=2e-12, atol=2e-12)


def test_decay_gradient_is_nonzero_and_finite():
    # A nonzero decay cotangent must flow: increasing the (negative) log decay
    # toward zero preserves more state, so the gradient carries a definite sign.
    args = sample(seed=23, t=6)
    rng = np.random.default_rng(7)
    dy = rng.normal(size=args["values"].shape)
    dm = rng.normal(size=args["memory"].shape)
    grads = recurrent_vjp(**args, output_cotangent=dy, final_cotangent=dm)
    assert grads["log_decay"].shape == args["log_decay"].shape
    assert np.all(np.isfinite(grads["log_decay"]))
    assert np.any(grads["log_decay"] != 0.0)
