"""Independent algebra and adjoint checks; NumPy only, no GPU dependency."""

import numpy as np
import pytest

from urm.oracles.sparse_slot import chunked, recurrent, recurrent_vjp


def sample(seed=3):
    rng = np.random.default_rng(seed)
    t, s, d = 7, 4, 3
    selected = rng.random((t, s)) > 0.3
    return {
        "memory": rng.normal(size=(s, d)),
        "writes": rng.uniform(0, 0.4, (t, s)) * selected,
        "reads": rng.normal(size=(t, s)),
        "values": rng.normal(size=(t, d)),
        "beta": rng.uniform(0.1, 0.9, t),
        "log_decay": -rng.uniform(0, 2, t),
        "selected": selected,
    }


@pytest.mark.parametrize("seed", [3, 19, 42])
@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7, 16])
def test_chunked_matches_independent_recurrence(seed, chunk_size):
    args = sample(seed)
    expected = recurrent(**args)
    actual = chunked(**args, chunk_size=chunk_size)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_long_decay_does_not_erase_recent_write(chunk_size):
    v = np.zeros((32, 1))
    v[-1] = 1
    args = {
        "memory": np.zeros((1, 1)),
        "writes": np.ones((32, 1)),
        "reads": np.ones((32, 1)),
        "values": v,
        "beta": np.ones(32),
        "log_decay": -np.ones(32),
        "selected": np.ones((32, 1), dtype=bool),
    }
    y, m = chunked(**args, chunk_size=chunk_size)
    np.testing.assert_allclose(y[-1], 1)
    np.testing.assert_allclose(m, 1)


def test_selected_zero_weight_still_decays():
    args = {
        "memory": np.ones((1, 1)),
        "writes": np.zeros((2, 1)),
        "reads": np.ones((2, 1)),
        "values": np.zeros((2, 1)),
        "beta": np.zeros(2),
        "log_decay": -np.ones(2),
        "selected": np.ones((2, 1), dtype=bool),
    }
    y, m = chunked(**args, chunk_size=1)
    np.testing.assert_allclose(y[:, 0], np.exp([-1, -2]))
    np.testing.assert_allclose(m, np.exp(-2))


@pytest.mark.parametrize(
    "name", ["memory", "writes", "reads", "values", "beta", "log_decay"]
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
        if name == "writes" and not args["selected"][index]:
            continue
        losses = []
        for sign in [1, -1]:
            perturbed = x.copy()
            perturbed[index] += sign * epsilon
            y, m = chunked(**{**args, name: perturbed}, chunk_size=3)
            losses.append(np.sum(y * dy) + np.sum(m * dm))
        numerical[index] = (losses[0] - losses[1]) / (2 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-8)
