import numpy as np
import pytest

from urm.ir import CollisionPolicy
from urm.presets import (
    BLOCK_SPARSE_ATTENTION,
    DENSE_ATTENTION,
    LINEAR_RECURRENT_MIXER,
    TOP2_MOE,
)
from urm.reference import execute, merge_writes


def test_dense_softmax_reduce_matches_manual_result() -> None:
    scores = np.array([[0.0, 0.0], [0.0, np.log(3.0)]])
    values = np.array([[2.0, 4.0], [6.0, 8.0]])

    result = execute(scores, values, DENSE_ATTENTION)

    np.testing.assert_allclose(result.weights[0], [0.5, 0.5])
    np.testing.assert_allclose(result.weights[1], [0.25, 0.75])
    np.testing.assert_allclose(result.output, [[4.0, 6.0], [5.0, 7.0]])


def test_block_sparse_mask_excludes_sources() -> None:
    scores = np.zeros((2, 3))
    values = np.array([[1.0], [10.0], [100.0]])
    mask = np.array([[True, False, True], [False, True, False]])

    result = execute(scores, values, BLOCK_SPARSE_ATTENTION, route_mask=mask)

    np.testing.assert_allclose(result.output, [[50.5], [10.0]])
    np.testing.assert_allclose(result.weights.sum(axis=1), [1.0, 1.0])


def test_top_k_is_stable_for_tied_scores() -> None:
    scores = np.array([[4.0, 4.0, 4.0, 1.0]])
    values = np.arange(4, dtype=np.float64)[:, None]

    result = execute(scores, values, TOP2_MOE)

    np.testing.assert_array_equal(result.indices, [[0, 1]])
    np.testing.assert_allclose(result.weights, [[0.5, 0.5]])


def test_every_query_needs_a_valid_route() -> None:
    with pytest.raises(ValueError, match="at least one valid route"):
        execute(
            np.zeros((1, 2)),
            np.ones((2, 1)),
            BLOCK_SPARSE_ATTENTION,
            route_mask=np.zeros((1, 2), dtype=bool),
        )


def test_top_k_requires_enough_valid_routes() -> None:
    with pytest.raises(ValueError, match="top_k valid routes"):
        execute(
            np.zeros((1, 3)),
            np.ones((3, 1)),
            TOP2_MOE,
            route_mask=np.array([[True, False, False]]),
        )


def test_recurrence_requires_dedicated_lowering() -> None:
    with pytest.raises(NotImplementedError, match="ordered scan semantics"):
        execute(np.ones((1, 1)), np.ones((1, 1)), LINEAR_RECURRENT_MIXER)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (CollisionPolicy.SUM, [[4.0, 6.0], [5.0, 6.0]]),
        (CollisionPolicy.MEAN, [[2.0, 3.0], [5.0, 6.0]]),
        (CollisionPolicy.LAST_WRITE, [[3.0, 4.0], [5.0, 6.0]]),
    ],
)
def test_merge_writes(policy: CollisionPolicy, expected: list[list[float]]) -> None:
    addresses, deltas, collisions = merge_writes(
        [7, 7, 9],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        policy,
    )

    np.testing.assert_array_equal(addresses, [7, 9])
    np.testing.assert_allclose(deltas, expected)
    assert collisions == 1


def test_merge_writes_can_reject_collisions() -> None:
    with pytest.raises(ValueError, match="collisions"):
        merge_writes([3, 3], [[1.0], [2.0]], CollisionPolicy.REJECT)


def test_ordered_updates_are_not_reduced_as_a_transaction() -> None:
    with pytest.raises(ValueError, match="dedicated lowering"):
        merge_writes([3, 3], [[1.0], [2.0]], CollisionPolicy.ORDERED)
