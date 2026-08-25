"""Slow NumPy oracle for URM routing, reduction, and write merging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .ir import CollisionPolicy, MixerSpec, Normalization, RoutingKind

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    output: FloatArray
    indices: IntArray
    weights: FloatArray


def _normalize(selected_scores: FloatArray, mode: Normalization) -> FloatArray:
    valid = np.isfinite(selected_scores)

    if mode is Normalization.NONE:
        return np.where(valid, selected_scores, 0.0)
    if mode is Normalization.SIGMOID:
        clipped = np.clip(selected_scores, -80.0, 80.0)
        return np.where(valid, 1.0 / (1.0 + np.exp(-clipped)), 0.0)
    if mode is Normalization.L1:
        values = np.where(valid, selected_scores, 0.0)
        denominator = np.abs(values).sum(axis=-1, keepdims=True)
        return np.divide(
            values,
            denominator,
            out=np.zeros_like(values),
            where=denominator != 0,
        )
    if mode is Normalization.SOFTMAX:
        row_max = np.max(selected_scores, axis=-1, keepdims=True)
        exponentials = np.where(valid, np.exp(selected_scores - row_max), 0.0)
        denominator = exponentials.sum(axis=-1, keepdims=True)
        return exponentials / denominator

    raise AssertionError(f"unhandled normalization: {mode}")


def _route(
    scores: FloatArray,
    spec: MixerSpec,
    route_mask: npt.NDArray[np.bool_] | None,
) -> tuple[IntArray, FloatArray]:
    query_count, source_count = scores.shape
    if route_mask is None:
        valid_mask = np.ones(scores.shape, dtype=np.bool_)
    else:
        valid_mask = np.asarray(route_mask, dtype=np.bool_)
        if valid_mask.shape != scores.shape:
            raise ValueError("route_mask must have the same shape as scores")

    if not np.all(valid_mask.any(axis=1)):
        raise ValueError("every query must have at least one valid route")

    if spec.routing is RoutingKind.BLOCK_SPARSE and route_mask is None:
        raise ValueError("block-sparse routing requires route_mask")
    if spec.routing is RoutingKind.KERNELIZED_RECURRENCE:
        raise NotImplementedError(
            "recurrence has ordered scan semantics and needs a dedicated lowering"
        )

    masked_scores = np.where(valid_mask, scores, -np.inf)
    if spec.routing in {RoutingKind.TOP_K, RoutingKind.PRODUCT_KEY}:
        width = spec.top_k or 0
        if width > source_count:
            raise ValueError("top_k must not exceed the source count")
        if np.any(valid_mask.sum(axis=1) < width):
            raise ValueError("each query must expose at least top_k valid routes")
        indices = np.empty((query_count, width), dtype=np.int64)
        for query_index in range(query_count):
            order = np.argsort(-masked_scores[query_index], kind="stable")
            indices[query_index] = order[:width]
        selected_scores = np.take_along_axis(masked_scores, indices, axis=1)
    elif spec.routing in {RoutingKind.DENSE, RoutingKind.BLOCK_SPARSE}:
        indices = np.broadcast_to(
            np.arange(source_count, dtype=np.int64),
            (query_count, source_count),
        ).copy()
        selected_scores = masked_scores
    else:
        raise AssertionError(f"unhandled routing: {spec.routing}")

    weights = _normalize(selected_scores, spec.normalization)
    return indices, weights


def execute(
    scores: npt.ArrayLike,
    values: npt.ArrayLike,
    spec: MixerSpec,
    *,
    route_mask: npt.ArrayLike | None = None,
) -> ReferenceResult:
    """Execute gather-score-weighted-reduce semantics.

    `scores` has shape [queries, sources] and `values` has shape
    [sources, value_dim]. Mutation is deliberately handled separately by
    `merge_writes`, preserving snapshot-and-commit semantics.
    """

    score_array = np.asarray(scores, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if score_array.ndim != 2:
        raise ValueError("scores must have shape [queries, sources]")
    if value_array.ndim != 2:
        raise ValueError("values must have shape [sources, value_dim]")
    if score_array.shape[1] != value_array.shape[0]:
        raise ValueError("scores and values disagree on source count")

    bool_mask = None if route_mask is None else np.asarray(route_mask, dtype=np.bool_)
    indices, weights = _route(score_array, spec, bool_mask)
    gathered = value_array[indices]
    output = np.einsum("qk,qkd->qd", weights, gathered)
    return ReferenceResult(output=output, indices=indices, weights=weights)


def merge_writes(
    addresses: npt.ArrayLike,
    deltas: npt.ArrayLike,
    policy: CollisionPolicy,
) -> tuple[IntArray, FloatArray, int]:
    """Deterministically merge a write buffer by address."""

    address_array = np.asarray(addresses, dtype=np.int64)
    delta_array = np.asarray(deltas, dtype=np.float64)
    if address_array.ndim != 1:
        raise ValueError("addresses must have shape [writes]")
    if delta_array.ndim != 2 or delta_array.shape[0] != address_array.shape[0]:
        raise ValueError("deltas must have shape [writes, value_dim]")
    if policy is CollisionPolicy.NOT_APPLICABLE:
        raise ValueError("write merging requires an explicit collision policy")
    if policy is CollisionPolicy.ORDERED:
        raise ValueError("ordered recurrent updates require a dedicated lowering")

    unique_addresses, inverse, counts = np.unique(
        address_array, return_inverse=True, return_counts=True
    )
    collision_count = int(np.sum(counts - 1))
    if policy is CollisionPolicy.REJECT and collision_count:
        raise ValueError("write buffer contains address collisions")

    merged = np.zeros((unique_addresses.size, delta_array.shape[1]), dtype=np.float64)
    if policy in {CollisionPolicy.SUM, CollisionPolicy.MEAN, CollisionPolicy.REJECT}:
        np.add.at(merged, inverse, delta_array)
        if policy is CollisionPolicy.MEAN:
            merged /= counts[:, None]
    elif policy is CollisionPolicy.LAST_WRITE:
        for write_index, address_index in enumerate(inverse):
            merged[address_index] = delta_array[write_index]
    else:
        raise AssertionError(f"unhandled collision policy: {policy}")

    return unique_addresses, merged, collision_count
