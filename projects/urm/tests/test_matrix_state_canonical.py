"""Native K2 matrix-state kernel vs. the NumPy canonical core.

Each test runs the fused Triton kernel
(:func:`urm.backends.triton.recurrence.matrix_state.execute_matrix_state_recurrence`)
over a short sequence and compares it directly to the NumPy canonical core
:func:`urm.oracles.matrix_state.recurrent`, in fp32. The kernel must mirror the
canonical core's math (fp32 accumulation) to ~1e-4 for every covered option:
delta/additive, dual-gate, retrieval keys, normalizer, feature maps, multi-rank,
and the factored left transition.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for native matrix-state validation", allow_module_level=True
    )

from urm.oracles import matrix_state  # noqa: E402
from urm.backends.triton.recurrence.matrix_state import (  # noqa: E402
    execute_matrix_state_recurrence,
)

# fp32 kernel vs. the float64 canonical core. The kernel accumulates in fp32, so
# the agreement target is ~1e-4 (relative for growing states, absolute for O(1)
# outputs). Inputs are kept well-conditioned (normalized keys, bounded values) so
# the absolute tolerance is meaningful; the normalizer tests use a positive
# (elu_plus_one) feature map so the denominator stays bounded away from zero.
ATOL = 1e-4
RTOL = 1e-4


def _np(x):
    return x.detach().cpu().numpy().astype(np.float64)


def _canonical(memory, keys, queries, values, beta, log_decay, **kwargs):
    """Run the NumPy canonical core over a [B,T,H,*] grid, per (batch, head).

    ``memory`` is [B,H,K,V]; ``keys``/``queries`` are [B,T,H,K]; ``values`` is
    [B,T,H,V]; ``beta``/``log_decay`` follow the head/key-channel conventions.
    Returns (output [B,T,H,V], final [B,H,K,V]) or, with ``normalizer``,
    (output, final, final_norm [B,H,K]).
    """
    normalizer = kwargs.get("normalizer", False)
    batch, sequence, heads, _ = queries.shape
    value_dim = values.shape[-1]
    out = np.empty((batch, sequence, heads, value_dim))
    final = np.empty_like(memory)
    key_dim = memory.shape[2]
    final_norm = np.empty((batch, heads, key_dim)) if normalizer else None
    for b in range(batch):
        for h in range(heads):
            # Slice the per-(batch, head) token operands.
            call = dict(kwargs)
            call.pop("normalizer", None)
            for name in ("erase_gate", "retrieval_keys"):
                if call.get(name) is not None:
                    call[name] = call[name][b, :, h]
            if call.get("write_gate") is not None:
                call["write_gate"] = call["write_gate"][b, :, h]
            if call.get("left_transitions") is not None:
                call["left_transitions"] = call["left_transitions"][b, :, h]
            if call.get("update_keys") is not None:
                call["update_keys"] = call["update_keys"][b, :, :, h]
            if call.get("update_values") is not None:
                call["update_values"] = call["update_values"][b, :, :, h]
            if call.get("rank_beta") is not None:
                call["rank_beta"] = call["rank_beta"][b, :, :, h]
            result = matrix_state.recurrent(
                memory[b, h],
                keys[b, :, h],
                queries[b, :, h],
                values[b, :, h],
                beta[b, :, h],
                log_decay[b, :, h],
                normalizer=normalizer,
                **call,
            )
            if normalizer:
                out[b, :, h], (final[b, h], final_norm[b, h]) = result
            else:
                out[b, :, h], final[b, h] = result
    if normalizer:
        return out, final, final_norm
    return out, final


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _sample(b=2, t=6, h=3, k=8, v=5, seed=0):
    # Keys are L2-normalized (the standard regime for these recurrences) and the
    # values/initial state are bounded, so the state and outputs stay O(1) and a
    # 1e-4 absolute tolerance is a meaningful fp32-vs-fp64 comparison.
    rng = np.random.default_rng(seed)
    data = {
        "memory": rng.normal(size=(b, h, k, v)) * 0.1,
        "keys": _normalize(rng.normal(size=(b, t, h, k))),
        "queries": _normalize(rng.normal(size=(b, t, h, k))),
        "values": rng.normal(size=(b, t, h, v)) * 0.5,
        "beta": rng.uniform(0.1, 0.9, size=(b, t, h)),
        "log_decay": -rng.uniform(0.0, 0.4, size=(b, t, h)),
    }
    return data


def _torch(data, *names):
    return {
        name: torch.from_numpy(data[name]).to("cuda", dtype=torch.float32)
        for name in names
    }


def _assert_close(actual, expected, atol=ATOL, rtol=RTOL):
    actual = actual.detach().cpu().numpy().astype(np.float64)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    err = np.max(np.abs(actual - expected))
    # assert_allclose-style bound: |a - e| <= atol + rtol * |e|.
    assert np.all(np.abs(actual - expected) <= atol + rtol * np.abs(expected)), (
        f"max abs error {err:.3e} (atol={atol:.1e}, rtol={rtol:.1e})"
    )
    return err


def test_plain_delta_matches_canonical():
    data = _sample()
    t = _torch(data, "memory", "keys", "queries", "values", "beta", "log_decay")
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=t["beta"], initial_state=t["memory"],
        scale=1.0, decay_granularity="head", is_delta=True, read_before=False,
    )
    ref_out, ref_final = _canonical(
        data["memory"], data["keys"], data["queries"], data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=True,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_additive_matches_canonical():
    data = _sample(seed=1)
    t = _torch(data, "memory", "keys", "queries", "values", "beta", "log_decay")
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=None, initial_state=t["memory"],
        scale=1.0, decay_granularity="head", is_delta=False, read_before=False,
    )
    ref_out, ref_final = _canonical(
        data["memory"], data["keys"], data["queries"], data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=False,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_key_channel_decay_matches_canonical():
    data = _sample(seed=2)
    b, t_, h, k = data["keys"].shape
    data["log_decay"] = -np.random.default_rng(7).uniform(
        0.0, 0.4, size=(b, t_, h, k)
    )
    t = _torch(data, "memory", "keys", "queries", "values", "beta", "log_decay")
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=t["beta"], initial_state=t["memory"],
        scale=1.0, decay_granularity="key_channel", is_delta=True, read_before=False,
    )
    ref_out, ref_final = _canonical(
        data["memory"], data["keys"], data["queries"], data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=True,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_dual_gate_matches_canonical():
    data = _sample(seed=3)
    b, t_, h, k = data["keys"].shape
    v = data["values"].shape[-1]
    rng = np.random.default_rng(11)
    # gdn2 uses key-channel decay; erase [B,T,H,K], write [B,T,H,V].
    data["log_decay"] = -rng.uniform(0.0, 0.4, size=(b, t_, h, k))
    erase = rng.uniform(0.1, 0.9, size=(b, t_, h, k))
    write = rng.uniform(0.1, 0.9, size=(b, t_, h, v))
    t = _torch(data, "memory", "keys", "queries", "values", "log_decay")
    erase_t = torch.from_numpy(erase).to("cuda", dtype=torch.float32)
    write_t = torch.from_numpy(write).to("cuda", dtype=torch.float32)
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=None, initial_state=t["memory"],
        scale=1.0, decay_granularity="key_channel", is_delta=False,
        read_before=False, erase_gate=erase_t, write_gate=write_t,
    )
    ref_out, ref_final = _canonical(
        data["memory"], data["keys"], data["queries"], data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=False,
        erase_gate=erase, write_gate=write,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_retrieval_keys_match_canonical():
    data = _sample(seed=4)
    b, t_, h, k = data["keys"].shape
    rng = np.random.default_rng(13)
    retr = rng.normal(size=(b, t_, h, k))
    t = _torch(data, "memory", "keys", "queries", "values", "beta", "log_decay")
    retr_t = torch.from_numpy(retr).to("cuda", dtype=torch.float32)
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=t["beta"], initial_state=t["memory"],
        scale=1.0, decay_granularity="head", is_delta=True, read_before=False,
        retrieval_keys=retr_t,
    )
    ref_out, ref_final = _canonical(
        data["memory"], data["keys"], data["queries"], data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=True,
        retrieval_keys=retr,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_normalizer_delta_head_decay_matches_canonical():
    # Query/key normalizer with the delta rule and head decay. The elu_plus_one
    # feature map keeps the keys/queries positive so the denominator q^T z stays
    # bounded away from zero (the realistic linear-attention regime).
    data = _sample(seed=5)
    t = _torch(data, "memory", "keys", "queries", "values", "beta", "log_decay")
    out, final, final_norm = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=t["beta"], initial_state=t["memory"],
        scale=1.0, decay_granularity="head", is_delta=True, read_before=False,
        normalizer=True, epsilon=1e-6, feature_map="elu_plus_one",
    )
    from urm.oracles.composition import _feature_map
    from urm.ir.mixer import FeatureMap
    kf = _feature_map(data["keys"], FeatureMap.ELU_PLUS_ONE)
    qf = _feature_map(data["queries"], FeatureMap.ELU_PLUS_ONE)
    ref_out, ref_final, ref_norm = _canonical(
        data["memory"], kf, qf, data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=True,
        normalizer=True, epsilon=1e-6,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)
    _assert_close(final_norm, ref_norm)


def test_normalizer_additive_no_decay_matches_canonical():
    # The linear-attention form: additive update, no decay, query/key normalizer.
    data = _sample(seed=6)
    t = _torch(data, "memory", "keys", "queries", "values")
    out, final, final_norm = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=None, beta=None, initial_state=t["memory"],
        scale=1.0, decay_granularity="none", is_delta=False, read_before=False,
        normalizer=True, epsilon=1e-6, feature_map="elu_plus_one",
    )
    from urm.oracles.composition import _feature_map
    from urm.ir.mixer import FeatureMap
    kf = _feature_map(data["keys"], FeatureMap.ELU_PLUS_ONE)
    qf = _feature_map(data["queries"], FeatureMap.ELU_PLUS_ONE)
    zeros = np.zeros((data["keys"].shape[0], data["keys"].shape[1], data["keys"].shape[2]))
    ref_out, ref_final, ref_norm = _canonical(
        data["memory"], kf, qf, data["values"],
        data["beta"], zeros, scale=1.0, is_delta=False,
        normalizer=True, epsilon=1e-6,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)
    _assert_close(final_norm, ref_norm)


@pytest.mark.parametrize(
    "feature_map", ["l2_normalize", "relu", "elu_plus_one"]
)
def test_feature_map_matches_canonical(feature_map):
    data = _sample(seed=7)
    t = _torch(data, "memory", "keys", "queries", "values", "beta", "log_decay")
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=t["beta"], initial_state=t["memory"],
        scale=1.0, decay_granularity="head", is_delta=True, read_before=False,
        feature_map=feature_map,
    )
    # The canonical core receives the feature-mapped query/key.
    from urm.oracles.composition import _feature_map
    from urm.ir.mixer import FeatureMap
    kind = FeatureMap(feature_map)
    kf = _feature_map(data["keys"], kind)
    qf = _feature_map(data["queries"], kind)
    ref_out, ref_final = _canonical(
        data["memory"], kf, qf, data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=True,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_multi_rank_matches_canonical():
    data = _sample(seed=8)
    b, t_, h, k = data["keys"].shape
    v = data["values"].shape[-1]
    r = 3
    rng = np.random.default_rng(17)
    update_keys = _normalize(rng.normal(size=(b, t_, r, h, k)))
    update_values = rng.normal(size=(b, t_, r, h, v)) * 0.5
    rank_beta = rng.uniform(0.1, 0.9, size=(b, t_, r, h))
    t = _torch(data, "memory", "keys", "queries", "values", "log_decay")
    uk_t = torch.from_numpy(update_keys).to("cuda", dtype=torch.float32)
    uv_t = torch.from_numpy(update_values).to("cuda", dtype=torch.float32)
    rb_t = torch.from_numpy(rank_beta).to("cuda", dtype=torch.float32)
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=None, initial_state=t["memory"],
        scale=1.0, decay_granularity="head", is_delta=False, read_before=False,
        update_keys=uk_t, update_values=uv_t, rank_beta=rb_t,
    )
    ref_out, ref_final = _canonical(
        data["memory"], data["keys"], data["queries"], data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=False,
        update_keys=update_keys, update_values=update_values, rank_beta=rank_beta,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_left_transitions_match_canonical():
    # Generalized-delta DPLR-style factored transition:
    # left_t = diag(exp(log_decay)) + beta_t ⊗ alpha_t, applied as Z = left_t @ M.
    # The diagonal decay keeps the transition contractive so the state stays O(1).
    # Use K >= 16 so the kernel's tl.dot block is fully dense.
    b, t_, h, k, v = 2, 5, 2, 16, 16
    rng = np.random.default_rng(19)
    memory = rng.normal(size=(b, h, k, v)) * 0.1
    keys = _normalize(rng.normal(size=(b, t_, h, k)))
    queries = _normalize(rng.normal(size=(b, t_, h, k)))
    values = rng.normal(size=(b, t_, h, v)) * 0.5
    beta = rng.uniform(0.1, 0.9, size=(b, t_, h))
    alpha = _normalize(rng.normal(size=(b, t_, h, k)))
    tb = _normalize(rng.normal(size=(b, t_, h, k))) * 0.5
    diag = np.exp(-rng.uniform(0.1, 0.5, size=(b, t_, h, k)))
    left = np.empty((b, t_, h, k, k))
    for bi in range(b):
        for ti in range(t_):
            for hi in range(h):
                left[bi, ti, hi] = np.diag(diag[bi, ti, hi]) + np.outer(
                    tb[bi, ti, hi], alpha[bi, ti, hi]
                )
    zeros = np.zeros((b, t_, h))
    tt = lambda x: torch.from_numpy(x).to("cuda", dtype=torch.float32)
    out, final = execute_matrix_state_recurrence(
        query=tt(queries), key=tt(keys), value=tt(values),
        log_decay=None, beta=tt(beta), initial_state=tt(memory),
        scale=1.0, decay_granularity="none", is_delta=True, read_before=False,
        left_transitions=tt(left),
    )
    ref_out, ref_final = _canonical(
        memory, keys, queries, values, beta, zeros, scale=1.0, is_delta=True,
        left_transitions=left,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)


def test_polynomial_basis_via_pre_expansion_matches_canonical():
    # The polynomial quadratic bases expand the feature dimension; the caller
    # pre-expands the operands (matching the canonical core), so the kernel sees
    # the expanded width. Validate the additive + normalizer path on the
    # pre-expanded based_taylor2 features.
    from urm.oracles.composition import _polynomial_features
    from urm.ir.mixer import PolynomialBasis
    data = _sample(seed=9, k=6, v=4)
    key_dim = data["keys"].shape[-1]
    poly_scale = key_dim ** -0.5
    qf = _polynomial_features(
        data["queries"], PolynomialBasis.BASED_TAYLOR2, poly_scale, is_query=True
    )
    kf = _polynomial_features(
        data["keys"], PolynomialBasis.BASED_TAYLOR2, poly_scale, is_query=False
    )
    feat = kf.shape[-1]
    b, t_, h = data["keys"].shape[:3]
    v = data["values"].shape[-1]
    memory = np.random.default_rng(21).normal(size=(b, h, feat, v)) * 0.1
    tt = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(
        "cuda", dtype=torch.float32
    )
    out, final, final_norm = execute_matrix_state_recurrence(
        query=tt(qf), key=tt(kf), value=tt(data["values"]),
        log_decay=None, beta=None, initial_state=tt(memory),
        scale=1.0, decay_granularity="none", is_delta=False, read_before=False,
        normalizer=True, epsilon=1e-6,
    )
    zeros = np.zeros((b, t_, h))
    ref_out, ref_final, ref_norm = _canonical(
        memory, kf, qf, data["values"], data["beta"], zeros,
        scale=1.0, is_delta=False, normalizer=True, epsilon=1e-6,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)
    _assert_close(final_norm, ref_norm)


def test_read_before_update_matches_canonical():
    data = _sample(seed=10)
    t = _torch(data, "memory", "keys", "queries", "values", "beta", "log_decay")
    out, final = execute_matrix_state_recurrence(
        query=t["queries"], key=t["keys"], value=t["values"],
        log_decay=t["log_decay"], beta=t["beta"], initial_state=t["memory"],
        scale=1.0, decay_granularity="head", is_delta=True, read_before=True,
    )
    ref_out, ref_final = _canonical(
        data["memory"], data["keys"], data["queries"], data["values"],
        data["beta"], data["log_decay"], scale=1.0, is_delta=True,
        read_before_update=True,
    )
    _assert_close(out, ref_out)
    _assert_close(final, ref_final)
