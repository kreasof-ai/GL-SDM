"""The provider contract is uniform across every backend tier.

This is the regression guard for "backends share one interface": every
provider — reference Torch, native Triton, library SDPA, and the independent
NumPy oracle tier — is a :class:`~urm.backends.contract.Provider` with
``decline(request) -> str | None`` then ``execute(request, role-bound
operands)``. A future backend implements exactly that surface; nothing else
enters the runtime dispatch table. The NumPy oracle tier is cross-checked
against the Torch reference tier per family.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from urm.backends.contract import ProviderFamily, ProviderRequest
from urm.backends.numpy.k1 import K1NumpyProvider
from urm.backends.numpy.k2 import K2NumpyProvider
from urm.backends.numpy.k3 import K3NumpyProvider
from urm.runtime.bind import _PROVIDERS
from urm.ir.program import (
    DType,
    K1Descriptor,
    K2GateScope,
    LinearDeltaSpec,
    SparseReadTiming,
    SparseStateMixerSpec,
    SparseStateOperation,
)


def test_every_dispatch_entry_is_a_uniform_provider():
    for name, provider in _PROVIDERS.items():
        assert provider.name == name
        assert provider.family in {
            ProviderFamily.K1,
            ProviderFamily.K2,
            ProviderFamily.K3,
            ProviderFamily.K3_ROUTE,
            ProviderFamily.MERGE,
        }
        assert provider.tier in {"reference", "native", "library"}
        assert callable(provider.decline)
        assert callable(provider.execute)
        # Uniform structured decline on a wrong-typed descriptor.
        wrong = ProviderRequest(family=provider.family, descriptor=object(), mode="inference")
        assert provider.decline(wrong) is not None


def test_no_two_tiers_share_one_anchor_name():
    names = [p.name for p in _PROVIDERS.values()]
    assert len(names) == len(set(names)), "an anchor name maps to two providers"


def test_numpy_and_torch_k1_agree_through_the_contract():
    rng = np.random.default_rng(0)
    B, H, T, D = 2, 3, 8, 16
    q = rng.normal(size=(B, T, H, D))
    k = rng.normal(size=(B, T, H, D))
    v = rng.normal(size=(B, T, H, D))
    req = ProviderRequest(
        family=ProviderFamily.K1, descriptor=K1Descriptor(causal=True), mode="inference"
    )
    np_out = K1NumpyProvider().execute(req, {"query": q, "key": k, "value": v})["output"]
    from urm.backends.torch.k1 import torch_k1_softmax_attention

    t_out = torch_k1_softmax_attention(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), descriptor=K1Descriptor(causal=True)
    ).numpy()
    assert np.abs(np_out - t_out).max() < 1e-5


def test_numpy_and_torch_k2_agree_through_the_contract():
    rng = np.random.default_rng(0)
    T, K, V = 6, 4, 3
    m0 = rng.normal(size=(1, 2, K, V))
    k2 = rng.normal(size=(1, 2, T, K))
    q2 = rng.normal(size=(1, 2, T, K))
    v2 = rng.normal(size=(1, 2, T, V))
    b2 = rng.random(size=(1, 2, T))
    g2 = -rng.random(size=(1, 2, T))
    spec = LinearDeltaSpec(delta=True, gate_scope=K2GateScope.HEAD)
    req = ProviderRequest(family=ProviderFamily.K2, descriptor=spec, mode="training")
    np_out = K2NumpyProvider().execute(
        req, {"query": q2, "key": k2, "value": v2, "beta": b2, "log_decay": g2, "initial_state": m0}
    )
    from urm.backends.torch.k2 import torch_linear_delta_state

    t_out, t_state = torch_linear_delta_state(
        torch.tensor(m0), torch.tensor(k2), torch.tensor(q2), torch.tensor(v2),
        torch.tensor(b2), torch.tensor(g2), spec=spec,
    )
    assert np.abs(np_out["output"] - t_out.numpy()).max() < 1e-4
    assert np.abs(np_out["final_state"] - t_state.numpy()).max() < 1e-4


def test_numpy_and_torch_k3_share_address_operand_form():
    rng = np.random.default_rng(0)
    P, T, S, D, W = 1, 4, 16, 3, 2
    mem = rng.normal(size=(P, S, D))
    read_idx = np.stack([np.sort(rng.choice(S, W, replace=False)) for _ in range(P * T)]).reshape(P, T, W)
    write_idx = np.stack([np.sort(rng.choice(S, W, replace=False)) for _ in range(P * T)]).reshape(P, T, W)
    rw = np.exp(rng.normal(size=(P, T, W))); rw /= rw.sum(-1, keepdims=True)
    ww = np.exp(rng.normal(size=(P, T, W))); ww /= ww.sum(-1, keepdims=True)
    vals = rng.normal(size=(P, T, D))
    beta = rng.random(size=(P, T, 1)); ld = -rng.random(size=(P, T, 1))
    spec = SparseStateMixerSpec(
        parallel=P, sequence=T, slots_per_partition=S, value_dim=D, writes=W, reads=W,
        dtype=DType.FLOAT32, operation=SparseStateOperation.UPDATE,
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    req = ProviderRequest(family=ProviderFamily.K3, descriptor=spec, mode="inference")
    operands = {"memory": mem, "read_addresses": read_idx, "read_weights": rw,
                "write_addresses": write_idx, "write_weights": ww, "values": vals,
                "beta": beta, "log_decay": ld}
    np_out = K3NumpyProvider().execute(req, operands)
    from urm.backends.torch.k3 import torch_sparse_state_mixer

    t_out, t_state = torch_sparse_state_mixer(
        torch.tensor(mem), torch.tensor(read_idx), torch.tensor(rw),
        write_indices=torch.tensor(write_idx), write_weights=torch.tensor(ww),
        values=torch.tensor(vals), beta=torch.tensor(beta), log_decay=torch.tensor(ld),
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    assert np.abs(np_out["readings"] - t_out.numpy()).max() < 1e-4
    assert np.abs(np_out["updated_memory"] - t_state.numpy()).max() < 1e-4
