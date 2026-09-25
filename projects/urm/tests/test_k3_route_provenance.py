"""Gate 2: K3 route-provenance + read-timing hardening on the public graph path.

The K3 contract certifies that routes are "already certified logical addresses and
normalized weights". On the public graph path (``BoundGraphPlan`` binds raw tensors by
role), that certification is the reference provider's job: the route operands are
validated against the spec's bounds — partition-local in-bounds addresses, strictly
increasing and unique within each token, the declared route width, and finite
nonnegative normalized weights — before the equation runs. Malformed routes previously
hit a raw ``IndexError`` (out-of-bounds) or passed silently (non-normalized / duplicate
addresses); they now produce a structured decline. Read timing is honored end-to-end
(before-update reads the pre-update state; after-update the post-update state).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from urm.backends.contract import ProviderFamily, ProviderRequest
from urm.backends.torch.k3 import K3TorchReferenceProvider, validate_k3_route_provenance
from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document
from urm.ir.program import (
    DType,
    SparseReadTiming,
    SparseStateMixerSpec,
    SparseStateOperation,
)

P, T, S, D, W, R = 1, 2, 16, 4, 2, 2


def _spec(timing=SparseReadTiming.AFTER_UPDATE, operation=SparseStateOperation.UPDATE):
    return SparseStateMixerSpec(
        parallel=P, sequence=T, slots_per_partition=S, value_dim=D, writes=W, reads=R,
        dtype=DType.FLOAT32, operation=operation, read_timing=timing,
    )


def _valid_ops():
    torch.manual_seed(5)
    return {
        "memory": torch.randn(P, S, D),
        "read_addresses": torch.tensor([[[1, 3], [0, 2]]]),
        "read_weights": torch.tensor([[[0.5, 0.5], [0.25, 0.75]]]),
        "write_addresses": torch.tensor([[[0, 2], [1, 3]]]),
        "write_weights": torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]),
        "values": torch.randn(P, T, D),
        "beta": torch.rand(P, T, 1),
        "log_decay": -torch.rand(P, T, 1),
    }


def test_valid_routes_pass_provenance():
    assert validate_k3_route_provenance(_spec(), _valid_ops()) is None


@pytest.mark.parametrize("label,mutate,needle", [
    ("oob_read", lambda o: o.update(read_addresses=torch.tensor([[[1, 99], [0, 2]]])), "in bounds"),
    ("oob_write", lambda o: o.update(write_addresses=torch.tensor([[[0, 16], [1, 3]]])), "in bounds"),
    ("duplicate", lambda o: o.update(read_addresses=torch.tensor([[[1, 1], [0, 2]]])), "strictly increasing"),
    ("non_normalized", lambda o: o.update(read_weights=torch.tensor([[[9.0, 9.0], [0.5, 0.5]]])), "normalized"),
    ("negative", lambda o: o.update(read_weights=torch.tensor([[[1.5, -0.5], [0.5, 0.5]]])), "nonnegative"),
    ("wrong_width", lambda o: o.update(read_addresses=torch.tensor([[[1, 3, 5], [0, 2, 4]]]),
                                       read_weights=torch.tensor([[[0.3, 0.3, 0.4], [0.5, 0.5, 0.0]]])), "parallel, sequence, 2"),
])
def test_malformed_routes_decline(label, mutate, needle):
    ops = _valid_ops()
    mutate(ops)
    reason = validate_k3_route_provenance(_spec(), ops)
    assert reason is not None, f"{label} should be declined"
    assert needle in reason, f"{label}: expected {needle!r} in {reason!r}"


def test_provider_execute_rejects_malformed_routes():
    """The reference provider's execute runs the provenance gate before the equation."""
    provider = K3TorchReferenceProvider()
    ops = _valid_ops()
    ops["read_addresses"] = torch.tensor([[[1, 99], [0, 2]]])  # out of bounds
    req = ProviderRequest(family=ProviderFamily.K3, descriptor=_spec(), mode="inference")
    assert provider.decline(req) is None
    with pytest.raises(ValueError, match="in bounds"):
        provider.execute(req, ops)


def _mixer_doc(timing):
    return {
        "schema_version": 2, "name": "k3m", "kind": "kernel_fragment",
        "graph": {
            "inputs": [
                {"name": "read_addresses", "dtype": "int64", "shape": ["P", "T", "R"]},
                {"name": "read_weights", "dtype": "float32", "shape": ["P", "T", "R"]},
                {"name": "write_addresses", "dtype": "int64", "shape": ["P", "T", "W"]},
                {"name": "write_weights", "dtype": "float32", "shape": ["P", "T", "W"]},
                {"name": "values", "dtype": "float32", "shape": ["P", "T", "D"]},
                {"name": "beta", "dtype": "float32", "shape": ["P", "T", "1"]},
                {"name": "log_decay", "dtype": "float32", "shape": ["P", "T", "1"]},
                {"name": "memory", "dtype": "float32", "shape": ["P", "S", "D"]},
            ],
            "nodes": [{
                "id": "m", "op": "sparse_state_mixer",
                "inputs": ["memory", "read_addresses", "read_weights", "write_addresses",
                           "write_weights", "values", "beta", "log_decay"],
                "outputs": ["output", "final_state"],
                "params": {
                    "slots_per_partition": S, "value_dim": D, "writes": W, "reads": R,
                    "operation": "update", "read_timing": timing,
                    "update_rule": "decayed_delta", "collision_policy": "ordered",
                    "mode": "inference",
                },
            }],
            "outputs": ["output", "final_state"],
        },
    }


def _compile(timing):
    return compile_graph(
        normalize_graph_document(load_graph_recipe_document(_mixer_doc(timing)).document),
        target="reference", intent=CompilationIntent.INFERENCE,
    )


def test_public_path_enforces_provenance():
    plan = _compile("after_update")
    ops = _valid_ops()
    plan.execute(**ops)  # valid routes execute
    bad = _valid_ops()
    bad["read_weights"] = torch.tensor([[[9.0, 9.0], [0.5, 0.5]]])  # non-normalized
    with pytest.raises(ValueError, match="normalized"):
        plan.execute(**bad)


def test_read_timing_honored_end_to_end():
    """before_update reads the pre-update state; after_update the post-update state."""
    torch.manual_seed(7)
    ops = {
        "memory": torch.randn(P, 3, D) if False else torch.randn(P, S, D),
        "read_addresses": torch.tensor([[[0, 1]] * 3]),
        "read_weights": torch.tensor([[[0.5, 0.5]] * 3]),
        "write_addresses": torch.tensor([[[0, 1]] * 3]),
        "write_weights": torch.tensor([[[0.5, 0.5]] * 3]),
        "values": torch.randn(P, 3, D),
        "beta": torch.rand(P, 3, 1) * 0.9,
        "log_decay": -torch.rand(P, 3, 1) * 0.1,
    }
    # Rebuild docs with T=3 for this read/write shape.
    def doc3(timing):
        d = _mixer_doc(timing)
        for inp in d["graph"]["inputs"]:
            if inp["name"] in {"read_addresses", "read_weights", "write_addresses", "write_weights",
                               "values", "beta", "log_decay"}:
                inp["shape"] = ["P", "T3", *inp["shape"][2:]]
        return d
    oa = compile_graph(
        normalize_graph_document(load_graph_recipe_document(doc3("after_update")).document),
        target="reference", intent=CompilationIntent.INFERENCE,
    ).execute(**{k: v.clone() for k, v in ops.items()})["output"]
    ob = compile_graph(
        normalize_graph_document(load_graph_recipe_document(doc3("before_update")).document),
        target="reference", intent=CompilationIntent.INFERENCE,
    ).execute(**{k: v.clone() for k, v in ops.items()})["output"]
    # The two timings must differ (the update lands between them).
    assert (oa - ob).abs().max().item() > 1e-3
    # before_update at t=0 reads the initial memory, before any write lands.
    expected0 = 0.5 * ops["memory"][:, 0] + 0.5 * ops["memory"][:, 1]
    assert torch.allclose(ob[:, 0], expected0.float(), atol=1e-5)
