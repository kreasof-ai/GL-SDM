"""K2 public-path parity against pinned upstream sources.

These are the Batch-0 verifying clients for the K2 `linear_delta_state` node:
DeltaNet (arch-025) and GLA (arch-019) run through the public graph path
(compile_graph + BoundGraphPlan, reference tier) and are checked against the
pinned FLA source recurrences for both output and final state. The pinned
checkout is provisioned by ``benchmarks/provision_comparators.py``; the tests
skip cleanly when it or torch/fla is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

FLA_PIN = Path("/tmp/urm-comparator-pins/fla")

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_file


def _fla_naive(module_path: str, attr: str):
    if not FLA_PIN.exists():
        pytest.skip("pinned fla checkout is not provisioned")
    import importlib
    import sys

    if str(FLA_PIN) not in sys.path:
        sys.path.insert(0, str(FLA_PIN))
    try:
        module = importlib.import_module(module_path)
    except Exception as error:  # noqa: BLE001 - optional pinned dependency
        pytest.skip(f"pinned fla module unavailable: {error!r}")
    return getattr(module, attr)


def _compile(recipe: str):
    program = normalize_graph_document(load_graph_recipe_file(f"recipes/kernels/{recipe}.json").document)
    return compile_graph(program, target="reference", intent=CompilationIntent.TRAINING)


def test_deltanet_matches_pinned_fla_source():
    delta_rule_recurrence = _fla_naive("fla.ops.delta_rule.naive", "delta_rule_recurrence")
    B, H, T, K, V = 1, 2, 6, 8, 8
    torch.manual_seed(0)
    q = torch.randn(B, H, T, K)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, K), dim=-1)
    v = torch.randn(B, H, T, V)
    beta = torch.rand(B, H, T)
    m0 = torch.zeros(B, H, K, V)

    plan = _compile("deltanet")
    out = plan.execute(
        query=q, key=k, value=v, beta=beta,
        log_decay=torch.zeros(B, H, T), initial_state=m0,
    )
    ref_out, ref_state = delta_rule_recurrence(q, k, v, beta)
    if ref_out.shape != out["output"].shape:
        ref_out = ref_out.transpose(1, 2)
    assert (out["output"] - ref_out).abs().max() < 1e-5
    assert (out["final_state"] - ref_state).abs().max() < 1e-5


def test_gla_matches_pinned_fla_source():
    naive_recurrent_gla = _fla_naive("fla.ops.gla.naive", "naive_recurrent_gla")
    B, H, T, K, V = 1, 2, 6, 8, 8
    torch.manual_seed(0)
    # FLA naive takes [B, T, H, D]; URM binds [B, H, T, D] by role.
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, T, H, K))

    plan = _compile("gla")
    out = plan.execute(
        query=q.transpose(1, 2), key=k.transpose(1, 2), value=v.transpose(1, 2),
        beta=torch.zeros(B, H, T), log_decay=gk.transpose(1, 2),
        initial_state=torch.zeros(B, H, K, V),
    )
    ref_out, ref_h = naive_recurrent_gla(
        q, k, v, gk, initial_state=torch.zeros(B, H, K, V), output_final_state=True
    )
    assert (out["output"].transpose(1, 2) - ref_out).abs().max() < 1e-5
    assert (out["final_state"] - ref_h).abs().max() < 1e-5
