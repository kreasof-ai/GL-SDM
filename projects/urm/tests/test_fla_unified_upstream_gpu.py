"""Direct parity checks against the FLA revision in the coverage register."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
fla = pytest.importorskip("fla")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for pinned FLA parity", allow_module_level=True)

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"


def _loaded_fla_revision() -> str | None:
    root = Path(fla.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.mark.parametrize("recipe_name", ["simple_gla", "gla"])
def test_compiled_fla_gated_additive_matches_direct_pinned_upstream(recipe_name):
    """Compare the unified FLA path with a direct call to the pinned operator."""
    revision = _loaded_fla_revision()
    if revision != EXPECTED_FLA_REVISION:
        pytest.skip(
            "set PYTHONPATH to the coverage-register FLA checkout "
            f"{EXPECTED_FLA_REVISION}; loaded revision is {revision!r}"
        )
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    assert identity["revision_compatible"] is True
    assert identity["comparison_compatible"] is True

    from fla.ops.gla import chunk_gla
    from fla.ops.simple_gla import chunk_simple_gla
    from urm.compiler.unified_mixer import (
        MixerBackend,
        MixerIntent,
        compile_mixer,
    )
    from urm.frontend.mixer_recipes import named_mixer_recipe

    torch.manual_seed(44018 if recipe_name == "simple_gla" else 44019)
    query = torch.randn(1, 24, 2, 16, device="cuda", dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn(1, 24, 2, 12, device="cuda", dtype=torch.float32)
    gate_shape = (1, 24, 2) if recipe_name == "simple_gla" else (1, 24, 2, 16)
    log_decay = -torch.rand(gate_shape, device="cuda", dtype=torch.float32) * 0.25
    initial_state = torch.randn(1, 2, 16, 12, device="cuda", dtype=torch.float32)

    operands = {
        "query": query.detach().requires_grad_(),
        "key": key.detach().requires_grad_(),
        "value": value.detach().requires_grad_(),
        "log_decay": log_decay.detach().requires_grad_(),
        "initial_state": initial_state.detach().requires_grad_(),
    }
    plan = compile_mixer(
        named_mixer_recipe(recipe_name),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    compiled = plan.execute(**operands)

    direct_operands = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    direct_fn = chunk_simple_gla if recipe_name == "simple_gla" else chunk_gla
    direct_output, direct_state = direct_fn(
        direct_operands["query"].contiguous(),
        direct_operands["key"].contiguous(),
        direct_operands["value"].contiguous(),
        g=direct_operands["log_decay"].contiguous(),
        scale=1.0,
        initial_state=direct_operands["initial_state"],
        output_final_state=True,
    )

    torch.testing.assert_close(compiled.output, direct_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(compiled.final_state, direct_state, atol=1e-6, rtol=1e-6)
    compiled_loss = (
        compiled.output.square().mean() + compiled.final_state.square().mean()
    )
    direct_loss = direct_output.square().mean() + direct_state.square().mean()
    compiled_grads = torch.autograd.grad(compiled_loss, tuple(operands.values()))
    direct_grads = torch.autograd.grad(direct_loss, tuple(direct_operands.values()))
    for compiled_grad, direct_grad in zip(compiled_grads, direct_grads, strict=True):
        torch.testing.assert_close(compiled_grad, direct_grad, atol=2e-6, rtol=2e-6)
