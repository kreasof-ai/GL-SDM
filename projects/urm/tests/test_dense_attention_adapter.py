from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for adapter tests", allow_module_level=True)

from urm.adapters import UrmDenseCausalAttentionAdapter
from urm.adapters.dense_attention import (
    DenseAttentionSpec,
    flash_attn_version,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _sample(b=1, h=4, kv=None, s=64, d=32, dtype=torch.bfloat16, device="cuda"):
    kv = kv or h
    generator = torch.Generator(device=device).manual_seed(3)
    q = torch.randn((b, h, s, d), device=device, dtype=dtype, generator=generator)
    k = torch.randn((b, kv, s, d), device=device, dtype=dtype, generator=generator)
    v = torch.randn((b, kv, s, d), device=device, dtype=dtype, generator=generator)
    return q, k, v


def test_spec_rejects_bad_shapes_and_dtypes() -> None:
    q, k, v = _sample()
    DenseAttentionSpec.from_tensors(q, k, v)

    with pytest.raises(ValueError, match="rank-4"):
        DenseAttentionSpec.from_tensors(q[0], k[0], v[0])
    with pytest.raises(ValueError, match="GQA"):
        DenseAttentionSpec.from_tensors(q, k[:, :3], v[:, :3])
    # A divisible grouping is valid and must not raise.
    DenseAttentionSpec.from_tensors(q, k[:, :2], v[:, :2])
    with pytest.raises(ValueError, match="fp16/bf16"):
        DenseAttentionSpec.from_tensors(q.float(), k.float(), v.float(), causal=True)
    with pytest.raises(ValueError, match="device"):
        DenseAttentionSpec.from_tensors(q.cpu(), k.cpu(), v.cuda())


def test_adapter_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown adapter backend"):
        UrmDenseCausalAttentionAdapter("cutlass")


def test_flash_attn_identity_recorded_or_not_applicable() -> None:
    identity = flash_attn_version()
    if identity.get("package") == "flash-attn":
        assert identity["version"]
        assert identity["pin"]
    else:
        assert identity["status"] == "not_applicable"


def test_adapter_matches_direct_upstream_bitwise_and_records_evidence() -> None:
    identity = flash_attn_version()
    if identity.get("package") != "flash-attn":
        pytest.skip("flash-attn not installed")
    adapter = UrmDenseCausalAttentionAdapter("flash_attn")
    q, k, v = _sample(s=128)
    expected = adapter._flash_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=True,
        softmax_scale=q.shape[-1] ** -0.5,
    ).transpose(1, 2)
    actual, info = adapter.execute(q, k, v, return_info=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert info["adapter_backend"] == "flash_attn"
    assert info["spec"]["causal"] is True
    assert info["upstream"]["package"] == "flash-attn"


def test_adapter_sdpa_flash_output_close_to_direct_fa() -> None:
    if flash_attn_version().get("package") != "flash-attn":
        pytest.skip("flash-attn not installed")
    adapter = UrmDenseCausalAttentionAdapter("sdpa_flash")
    q, k, v = _sample(h=8, s=256)
    direct = UrmDenseCausalAttentionAdapter("flash_attn").execute(q, k, v)
    via_adapter = adapter.execute(q, k, v)
    torch.testing.assert_close(
        via_adapter.float(), direct.float(), atol=2e-2, rtol=2e-2
    )


def _oracle(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    import math

    scale = 1.0 / math.sqrt(q.shape[-1])
    if k.shape[1] != q.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    mask = torch.ones(
        q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device
    ).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.to(v.dtype), v)


def test_gqa_forward_backward_match_oracle() -> None:
    adapter = UrmDenseCausalAttentionAdapter(
        "flash_attn"
        if flash_attn_version().get("package") == "flash-attn"
        else "sdpa_flash"
    )
    q, k, v = _sample(b=1, h=8, kv=2, s=96, d=32, dtype=torch.float16)
    expected = _oracle(q, k, v)
    out = adapter.execute(q, k, v)
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)

    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)
    grad = torch.randn_like(out)
    adapter.execute(q_g, k_g, v_g).backward(grad)
    for tensor in (q_g, k_g, v_g):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_committed_attention_artifacts_validate_against_schema() -> None:
    schema_path = PROJECT_ROOT / "benchmarks" / "attention-result-schema.json"
    artifact_path = PROJECT_ROOT / "results" / "attention" / "dense-causal.json"
    if not artifact_path.exists():
        pytest.skip("attention comparator has not been run yet")
    from jsonschema import validate

    validate(json.loads(artifact_path.read_text()), json.loads(schema_path.read_text()))
