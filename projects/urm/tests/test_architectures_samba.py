"""Parity gates for arch-053 Samba attention branch.

Verified against the pinned Samba source (lit_gpt/model.py CausalSelfAttention
@ 617c7a0f, the sweep's verification origin) via the AST-extracted class: the
batched GQA QKV projection + RoPE + causal U1.S mixer composition matches on
identical parameters, in both MHA and GQA query-group configurations. The
Mamba-selecting configurations (use_mamba, mamba_swa_mlp) are gated on the A8
axis and not exercised here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import architectures.samba_attention as urm_samba
from architectures.samba_attention import SambaAttentionLayer
from benchmarks.comparators import samba as samba_comparator


def _pinned_class():
    cls, identity = samba_comparator.load_samba_attention()
    # The pinned class calls apply_rotary_emb_func (a fused rotary extension)
    # from its module globals; the AST extraction leaves that name unbound.
    # Inject the non-interleaved rotary equivalent (the same function the URM
    # module transcribes) so the pinned class runs its RoPE path.
    import sys
    module = sys.modules[cls.__module__]
    module.apply_rotary_emb_func = (
        lambda q, cos, sin, interleaved, fused: urm_samba.apply_rotary_emb(q, cos, sin)
    )
    return cls, identity


def _config(n_head=4, n_query_groups=4, head_size=8):
    n_embd = n_head * head_size
    return SimpleNamespace(
        full_per_layer=10**9,  # layer_idx 0 -> not local (full attention)
        head_size=head_size,
        n_head=n_head,
        n_query_groups=n_query_groups,
        bias=False,
        sc_attn=False,
        nope=False,
        local_window=-1,
    ), n_embd


def _build_pair(seed: int, n_query_groups: int):
    config, n_embd = _config(n_query_groups=n_query_groups)
    cls, _id = _pinned_class()
    torch.manual_seed(seed)
    pinned = cls(config, layer_idx=0, n_embd=n_embd)
    urm = SambaAttentionLayer(n_embd, config.n_head, config.head_size, n_query_groups)
    with torch.no_grad():
        urm.attn.weight.copy_(pinned.attn.weight)
        urm.proj.weight.copy_(pinned.proj.weight)
    x = torch.randn(2, 6, n_embd)
    return pinned, urm, x, config


def _run_parity(seed: int, n_query_groups: int):
    pinned, urm, x, config = _build_pair(seed, n_query_groups)
    # Inject the pinned module's RoPE dependencies (fp32 cache + the same
    # non-interleaved rotary function the URM module transcribes).
    rope = urm_samba.build_rope_cache(
        x.shape[1], config.head_size, x.dtype, x.device
    )
    with torch.no_grad():
        expected, _ = pinned(x, rope, max_seq_length=x.shape[1])
        actual = urm(x)
    return (actual - expected).abs().max().item()


def test_attention_branch_matches_pinned_mha():
    err = _run_parity(seed=5, n_query_groups=4)
    assert err < 1e-5, f"mha parity: max abs err {err}"


def test_attention_branch_matches_pinned_gqa():
    err = _run_parity(seed=7, n_query_groups=2)
    assert err < 1e-5, f"gqa parity: max abs err {err}"


def test_gradients_flow_through_qkv_and_output_projections():
    _pinned, urm, x, _cfg = _build_pair(seed=13, n_query_groups=2)
    urm(x).square().sum().backward()
    assert urm.attn.weight.grad is not None and urm.attn.weight.grad.abs().sum().item() > 0
    assert urm.proj.weight.grad is not None and urm.proj.weight.grad.abs().sum().item() > 0
