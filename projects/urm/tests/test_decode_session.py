"""Decode-session correctness: the single-token decode-step kernels must match
the sequential recurrence composed over T tokens, for every native family.

This pins the "how you use the kernel" contract: the decode path runs a fused
single-token kernel against a persistent state (in place, no autograd, no
per-step host work), and the composed result must equal the training-oriented
sequence scan. A divergence here is a correctness bug in the decode path, not a
performance detail.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for decode-session validation", allow_module_level=True)


def _compose_matrix_state_decode(T=8, B=3, H=4, K=32, V=32, dtype=torch.float32):
    from urm.backends.triton.k2.matrix import (
        execute_matrix_state_decode_step,
        execute_matrix_state_recurrence,
    )

    torch.manual_seed(0)
    state0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.1
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda", dtype=dtype).float(), dim=-1
    ).to(dtype)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g = (-torch.rand(B, T, H, device="cuda", dtype=dtype) * 0.3)
    beta = torch.rand(B, T, H, device="cuda", dtype=dtype)
    ref_out, ref_state = execute_matrix_state_recurrence(
        query=q, key=k, value=v, log_decay=g, beta=beta, initial_state=state0,
        scale=1.0, decay_granularity="head", is_delta=True, read_before=False,
    )
    state = state0.clone()
    outs = []
    with torch.no_grad():
        for t in range(T):
            outs.append(
                execute_matrix_state_decode_step(
                    query=q[:, t].float(), key=k[:, t].float(), value=v[:, t].float(),
                    log_decay=g[:, t].float(), beta=beta[:, t].float(), state=state,
                    scale=1.0, decay_granularity="head", is_delta=True, read_before=False,
                )
            )
    step_out = torch.stack(outs, dim=1)
    return step_out, ref_out, state, ref_state


def test_matrix_state_decode_step_matches_sequential():
    step_out, ref_out, state, ref_state = _compose_matrix_state_decode()
    # fp32 inputs: the decode-step kernel must match the sequential scan closely.
    assert (state - ref_state).abs().max().item() < 1e-5
    assert (step_out - ref_out.float()).abs().max().item() < 1e-4


def test_diagonal_decode_step_matches_sequential():
    from urm.backends.triton.k2.diagonal import (
        execute_diagonal_decode_step,
        execute_diagonal_recurrence,
    )

    B, C, N, T = 3, 64, 1, 8
    torch.manual_seed(0)
    state0 = torch.randn(B, C, N, device="cuda", dtype=torch.float32) * 0.1
    x = torch.randn(B, T, C, device="cuda", dtype=torch.float32) * 0.2
    ld = (-torch.rand(B, T, C, device="cuda", dtype=torch.float32) * 0.05)
    ld_e = ld.unsqueeze(-1)
    ref_out, ref_state = execute_diagonal_recurrence(
        x=x, input_gate=ld_e, read_gate=ld_e, log_decay=ld_e, initial_state=state0,
        step_size=None, skip=0.0, read_before=False, gates_one=True,
    )
    state = state0.clone()
    outs = []
    with torch.no_grad():
        for t in range(T):
            outs.append(
                execute_diagonal_decode_step(
                    x=x[:, t], log_decay=ld[:, t], input_gate=None, read_gate=None,
                    state=state, read_before=False,
                )
            )
    step_out = torch.stack(outs, dim=1)
    assert (state.squeeze(-1) - ref_state.squeeze(-1)).abs().max().item() < 1e-5
    assert (step_out - ref_out.float()).abs().max().item() < 1e-4


def test_k1_decode_matches_full_history():
    from urm.backends.triton.k1.online import (
        execute_online_softmax,
        execute_online_softmax_decode,
    )

    B, H, K, V, S = 2, 8, 64, 64, 256
    torch.manual_seed(0)
    q = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, V, device="cuda", dtype=torch.bfloat16)
    scale = K**-0.5
    # The fused decode kernel must match the full-plan qlen=1 (full-history) output.
    ref = execute_online_softmax(
        q, k, v, attention_mask=None, score_bias=None, causal=True, scale=scale
    )
    decode = execute_online_softmax_decode(q[:, 0], k, v, scale=scale, causal=True)
    assert (decode.float() - ref[:, 0].float()).abs().max().item() < 5e-3


def test_open_decode_session_dispatch():
    """The plan exposes a decode session for the native K2/K3 families."""
    from urm.compiler.pipeline import MixerBackend, MixerIntent, compile_mixer
    from urm.frontend.recipes import named_mixer_recipe
    from urm.runtime.state import MatrixStateDecodeSession

    plan = compile_mixer(
        named_mixer_recipe("gated_delta_net"),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.INFERENCE,
        dtype="bfloat16",
    )
    state = torch.randn(2, 4, 32, 32, device="cuda", dtype=torch.float32) * 0.1
    session = plan.open_decode_session(
        initial_state=state, scale=1.0, decay_granularity="head",
        is_delta=True, read_before=False,
    )
    assert isinstance(session, MatrixStateDecodeSession)
    out = session.step(
        query=torch.randn(2, 4, 32, device="cuda"),
        key=torch.nn.functional.normalize(
            torch.randn(2, 4, 32, device="cuda"), dim=-1
        ),
        value=torch.randn(2, 4, 32, device="cuda"),
        log_decay=-torch.rand(2, 4, device="cuda") * 0.3,
        beta=torch.rand(2, 4, device="cuda"),
    )
    assert out.shape == (2, 4, 32)
