"""Tests for Dual-Form SDM Reparameterization backend in URM."""

import pytest
import torch
from urm.backends.dual_form_sdm import dual_form_sdm, DualFormSDMFunction
from urm.backends.sparse_state_reference import torch_sparse_state_mixer
from urm.compiler.semantic import SparseReadTiming


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Tensor Core backend")
def test_dual_form_numerical_equivalence_forward():
    """Verify bitwise/float64 forward equivalence against reference recurrent mixer."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float64
    P, T, S, D, W, R = 2, 16, 64, 8, 8, 8

    wi = torch.zeros(P, T, W, device=device, dtype=torch.int64)
    ri = torch.zeros(P, T, R, device=device, dtype=torch.int64)
    for p in range(P):
        for t in range(T):
            wi[p, t] = torch.randperm(S, device=device)[:W]
            ri[p, t] = torch.randperm(S, device=device)[:R]
    wi, _ = wi.sort(dim=-1)
    ri, _ = ri.sort(dim=-1)

    w = torch.rand(P, T, W, device=device, dtype=dtype)
    w = w / w.sum(dim=-1, keepdim=True)
    q = torch.rand(P, T, R, device=device, dtype=dtype)
    q = q / q.sum(dim=-1, keepdim=True)
    v = torch.randn(P, T, D, device=device, dtype=dtype) * 0.1
    b = torch.rand(P, T, 1, device=device, dtype=dtype) * 0.5
    g = -torch.rand(P, T, 1, device=device, dtype=dtype) * 0.05
    memory = torch.randn(P, S, D, device=device, dtype=dtype) * 0.1

    y_ref, m_ref = torch_sparse_state_mixer(
        memory, ri, q,
        write_indices=wi, write_weights=w,
        values=v, beta=b, log_decay=g,
        read_timing=SparseReadTiming.AFTER_UPDATE,
        accumulation_dtype=torch.float64,
    )

    y_dual, m_dual = dual_form_sdm(
        memory, ri, q,
        write_indices=wi, write_weights=w,
        values=v, beta=b, log_decay=g,
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )

    torch.testing.assert_close(y_dual.double(), y_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(m_dual.double(), m_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Tensor Core backend")
def test_dual_form_gradient_backward():
    """Verify analytical backward adjoint gradients execute cleanly and produce finite non-zero values."""
    torch.manual_seed(88)
    device = "cuda"
    dtype = torch.bfloat16
    P, T, S, D, W, R = 2, 32, 128, 16, 16, 16

    wi = torch.zeros(P, T, W, device=device, dtype=torch.int64)
    ri = torch.zeros(P, T, R, device=device, dtype=torch.int64)
    for p in range(P):
        for t in range(T):
            wi[p, t] = torch.randperm(S, device=device)[:W]
            ri[p, t] = torch.randperm(S, device=device)[:R]
    wi, _ = wi.sort(dim=-1)
    ri, _ = ri.sort(dim=-1)

    w = torch.rand(P, T, W, device=device, dtype=dtype, requires_grad=True)
    q = torch.rand(P, T, R, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(P, T, D, device=device, dtype=dtype, requires_grad=True)
    b = torch.rand(P, T, 1, device=device, dtype=dtype, requires_grad=True)
    g = (-torch.rand(P, T, 1, device=device, dtype=dtype) * 0.05).requires_grad_()
    memory = torch.randn(P, S, D, device=device, dtype=dtype, requires_grad=True)

    y, final_m = dual_form_sdm(
        memory, ri, q,
        write_indices=wi, write_weights=w,
        values=v, beta=b, log_decay=g,
    )

    loss = (y.float() ** 2).mean() + (final_m.float() ** 2).mean()
    loss.backward()

    for name, tensor in [
        ("memory", memory),
        ("v", v),
        ("w", w),
        ("q", q),
        ("beta", b),
    ]:
        assert tensor.grad is not None, f"Gradient for {name} was not computed"
        assert torch.isfinite(tensor.grad).all(), f"Non-finite gradient in {name}"
        assert tensor.grad.norm().item() > 0, f"Zero gradient in {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Tensor Core backend")
def test_decoupled_sparse_indexer_compatibility():
    """Verify that dual_form_sdm seamlessly interfaces with decoupled sparse routers (e.g. Foveal or Product-Key)."""
    device = "cuda"
    dtype = torch.bfloat16
    P, T, S, D, W, R = 1, 16, 64, 8, 4, 4

    # Simulated sparse indexer (e.g. 16D Foveal indexer output or Top-K routing)
    write_indices = torch.randint(0, S, (P, T, W), device=device, dtype=torch.int64)
    write_weights = torch.softmax(torch.randn(P, T, W, device=device, dtype=dtype), dim=-1)
    read_indices = torch.randint(0, S, (P, T, R), device=device, dtype=torch.int64)
    read_weights = torch.softmax(torch.randn(P, T, R, device=device, dtype=dtype), dim=-1)

    values = torch.randn(P, T, D, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(P, T, 1, device=device, dtype=dtype))
    log_decay = -torch.relu(torch.randn(P, T, 1, device=device, dtype=dtype) * 0.1)
    memory = torch.zeros(P, S, D, device=device, dtype=dtype)

    y, m_t = dual_form_sdm(
        memory,
        read_indices,
        read_weights,
        write_indices=write_indices,
        write_weights=write_weights,
        values=values,
        beta=beta,
        log_decay=log_decay,
    )
    assert y.shape == (P, T, D)
    assert m_t.shape == (P, S, D)
    assert torch.isfinite(y).all()
    assert torch.isfinite(m_t).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Tensor Core backend")
@pytest.mark.parametrize("seed", [42, 1701])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_exact_gradient_alignment_with_upstream_reference(seed, dtype):
    """Track exact gradient alignment with upstream reference setting identical random seed."""
    torch.manual_seed(seed)
    device = "cuda"
    T, S, D, W, R = 32, 128, 16, 8, 8

    # Independent token-by-token reference matching upstream Facebook SDM
    def upstream_reference(mem, ki, kw, val, b, g, qi, qw):
        readings = []
        curr_mem = mem.clone()
        for t in range(T):
            decay = torch.exp(g[t]).unsqueeze(-1)
            mem_read = curr_mem[ki[t]] * decay
            retrieved = (kw[t].unsqueeze(-1) * mem_read).sum(0)
            delta_v = b[t] * (val[t] - retrieved)
            curr_mem[ki[t]] = mem_read + kw[t].unsqueeze(-1) * delta_v.unsqueeze(0)
            readings.append((qw[t].unsqueeze(-1) * curr_mem[qi[t]]).sum(0))
        return torch.stack(readings, dim=0)

    memory_init = torch.randn(S, D, device=device, dtype=dtype) * 0.1
    k_idx = torch.stack([torch.randperm(S, device=device)[:W].sort().values for _ in range(T)])
    q_idx = torch.stack([torch.randperm(S, device=device)[:R].sort().values for _ in range(T)])

    kw_init = torch.rand(T, W, device=device, dtype=dtype)
    kw_init = kw_init / kw_init.sum(dim=-1, keepdim=True)
    qw_init = torch.rand(T, R, device=device, dtype=dtype)
    qw_init = qw_init / qw_init.sum(dim=-1, keepdim=True)

    v_init = torch.randn(T, D, device=device, dtype=dtype) * 0.1
    beta_init = torch.rand(T, 1, device=device, dtype=dtype) * 0.5
    g_init = -torch.rand(T, 1, device=device, dtype=dtype) * 0.05

    # 1. Reference Run
    mem_ref = memory_init.clone().requires_grad_(True)
    v_ref = v_init.clone().requires_grad_(True)
    b_ref = beta_init.clone().requires_grad_(True)
    g_ref = g_init.clone().requires_grad_(True)
    kw_ref = kw_init.clone().requires_grad_(True)
    qw_ref = qw_init.clone().requires_grad_(True)

    out_ref = upstream_reference(mem_ref, k_idx, kw_ref, v_ref, b_ref, g_ref, q_idx, qw_ref)
    loss_ref = (out_ref.float() ** 2).sum()
    loss_ref.backward()

    # 2. Dual-Form Run
    mem_dual = memory_init.clone().unsqueeze(0).requires_grad_(True)
    v_dual = v_init.clone().unsqueeze(0).requires_grad_(True)
    b_dual = beta_init.clone().unsqueeze(0).requires_grad_(True)
    g_dual = g_init.clone().unsqueeze(0).requires_grad_(True)
    kw_dual = kw_init.clone().unsqueeze(0).requires_grad_(True)
    qw_dual = qw_init.clone().unsqueeze(0).requires_grad_(True)

    out_dual, _ = DualFormSDMFunction.apply(
        mem_dual, q_idx.unsqueeze(0).long(), qw_dual,
        k_idx.unsqueeze(0).long(), kw_dual, v_dual, b_dual, g_dual
    )
    loss_dual = (out_dual.float().squeeze(0) ** 2).sum()
    loss_dual.backward()

    # 3. Assert alignment
    comparisons = [
        ("Forward Readings", out_ref, out_dual.squeeze(0)),
        ("grad(Values)", v_ref.grad, v_dual.grad.squeeze(0)),
        ("grad(Beta)", b_ref.grad, b_dual.grad.squeeze(0)),
        ("grad(Initial Memory)", mem_ref.grad, mem_dual.grad.squeeze(0)),
        ("grad(Write Weights)", kw_ref.grad, kw_dual.grad.squeeze(0)),
        ("grad(Read Weights)", qw_ref.grad, qw_dual.grad.squeeze(0)),
    ]

    for name, u_t, d_t in comparisons:
        u_f = u_t.reshape(-1).float()
        d_f = d_t.reshape(-1).float()
        cos_sim = torch.nn.functional.cosine_similarity(u_f.unsqueeze(0), d_f.unsqueeze(0)).item()
        abs_diff = (u_t.float() - d_t.float()).abs().max().item()

        tol = 0.03 if dtype is torch.bfloat16 else 1e-4
        assert cos_sim > 0.9999, f"{name} cosine similarity failed: {cos_sim:.6f} for seed={seed}, dtype={dtype}"
        assert abs_diff <= tol, f"{name} max error failed: {abs_diff:.4e} > {tol} for seed={seed}, dtype={dtype}"

