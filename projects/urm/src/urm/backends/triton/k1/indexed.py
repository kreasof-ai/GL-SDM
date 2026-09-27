"""Native indexed K1 gather-attend (the A2 family): per-query gathered source set."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _indexed_k1_forward_tiled(
    Q,
    K,
    V,
    GATHER,
    OUTPUT,
    LOGSUMEXP,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Indexed K1 gather-attend forward: softmax over a per-query gathered set.

    One program per ``(batch, query_head, query_block)``. The gather_indices
    operand ``GATHER`` is ``[B, HK, TQ, W]`` (per KV head, int32; -1 = padding →
    masked). Each query row owns W source positions; the kernel loops over the W
    slots, gathering the K/V row at each query's position (a per-row gather, no
    [BLOCK_M, W, D] tile — SMEM stays at the dense kernel's [BLOCK_M, BLOCK_D]
    tiles) and accumulating the online softmax in fp32. The route (indices) is
    external; there is no causal mask here — visibility is carried by the
    indices (-1 padding).
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
    query = tl.load(
        Q
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
        + key_dims[None, :],
        query_valid[:, None] & (key_dims[None, :] < D),
        other=0.0,
    )
    LOG2E: tl.constexpr = 1.4426950408889634
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_value = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    gather_base = GATHER + ((batch * HK + key_head) * TQ + query_offsets) * W
    for w in range(0, W):
        pos = tl.load(gather_base + w, query_valid, other=-1)
        slot_valid = query_valid & (pos >= 0)
        safe_pos = tl.where(pos >= 0, pos, 0)
        key = tl.load(
            K
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * D
            + key_dims[None, :],
            slot_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(query.to(tl.float32) * key, axis=1) * (SCALE * LOG2E)
        scores = tl.where(slot_valid, scores, float("-inf"))
        next_max = tl.maximum(running_max, scores)
        old_scale = tl.where(
            running_sum > 0.0, tl.exp2(running_max - next_max), 0.0
        )
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        probabilities = tl.where(
            scores == float("-inf"), 0.0, tl.exp2(scores - safe_max)
        )
        value = tl.load(
            V
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * DV
            + value_dims[None, :],
            slot_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        ).to(tl.float32)
        running_value = (
            running_value * old_scale[:, None]
            + probabilities[:, None] * value
        )
        running_sum = running_sum * old_scale + probabilities
        running_max = next_max
    output = tl.where(
        running_sum[:, None] > 0.0,
        running_value / tl.maximum(running_sum[:, None], 1.0e-30),
        0.0,
    )
    output_offset = (
        (batch * TQ + query_offsets[:, None]) * HQ + query_head
    ) * DV + value_dims[None, :]
    tl.store(
        OUTPUT + output_offset,
        output,
        query_valid[:, None] & (value_dims[None, :] < DV),
    )
    logsumexp = tl.where(
        running_sum > 0.0,
        running_max + tl.log2(tl.maximum(running_sum, 1.0e-30)),
        float("-inf"),
    )
    tl.store(
        LOGSUMEXP + (batch * HQ + query_head) * TQ + query_offsets,
        logsumexp,
        query_valid,
    )


@triton.jit
def _indexed_k1_backward_tiled(
    Q,
    K,
    V,
    GATHER,
    OUTPUT,
    LOGSUMEXP,
    GRAD_OUTPUT,
    GRAD_Q,
    GRAD_K,
    GRAD_V,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Indexed K1 gather-attend backward: dq plus a gather-scatter for dk/dv.

    One program per ``(batch, query_head, query_block)`` — it owns each query
    row exactly once, so dq is a plain store. The K/V cotangent at a gathered
    source position accumulates over every query that routed to it; that scatter
    reduces through relaxed ``tl.atomic_add`` (the K3 native backward policy, so
    cross-program accumulation order is not guaranteed). There is no cotangent
    for the integer route (gather_indices). delta = sum(dO·O) and the forward
    logsumexp recombine the softmax probabilities; the loop re-gathers the same
    K/V rows the forward read.
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
    query = tl.load(
        Q
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
        + key_dims[None, :],
        query_valid[:, None] & (key_dims[None, :] < D),
        other=0.0,
    )
    grad_output = tl.load(
        GRAD_OUTPUT
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    output = tl.load(
        OUTPUT
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    logsumexp = tl.load(
        LOGSUMEXP + (batch * HQ + query_head) * TQ + query_offsets,
        query_valid,
        other=float("-inf"),
    )
    row_valid = query_valid & (logsumexp != float("-inf"))
    safe_logsumexp = tl.where(logsumexp != float("-inf"), logsumexp, 0.0)
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    gather_base = GATHER + ((batch * HK + key_head) * TQ + query_offsets) * W
    for w in range(0, W):
        pos = tl.load(gather_base + w, query_valid, other=-1)
        slot_valid = row_valid & (pos >= 0)
        safe_pos = tl.where(pos >= 0, pos, 0)
        key = tl.load(
            K
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * D
            + key_dims[None, :],
            slot_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            V
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * DV
            + value_dims[None, :],
            slot_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        ).to(tl.float32)
        query_f = query.to(tl.float32)
        grad_output_f = grad_output.to(tl.float32)
        scores = tl.sum(query_f * key, axis=1) * SCALE
        scores = tl.where(slot_valid, scores, float("-inf"))
        probabilities = tl.where(
            scores == float("-inf"),
            0.0,
            tl.exp2(scores * 1.4426950408889634 - safe_logsumexp),
        )
        grad_probabilities = tl.sum(grad_output_f * value, axis=1)
        grad_scores = probabilities * (grad_probabilities - delta)
        grad_scores = tl.where(slot_valid, grad_scores, 0.0)
        grad_query += grad_scores[:, None] * key * SCALE
        # Gather-scatter: the source at safe_pos accumulates this query's
        # contribution; multiple queries share a source, so reduce via atomics.
        grad_key = grad_scores[:, None] * query_f * SCALE
        grad_value = probabilities[:, None] * grad_output_f
        grad_key_offset = (
            ((batch * TK + safe_pos[:, None]) * HK + key_head) * D
            + key_dims[None, :]
        )
        grad_value_offset = (
            ((batch * TK + safe_pos[:, None]) * HK + key_head) * DV
            + value_dims[None, :]
        )
        tl.atomic_add(
            GRAD_K + grad_key_offset,
            grad_key,
            slot_valid[:, None] & (key_dims[None, :] < D),
            sem="relaxed",
        )
        tl.atomic_add(
            GRAD_V + grad_value_offset,
            grad_value,
            slot_valid[:, None] & (value_dims[None, :] < DV),
            sem="relaxed",
        )
    grad_query_offset = (
        (batch * TQ + query_offsets[:, None]) * HQ + query_head
    ) * D + key_dims[None, :]
    tl.store(
        GRAD_Q + grad_query_offset,
        grad_query,
        query_valid[:, None] & (key_dims[None, :] < D),
    )


def execute_indexed_k1(
    query: Any,
    key: Any,
    value: Any,
    gather_indices: Any,
    *,
    scale: float,
) -> Any:
    """Run the indexed K1 gather-attend natively (softmax over the gathered set).

    ``query``/``key``/``value`` are BTHD rank-4; ``gather_indices`` is the
    external per-KV-head route ``[B, HK, TQ, W]`` (source positions, -1 = padding
    → masked). W is small (the clients use 2–64); the kernel loops over the W
    slots gathering per-row K/V, so SMEM stays at the dense kernel's tiles and
    the A10G's 101KB limit is met at head_dim <= 64 (same bound as the dense
    native K1). fp32 accumulation throughout; the backward scatters dk/dv via
    relaxed atomics (the K3 native policy).
    """
    if query.device.type != "cuda":
        raise ValueError("URM-native indexed K1 requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("indexed K1 query/key/value use BTHD rank-4 layout")
    batch, query_length, query_heads, key_dim = query.shape
    _, key_length, key_heads, _ = key.shape
    value_dim = value.shape[-1]
    if min(batch, query_length, query_heads, key_dim, key_length, key_heads, value_dim) <= 0:
        raise ValueError("indexed K1 dimensions must be positive")
    if gather_indices.ndim != 4:
        raise ValueError("indexed K1 gather_indices must be [B, HK, TQ, W]")
    if tuple(gather_indices.shape[:3]) != (batch, key_heads, query_length):
        raise ValueError("indexed K1 gather_indices must be [B, HK, TQ, W]")
    width = gather_indices.shape[-1]
    if width <= 0:
        raise ValueError("indexed K1 gather width W must be positive")
    if query_heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if not (query.dtype == key.dtype == value.dtype) or not query.is_floating_point():
        raise ValueError("indexed K1 query/key/value must use one floating-point dtype")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native indexed K1 supports float16, bfloat16, and float32")
    if not (query.device == key.device == value.device == gather_indices.device):
        raise ValueError("indexed K1 operands must share one CUDA device")
    # The forward holds [BLOCK_M, BLOCK_D] q + per-slot gathered [BLOCK_M,
    # BLOCK_D] k / [BLOCK_M, BLOCK_V] v tiles; the A10G's 101KB SMEM caps head
    # widths at 64 (the dense native K1 bound). Decline wider heads loudly.
    if key_dim > 64 or value_dim > 64:
        raise ValueError(
            f"native indexed K1 supports head widths <= 64 on this device "
            f"(SMEM limit); got key_dim={key_dim}, value_dim={value_dim} — "
            f"the reference tier owns wider heads"
        )
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    gather = gather_indices.contiguous().to(torch.int32)
    block_m = 64 if query_length >= 64 else max(16, triton.next_power_of_2(query_length))
    block_d = max(16, triton.next_power_of_2(key_dim))
    block_v = max(16, triton.next_power_of_2(value_dim))
    # The per-slot gather keeps only dense-sized tiles live, so the forward runs
    # at block_m<=64, num_stages=1 like the SMEM-safe backward (measured under
    # the A10G's 101KB at head_dim=64).
    num_warps = 4
    num_stages = 1

    class _IndexedK1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, gather_idx):
            output = torch.empty(
                (batch, query_length, query_heads, value_dim),
                device=q.device,
                dtype=q.dtype,
            )
            logsumexp = torch.empty(
                (batch, query_heads, query_length), device=q.device, dtype=torch.float32
            )
            _indexed_k1_forward_tiled[
                (batch, query_heads, triton.cdiv(query_length, block_m))
            ](
                q,
                k,
                v,
                gather_idx,
                output,
                logsumexp,
                query_length,
                key_length,
                query_heads,
                key_heads,
                key_dim,
                value_dim,
                width,
                scale,
                q.dtype is torch.float32,
                block_m,
                block_d,
                block_v,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            ctx.save_for_backward(q, k, v, gather_idx, output, logsumexp)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, gather_idx, output, logsumexp = ctx.saved_tensors
            if grad_output is None:
                return None, None, None, None
            grad_q = torch.empty(q.shape, device=q.device, dtype=torch.float32)
            # dk/dv accumulate a gather-scatter across queries (relaxed atomics),
            # so they start from zero.
            grad_k = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
            grad_v = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            _indexed_k1_backward_tiled[
                (batch, query_heads, triton.cdiv(query_length, block_m))
            ](
                q,
                k,
                v,
                gather_idx,
                output,
                logsumexp,
                grad_output.contiguous(),
                grad_q,
                grad_k,
                grad_v,
                query_length,
                key_length,
                query_heads,
                key_heads,
                key_dim,
                value_dim,
                width,
                scale,
                q.dtype is torch.float32,
                block_m,
                block_d,
                block_v,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return (
                grad_q.to(q.dtype),
                grad_k.to(k.dtype),
                grad_v.to(v.dtype),
                None,  # gather_indices is an integer route; no cotangent
            )

    return _IndexedK1.apply(query, key, value, gather)


class K1NativeIndexedTritonProvider:
    """The native Triton provider for the A2 indexed-K1 gather-attend law.

    Serves ONLY ``descriptor.indexed == True`` (the dense native anchor declines
    those); the gathered source set comes from the external ``gather_indices``
    route. The equation is the same softmax-over-inline-Q·K law, restricted to
    the per-query gathered set — the descriptor's causal field is irrelevant on
    the indexed path (visibility is carried by the -1 padding in the route).
    """

    name = "urm_native_k1_indexed_gather_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ....ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        # This provider owns ONLY the indexed gather-attend: the plain dot score
        # with a softmax reducer over the gathered set. Decline everything else
        # (the dense native anchor serves the non-indexed softmax law).
        if not request.descriptor.indexed:
            return "native indexed K1 serves only indexed gather descriptors"
        if request.descriptor.score_law is not K1ScoreLaw.DOT:
            return "native indexed K1 requires the DOT score law"
        if request.descriptor.reducer_law is not K1ReducerLaw.SOFTMAX:
            return "native indexed K1 requires the SOFTMAX reducer law"
        import torch

        if not torch.cuda.is_available():
            return "native indexed K1 requires CUDA"
        return None

    def execute(self, request, operands):
        from urm.ir.program import K1ScaleRule as _SR

        descriptor = request.descriptor
        gather_indices = operands.get("gather_indices")
        if gather_indices is None:
            raise ValueError("indexed K1 requires a gather_indices operand")
        scale_op = operands.get("scale")
        if descriptor.scale_rule is _SR.EXPLICIT_OPERAND:
            if scale_op is None:
                raise ValueError("K1 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        else:
            scale = float(operands["query"].shape[-1]) ** -0.5
        out = execute_indexed_k1(
            operands["query"],
            operands["key"],
            operands["value"],
            gather_indices,
            scale=scale,
        )
        return {"output": out}
