"""Fused Triton scan for diagonal affine state recurrences."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def compose_affine(a_left, b_left, a_right, b_right):
        return a_right * a_left, b_right + a_right * b_left

    @triton.jit
    def forward_kernel(
        X,
        INPUT_GATE,
        READ_GATE,
        LOG_DECAY,
        INITIAL,
        STEP_SIZE,
        SKIP,
        OUTPUT,
        STATES,
        FINAL,
        T: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        X_SB: tl.constexpr,
        X_ST: tl.constexpr,
        X_SC: tl.constexpr,
        IG_SB: tl.constexpr,
        IG_ST: tl.constexpr,
        IG_SC: tl.constexpr,
        IG_SN: tl.constexpr,
        RG_SB: tl.constexpr,
        RG_ST: tl.constexpr,
        RG_SC: tl.constexpr,
        RG_SN: tl.constexpr,
        LD_SB: tl.constexpr,
        LD_ST: tl.constexpr,
        LD_SC: tl.constexpr,
        LD_SN: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_STEP_SIZE: tl.constexpr,
        STEP_SB: tl.constexpr,
        STEP_ST: tl.constexpr,
        STEP_SC: tl.constexpr,
        SKIP_SCALAR: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CHUNK_T: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // C
        channel = row % C
        state_index = tl.arange(0, BLOCK_N)
        state_mask = state_index < N
        if HAS_INITIAL:
            state = tl.load(
                INITIAL + batch * C * N + channel * N + state_index,
                state_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            state = tl.full((BLOCK_N,), 0.0, tl.float32)
        skip_index = 0 if SKIP_SCALAR else channel
        skip = tl.load(SKIP + skip_index).to(tl.float32)
        if READ_BEFORE:
            for token in range(T):
                x = tl.load(X + batch * X_SB + token * X_ST + channel * X_SC).to(
                    tl.float32
                )
                input_gate = tl.load(
                    INPUT_GATE
                    + batch * IG_SB
                    + token * IG_ST
                    + channel * IG_SC
                    + state_index * IG_SN,
                    state_mask,
                    other=0.0,
                ).to(tl.float32)
                read_gate = tl.load(
                    READ_GATE
                    + batch * RG_SB
                    + token * RG_ST
                    + channel * RG_SC
                    + state_index * RG_SN,
                    state_mask,
                    other=0.0,
                ).to(tl.float32)
                log_decay = tl.load(
                    LOG_DECAY
                    + batch * LD_SB
                    + token * LD_ST
                    + channel * LD_SC
                    + state_index * LD_SN,
                    state_mask,
                    other=0.0,
                ).to(tl.float32)
                if HAS_STEP_SIZE:
                    step = tl.load(
                        STEP_SIZE
                        + batch * STEP_SB
                        + token * STEP_ST
                        + channel * STEP_SC
                    ).to(tl.float32)
                else:
                    step = 1.0
                output = tl.sum(state * read_gate, 0) + x * skip
                tl.store(OUTPUT + batch * T * C + token * C + channel, output)
                state = tl.exp(log_decay * step) * state + (x * step) * input_gate
                tl.store(
                    STATES
                    + batch * T * C * N
                    + token * C * N
                    + channel * N
                    + state_index,
                    state,
                    state_mask,
                )
            tl.store(
                FINAL + batch * C * N + channel * N + state_index,
                state,
                state_mask,
            )
        else:
            # Chunked forward scan: loop over sequence chunks of CHUNK_T tokens,
            # running a parallel associative_scan within each chunk and carrying the
            # state in registers across chunks. This bounds register pressure (the
            # tile is [CHUNK_T, BLOCK_N] instead of [T, BLOCK_N]) while keeping the
            # whole sequence in one program.
            state_offset = state_index[None, :]
            for chunk_start in range(0, T, CHUNK_T):
                token = chunk_start + tl.arange(0, CHUNK_T)
                token_mask = token < T
                x = tl.load(
                    X + batch * X_SB + token * X_ST + channel * X_SC,
                    token_mask,
                    other=0.0,
                ).to(tl.float32)
                input_gate = tl.load(
                    INPUT_GATE
                    + batch * IG_SB
                    + token[:, None] * IG_ST
                    + channel * IG_SC
                    + state_offset * IG_SN,
                    token_mask[:, None] & state_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                read_gate = tl.load(
                    READ_GATE
                    + batch * RG_SB
                    + token[:, None] * RG_ST
                    + channel * RG_SC
                    + state_offset * RG_SN,
                    token_mask[:, None] & state_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                log_decay = tl.load(
                    LOG_DECAY
                    + batch * LD_SB
                    + token[:, None] * LD_ST
                    + channel * LD_SC
                    + state_offset * LD_SN,
                    token_mask[:, None] & state_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if HAS_STEP_SIZE:
                    step = tl.load(
                        STEP_SIZE + batch * STEP_SB + token * STEP_ST + channel * STEP_SC,
                        token_mask,
                        other=1.0,
                    ).to(tl.float32)
                else:
                    step = tl.full((CHUNK_T,), 1.0, tl.float32)
                decay = tl.exp(log_decay * step[:, None])
                decay = tl.where(token_mask[:, None], decay, 1.0)
                update = (x * step)[:, None] * input_gate
                update = tl.where(token_mask[:, None], update, 0.0)
                prefix_decay, prefix_update = tl.associative_scan(
                    (decay, update), axis=0, combine_fn=compose_affine
                )
                state_sequence = prefix_decay * state[None, :] + prefix_update
                output = tl.sum(state_sequence * read_gate, axis=1) + x * skip
                output_offset = batch * T * C + token * C + channel
                tl.store(OUTPUT + output_offset, output, token_mask)
                tl.store(
                    STATES
                    + batch * T * C * N
                    + token[:, None] * C * N
                    + channel * N
                    + state_offset,
                    state_sequence,
                    token_mask[:, None] & state_mask[None, :],
                )
                # Carry the post-update state of the last valid token to the next chunk.
                last_valid = tl.sum(tl.where(token_mask, 1, 0), 0) - 1
                state = tl.sum(
                    tl.where((tl.arange(0, CHUNK_T) == last_valid)[:, None], state_sequence, 0.0),
                    axis=0,
                )
            tl.store(
                FINAL + batch * C * N + channel * N + state_index,
                state,
                state_mask,
            )

    @triton.jit
    def backward_kernel(
        X,
        INPUT_GATE,
        READ_GATE,
        LOG_DECAY,
        INITIAL,
        STEP_SIZE,
        SKIP,
        GRAD_OUTPUT,
        STATES,
        GRAD_X,
        GRAD_INPUT_GATE,
        GRAD_READ_GATE,
        GRAD_LOG_DECAY,
        GRAD_INITIAL,
        GRAD_STEP_SIZE,
        GRAD_FINAL,
        T: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        X_SB: tl.constexpr,
        X_ST: tl.constexpr,
        X_SC: tl.constexpr,
        IG_SB: tl.constexpr,
        IG_ST: tl.constexpr,
        IG_SC: tl.constexpr,
        IG_SN: tl.constexpr,
        RG_SB: tl.constexpr,
        RG_ST: tl.constexpr,
        RG_SC: tl.constexpr,
        RG_SN: tl.constexpr,
        LD_SB: tl.constexpr,
        LD_ST: tl.constexpr,
        LD_SC: tl.constexpr,
        LD_SN: tl.constexpr,
        GIG_SB: tl.constexpr,
        GIG_ST: tl.constexpr,
        GIG_SC: tl.constexpr,
        GIG_SN: tl.constexpr,
        GRG_SB: tl.constexpr,
        GRG_ST: tl.constexpr,
        GRG_SC: tl.constexpr,
        GRG_SN: tl.constexpr,
        GLD_SB: tl.constexpr,
        GLD_ST: tl.constexpr,
        GLD_SC: tl.constexpr,
        GLD_SN: tl.constexpr,
        STEP_SB: tl.constexpr,
        STEP_ST: tl.constexpr,
        STEP_SC: tl.constexpr,
        GST_SB: tl.constexpr,
        GST_ST: tl.constexpr,
        GST_SC: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_STEP_SIZE: tl.constexpr,
        SKIP_SCALAR: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        HAS_GRAD_FINAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // C
        channel = row % C
        state_index = tl.arange(0, BLOCK_N)
        state_mask = state_index < N
        if HAS_GRAD_FINAL:
            carry = tl.load(
                GRAD_FINAL + batch * C * N + channel * N + state_index,
                state_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            carry = tl.full((BLOCK_N,), 0.0, tl.float32)
        for reverse_index in range(T):
            token = T - reverse_index - 1
            x = tl.load(X + batch * X_SB + token * X_ST + channel * X_SC).to(tl.float32)
            grad_output = tl.load(GRAD_OUTPUT + batch * T * C + token * C + channel).to(
                tl.float32
            )
            input_gate = tl.load(
                INPUT_GATE
                + batch * IG_SB
                + token * IG_ST
                + channel * IG_SC
                + state_index * IG_SN,
                state_mask,
                other=0.0,
            ).to(tl.float32)
            read_gate = tl.load(
                READ_GATE
                + batch * RG_SB
                + token * RG_ST
                + channel * RG_SC
                + state_index * RG_SN,
                state_mask,
                other=0.0,
            ).to(tl.float32)
            log_decay = tl.load(
                LOG_DECAY
                + batch * LD_SB
                + token * LD_ST
                + channel * LD_SC
                + state_index * LD_SN,
                state_mask,
                other=0.0,
            ).to(tl.float32)
            if HAS_STEP_SIZE:
                step = tl.load(
                    STEP_SIZE + batch * STEP_SB + token * STEP_ST + channel * STEP_SC
                ).to(tl.float32)
            else:
                step = 1.0
            decay = tl.exp(log_decay * step)
            state_offset = batch * T * C * N + token * C * N + channel * N + state_index
            state_after = tl.load(STATES + state_offset, state_mask, other=0.0).to(
                tl.float32
            )
            if token > 0:
                state_before = tl.load(
                    STATES + state_offset - C * N, state_mask, other=0.0
                ).to(tl.float32)
            elif HAS_INITIAL:
                state_before = tl.load(
                    INITIAL + batch * C * N + channel * N + state_index,
                    state_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                state_before = tl.full((BLOCK_N,), 0.0, tl.float32)
            if READ_BEFORE:
                grad_read = grad_output * state_before
                state_cotangent = carry
                grad_input = state_cotangent * x * step
                grad_decay = state_cotangent * decay * state_before * step
                grad_x = tl.sum(state_cotangent * input_gate * step, 0)
                if HAS_STEP_SIZE:
                    grad_step = tl.sum(
                        state_cotangent
                        * (decay * log_decay * state_before + x * input_gate),
                        0,
                    )
                carry = state_cotangent * decay + grad_output * read_gate
            else:
                grad_read = grad_output * state_after
                state_cotangent = carry + grad_output * read_gate
                grad_input = state_cotangent * x * step
                grad_decay = state_cotangent * decay * state_before * step
                grad_x = tl.sum(state_cotangent * input_gate * step, 0)
                if HAS_STEP_SIZE:
                    grad_step = tl.sum(
                        state_cotangent
                        * (decay * log_decay * state_before + x * input_gate),
                        0,
                    )
                carry = state_cotangent * decay
            skip = tl.load(SKIP + (0 if SKIP_SCALAR else channel)).to(tl.float32)
            grad_x += grad_output * skip
            tl.store(
                GRAD_INPUT_GATE
                + batch * GIG_SB
                + token * GIG_ST
                + channel * GIG_SC
                + state_index * GIG_SN,
                grad_input,
                state_mask,
            )
            tl.store(
                GRAD_READ_GATE
                + batch * GRG_SB
                + token * GRG_ST
                + channel * GRG_SC
                + state_index * GRG_SN,
                grad_read,
                state_mask,
            )
            tl.store(
                GRAD_LOG_DECAY
                + batch * GLD_SB
                + token * GLD_ST
                + channel * GLD_SC
                + state_index * GLD_SN,
                grad_decay,
                state_mask,
            )
            tl.store(GRAD_X + batch * T * C + token * C + channel, grad_x)
            if HAS_STEP_SIZE:
                tl.store(
                    GRAD_STEP_SIZE + batch * GST_SB + token * GST_ST + channel * GST_SC,
                    grad_step,
                )
        if HAS_INITIAL:
            tl.store(
                GRAD_INITIAL + batch * C * N + channel * N + state_index,
                carry,
                state_mask,
            )

    @triton.jit
    def backward_kernel_parallel(
        X,
        INPUT_GATE,
        READ_GATE,
        LOG_DECAY,
        INITIAL,
        STEP_SIZE,
        SKIP,
        GRAD_OUTPUT,
        STATES,
        GRAD_X,
        GRAD_INPUT_GATE,
        GRAD_READ_GATE,
        GRAD_LOG_DECAY,
        GRAD_INITIAL,
        GRAD_STEP_SIZE,
        GRAD_FINAL,
        T: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        X_SB: tl.constexpr,
        X_ST: tl.constexpr,
        X_SC: tl.constexpr,
        IG_SB: tl.constexpr,
        IG_ST: tl.constexpr,
        IG_SC: tl.constexpr,
        IG_SN: tl.constexpr,
        RG_SB: tl.constexpr,
        RG_ST: tl.constexpr,
        RG_SC: tl.constexpr,
        RG_SN: tl.constexpr,
        LD_SB: tl.constexpr,
        LD_ST: tl.constexpr,
        LD_SC: tl.constexpr,
        LD_SN: tl.constexpr,
        GIG_SB: tl.constexpr,
        GIG_ST: tl.constexpr,
        GIG_SC: tl.constexpr,
        GIG_SN: tl.constexpr,
        GRG_SB: tl.constexpr,
        GRG_ST: tl.constexpr,
        GRG_SC: tl.constexpr,
        GRG_SN: tl.constexpr,
        GLD_SB: tl.constexpr,
        GLD_ST: tl.constexpr,
        GLD_SC: tl.constexpr,
        GLD_SN: tl.constexpr,
        STEP_SB: tl.constexpr,
        STEP_ST: tl.constexpr,
        STEP_SC: tl.constexpr,
        GST_SB: tl.constexpr,
        GST_ST: tl.constexpr,
        GST_SC: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_STEP_SIZE: tl.constexpr,
        SKIP_SCALAR: tl.constexpr,
        HAS_GRAD_FINAL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # Parallel reverse-scan backward for the read-after-update (READ_BEFORE=False)
        # diagonal recurrence. The state cotangent sc_t satisfies the reverse affine
        # recurrence sc_t = decay_{t+1} * sc_{t+1} + grad_output_t * read_gate_t, which
        # tl.associative_scan(reverse=True) evaluates in parallel over the sequence.
        row = tl.program_id(0)
        batch = row // C
        channel = row % C
        token = tl.arange(0, BLOCK_T)
        state_index = tl.arange(0, BLOCK_N)
        token_mask = token < T
        state_mask = state_index < N
        mask2 = token_mask[:, None] & state_mask[None, :]
        # Per-token loads over the whole sequence tile.
        x = tl.load(X + batch * X_SB + token * X_ST + channel * X_SC, token_mask, other=0.0).to(tl.float32)
        grad_output = tl.load(
            GRAD_OUTPUT + batch * T * C + token * C + channel, token_mask, other=0.0
        ).to(tl.float32)
        ig = tl.load(
            INPUT_GATE + batch * IG_SB + token[:, None] * IG_ST + channel * IG_SC + state_index[None, :] * IG_SN,
            mask2, other=0.0,
        ).to(tl.float32)
        rg = tl.load(
            READ_GATE + batch * RG_SB + token[:, None] * RG_ST + channel * RG_SC + state_index[None, :] * RG_SN,
            mask2, other=0.0,
        ).to(tl.float32)
        ld = tl.load(
            LOG_DECAY + batch * LD_SB + token[:, None] * LD_ST + channel * LD_SC + state_index[None, :] * LD_SN,
            mask2, other=0.0,
        ).to(tl.float32)
        if HAS_STEP_SIZE:
            step = tl.load(
                STEP_SIZE + batch * STEP_SB + token * STEP_ST + channel * STEP_SC, token_mask, other=1.0
            ).to(tl.float32)
        else:
            step = tl.full((BLOCK_T,), 1.0, tl.float32)
        decay = tl.exp(ld * step[:, None])
        decay = tl.where(mask2, decay, 1.0)
        # state_after_t = STATES[t]; state_before_t = STATES[t-1] (or initial at t=0).
        state_after = tl.load(
            STATES + batch * T * C * N + token[:, None] * C * N + channel * N + state_index[None, :],
            mask2, other=0.0,
        ).to(tl.float32)
        prev_offset = batch * T * C * N + (token[:, None] - 1) * C * N + channel * N + state_index[None, :]
        state_before = tl.load(
            STATES + prev_offset, (token[:, None] > 0) & state_mask[None, :], other=0.0
        ).to(tl.float32)
        if HAS_INITIAL:
            init = tl.load(INITIAL + batch * C * N + channel * N + state_index, state_mask, other=0.0).to(tl.float32)
            state_before = tl.where((token[:, None] == 0) & state_mask[None, :], init[None, :], state_before)
        # decay_{t+1}: shift log_decay and step forward by one token.
        ld_next = tl.load(
            LOG_DECAY + batch * LD_SB + (token[:, None] + 1) * LD_ST + channel * LD_SC + state_index[None, :] * LD_SN,
            (token[:, None] + 1 < T) & state_mask[None, :], other=0.0,
        ).to(tl.float32)
        if HAS_STEP_SIZE:
            step_next = tl.load(
                STEP_SIZE + batch * STEP_SB + (token + 1) * STEP_ST + channel * STEP_SC,
                token + 1 < T, other=1.0,
            ).to(tl.float32)
        else:
            step_next = tl.full((BLOCK_T,), 1.0, tl.float32)
        decay_next = tl.exp(ld_next * step_next[:, None])
        # Reverse scan for sc. g_t = grad_output_t * read_gate_t.
        g_t = grad_output[:, None] * rg
        if HAS_GRAD_FINAL:
            gf = tl.load(GRAD_FINAL + batch * C * N + channel * N + state_index, state_mask, other=0.0).to(tl.float32)
        else:
            gf = tl.zeros((BLOCK_N,), tl.float32)
        is_last = (token == (T - 1))[:, None]
        a = tl.where(is_last, 0.0, decay_next)
        b = tl.where(is_last, gf[None, :] + g_t, g_t)
        a = tl.where(mask2, a, 1.0)
        b = tl.where(mask2, b, 0.0)
        _, sc = tl.associative_scan((a, b), axis=0, combine_fn=compose_affine, reverse=True)
        # Per-token gradients from the state cotangent sc_t.
        skip_index = 0 if SKIP_SCALAR else channel
        skip = tl.load(SKIP + skip_index).to(tl.float32)
        grad_read = grad_output[:, None] * state_after
        grad_input = sc * (x * step)[:, None]
        grad_decay = sc * decay * state_before * step[:, None]
        grad_x = tl.sum(sc * ig * step[:, None], 1) + grad_output * skip
        tl.store(
            GRAD_INPUT_GATE + batch * GIG_SB + token[:, None] * GIG_ST + channel * GIG_SC + state_index[None, :] * GIG_SN,
            grad_input, mask2,
        )
        tl.store(
            GRAD_READ_GATE + batch * GRG_SB + token[:, None] * GRG_ST + channel * GRG_SC + state_index[None, :] * GRG_SN,
            grad_read, mask2,
        )
        tl.store(
            GRAD_LOG_DECAY + batch * GLD_SB + token[:, None] * GLD_ST + channel * GLD_SC + state_index[None, :] * GLD_SN,
            grad_decay, mask2,
        )
        tl.store(GRAD_X + batch * T * C + token * C + channel, grad_x, token_mask)
        if HAS_STEP_SIZE:
            grad_step = tl.sum(sc * (decay * ld * state_before + x[:, None] * ig), 1)
            tl.store(
                GRAD_STEP_SIZE + batch * GST_SB + token * GST_ST + channel * GST_SC, grad_step, token_mask
            )
        if HAS_INITIAL:
            carry0 = tl.sum(tl.where((token == 0)[:, None], sc * decay, 0.0), axis=0)
            tl.store(GRAD_INITIAL + batch * C * N + channel * N + state_index, carry0, state_mask)

    return triton, forward_kernel, backward_kernel, backward_kernel_parallel


def execute_diagonal_recurrence(
    *,
    x: Any,
    input_gate: Any,
    read_gate: Any,
    log_decay: Any,
    initial_state: Any | None,
    step_size: Any | None,
    skip: Any,
    read_before: bool,
) -> tuple[Any, Any]:
    """Run the fused recurrence and return output plus the final FP32 state."""
    import torch

    triton, forward_kernel, backward_kernel, backward_kernel_parallel = _kernels()
    if x.device.type != "cuda":
        raise ValueError("native diagonal SSM requires CUDA tensors")
    if x.dtype is not torch.float32:
        raise ValueError("native diagonal SSM currently supports float32")
    batch, sequence, channels = x.shape
    state_width = input_gate.shape[-1]
    if max(batch, sequence, channels, state_width) <= 0:
        raise ValueError("native diagonal SSM dimensions must be positive")
    if any(
        tensor.dtype is not torch.float32
        for tensor in (input_gate, read_gate, log_decay)
    ):
        raise ValueError("native diagonal SSM gates must be float32")
    if initial_state is None:
        initial_tensor = torch.zeros(
            (batch, channels, state_width), device=x.device, dtype=x.dtype
        )
        has_initial = False
    else:
        # The kernels index the state as dense [B,C,N]. This copy remains in
        # the autograd graph, so gradients still return to transposed or
        # expanded caller storage through PyTorch's copy backward.
        initial_tensor = initial_state.contiguous()
        has_initial = True
        if initial_tensor.shape != (batch, channels, state_width):
            raise ValueError("initial_state must have shape [B,C,N]")
        if initial_tensor.dtype is not torch.float32:
            raise ValueError("native diagonal initial_state must be float32")
    input_gate = _expand_gate(input_gate, batch, sequence, channels)
    read_gate = _expand_gate(read_gate, batch, sequence, channels)
    log_decay = _expand_gate(log_decay, batch, sequence, channels)
    has_step_size = step_size is not None
    if step_size is None:
        step_tensor = torch.ones(
            (batch, sequence, channels), device=x.device, dtype=x.dtype
        )
    else:
        if step_size.shape not in ((batch, sequence), (batch, sequence, channels)):
            raise ValueError("step_size must use [B,T] or [B,T,C]")
        if step_size.ndim == 2:
            step_size = step_size.unsqueeze(-1).expand(batch, sequence, channels)
        if step_size.dtype is not torch.float32 or step_size.device != x.device:
            raise ValueError(
                "native diagonal step_size must be float32 on the x device"
            )
        step_tensor = step_size
    if any(
        tensor.device != x.device
        for tensor in (input_gate, read_gate, log_decay, initial_tensor, step_tensor)
    ):
        raise ValueError("native diagonal SSM tensors must share a device")
    skip_tensor = torch.as_tensor(skip, device=x.device, dtype=torch.float32)
    skip_scalar = skip_tensor.numel() == 1
    if not skip_scalar and tuple(skip_tensor.shape) != (channels,):
        raise ValueError("skip must be a scalar or channel vector")
    skip_tensor = skip_tensor.contiguous()
    block_n = triton.next_power_of_2(state_width)
    block_t = triton.next_power_of_2(sequence)
    # Chunk the forward scan to bound register pressure: the per-chunk tile is
    # [chunk_t, block_n] instead of [block_t, block_n]. 128 balances parallelism
    # against the sequential carry across chunks.
    chunk_t = min(block_t, 128)
    warps = 4 if block_n <= 128 else 8

    # Cache the autograd.Function subclass per launch configuration: defining the
    # class runs `__build_class__` every call, which is pure dispatch overhead on
    # the ordinary-invocation path the production budget measures.
    scan_cls = _scan_class(
        sequence, channels, state_width, has_initial, has_step_size, skip_scalar,
        read_before, block_t, block_n, chunk_t, warps,
    )
    return scan_cls.apply(
        x, input_gate, read_gate, log_decay, initial_tensor, step_tensor, skip_tensor
    )


@lru_cache(maxsize=128)
def _scan_class(
    sequence, channels, state_width, has_initial, has_step_size, skip_scalar,
    read_before, block_t, block_n, chunk_t, warps,
):
    import torch

    triton, forward_kernel, backward_kernel, backward_kernel_parallel = _kernels()
    batch = None  # batch varies per call; read from x inside forward/backward

    class _DiagonalScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, input_gate, read_gate, log_decay, initial, step, skip):
            batch = x.shape[0]
            output = torch.empty(
                (batch, sequence, channels), device=x.device, dtype=x.dtype
            )
            states = torch.empty(
                (batch, sequence, channels, state_width),
                device=x.device,
                dtype=torch.float32,
            )
            final = torch.empty(
                (batch, channels, state_width), device=x.device, dtype=torch.float32
            )
            forward_kernel[(batch * channels,)](
                x,
                input_gate,
                read_gate,
                log_decay,
                initial,
                step,
                skip,
                output,
                states,
                final,
                sequence,
                channels,
                state_width,
                *x.stride(),
                *input_gate.stride(),
                *read_gate.stride(),
                *log_decay.stride(),
                has_initial,
                has_step_size,
                *step.stride(),
                skip_scalar,
                read_before,
                block_t,
                block_n,
                chunk_t,
                num_warps=warps,
            )
            ctx.save_for_backward(
                x, input_gate, read_gate, log_decay, initial, step, skip, states
            )
            ctx.has_initial = has_initial
            ctx.has_step_size = has_step_size
            ctx.read_before = read_before
            ctx.skip_scalar = skip_scalar
            return output, final

        @staticmethod
        def backward(ctx, grad_output, grad_final):
            x, input_gate, read_gate, log_decay, initial, step, skip, states = (
                ctx.saved_tensors
            )
            batch = x.shape[0]
            if grad_output is None:
                grad_output = torch.zeros_like(x)
            grad_output = grad_output.contiguous()
            grad_x = torch.empty(
                (batch, sequence, channels), device=x.device, dtype=x.dtype
            )
            grad_input_gate = torch.empty_like(
                input_gate, memory_format=torch.contiguous_format
            )
            grad_read_gate = torch.empty_like(
                read_gate, memory_format=torch.contiguous_format
            )
            grad_log_decay = torch.empty_like(
                log_decay, memory_format=torch.contiguous_format
            )
            grad_initial = torch.empty_like(initial)
            grad_step = torch.empty_like(step, memory_format=torch.contiguous_format)
            grad_final_tensor = (
                torch.zeros_like(initial)
                if grad_final is None
                else grad_final.contiguous()
            )
            if not ctx.read_before:
                # Parallel reverse-scan backward (read-after-update): evaluates the
                # state-cotangent recurrence with tl.associative_scan instead of a
                # sequential reverse loop.
                backward_kernel_parallel[(batch * channels,)](
                    x,
                    input_gate,
                    read_gate,
                    log_decay,
                    initial,
                    step,
                    skip,
                    grad_output,
                    states,
                    grad_x,
                    grad_input_gate,
                    grad_read_gate,
                    grad_log_decay,
                    grad_initial,
                    grad_step,
                    grad_final_tensor,
                    sequence,
                    channels,
                    state_width,
                    *x.stride(),
                    *input_gate.stride(),
                    *read_gate.stride(),
                    *log_decay.stride(),
                    *grad_input_gate.stride(),
                    *grad_read_gate.stride(),
                    *grad_log_decay.stride(),
                    *step.stride(),
                    *grad_step.stride(),
                    ctx.has_initial,
                    ctx.has_step_size,
                    ctx.skip_scalar,
                    grad_final is not None,
                    block_t,
                    block_n,
                    num_warps=warps,
                )
            else:
                backward_kernel[(batch * channels,)](
                    x,
                    input_gate,
                    read_gate,
                    log_decay,
                    initial,
                    step,
                    skip,
                    grad_output,
                    states,
                    grad_x,
                    grad_input_gate,
                    grad_read_gate,
                    grad_log_decay,
                    grad_initial,
                    grad_step,
                    grad_final_tensor,
                    sequence,
                    channels,
                    state_width,
                    *x.stride(),
                    *input_gate.stride(),
                    *read_gate.stride(),
                    *log_decay.stride(),
                    *grad_input_gate.stride(),
                    *grad_read_gate.stride(),
                    *grad_log_decay.stride(),
                    *step.stride(),
                    *grad_step.stride(),
                    ctx.has_initial,
                    ctx.has_step_size,
                    ctx.skip_scalar,
                    ctx.read_before,
                    grad_final is not None,
                    block_n,
                    num_warps=warps,
                )
            grad_skip_full = grad_output * x
            if ctx.skip_scalar:
                grad_skip = grad_skip_full.sum().reshape_as(skip)
            else:
                grad_skip = grad_skip_full.sum(dim=(0, 1)).reshape_as(skip)
            return (
                grad_x,
                grad_input_gate,
                grad_read_gate,
                grad_log_decay,
                grad_initial if ctx.has_initial else None,
                grad_step if ctx.has_step_size else None,
                grad_skip,
            )

    return _DiagonalScan


def _expand_gate(gate: Any, batch: int, sequence: int, channels: int):
    if gate.ndim == 3:
        if gate.shape[:2] != (batch, sequence):
            raise ValueError("diagonal SSM gate batch/sequence shape must match x")
        return gate.unsqueeze(2).expand(batch, sequence, channels, gate.shape[-1])
    if gate.ndim == 4:
        if gate.shape[:2] != (batch, sequence) or gate.shape[2] not in (1, channels):
            raise ValueError("diagonal SSM gate must use channel width one or C")
        return gate.expand(batch, sequence, channels, gate.shape[-1])
    raise ValueError("diagonal SSM gates use [B,T,N] or [B,T,C,N]")


__all__ = ["execute_diagonal_recurrence"]
