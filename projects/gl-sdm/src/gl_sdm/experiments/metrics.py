"""6ND estimates with execution-weighted parameters for the tied GL reasoner.

This remains a parameter-based FLOP estimate, not an instruction counter. The
6ND convention includes embeddings/biases/norm parameters and excludes sparse
state arithmetic and attention FLOPs. Unique parameter counts describe model
capacity; repeated execution determines GL-SDM's effective N for utilization.
"""
import torch


def parameter_counts(model):
    banks = {id(p) for p in model.parameters() if getattr(p, "_sdm_memory_bank", False)}
    if model.cfg["arch_type"] == "gl_sdm":
        # Parameter deepcopy does not retain arbitrary Python attributes.
        banks.update(id(block.bank.memory) for block in model.blocks)
    memory = sum(p.numel() for p in model.parameters() if id(p) in banks)
    total = sum(p.numel() for p in model.parameters())
    return {"num_params": total, "active_params": total - memory, "memory_params": memory}


def peak_flops(explicit=None):
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("peak_tflops must be positive")
        return explicit * 1e12
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    for key, value in (("A10G", 70), ("A100", 312), ("H100", 989), ("H200", 989), ("L40S", 362), ("L4", 121), ("T4", 65)):
        if key in name:
            return value * 1e12
    raise ValueError(f"specify --peak-tflops for GPU {name}")


def gl_parameter_groups(model):
    """Partition unique parameters by their use in a training forward pass."""
    block = model.blocks[0]
    reasoner = [block.attn.q, block.attn.read_norm, block.attn.proj,
                block.norm1, block.norm2, block.mlp]
    if block.halt is not None:
        reasoner.append(block.halt)
    writes = [block.attn.k, block.attn.v, block.attn.beta, block.attn.decay]
    read_ids = {id(p) for module in reasoner for p in module.parameters()}
    write_ids = {id(p) for module in writes for p in module.parameters()}
    groups = {"once_params": 0, "reasoner_params": 0, "write_params": 0}
    for p in model.parameters():
        if p is block.bank.memory or getattr(p, "_sdm_memory_bank", False):
            continue
        name = "reasoner_params" if id(p) in read_ids else "write_params" if id(p) in write_ids else "once_params"
        groups[name] += p.numel()
    return groups


def gl_execution_counts(model, tokens, depth=None):
    """Count active token/passes, retaining position for terminal write pruning.

    Used only for complete, fresh-cache training sequences. Serving metrics do
    not call this function because their pending cache changes the write clock.
    """
    block = model.blocks[0]
    observed = block.last_depth if depth is None else depth
    if observed is None:
        if block.reasoning != "fixed":
            raise ValueError("adaptive MFU requires observed per-token reasoning depth")
        observed = torch.full((tokens,), block.max_steps, dtype=torch.int64)
    observed = observed.detach().cpu().flatten()
    if observed.numel() != tokens:
        raise ValueError("MFU depth telemetry must contain one entry per measured token")
    if not torch.isfinite(observed).all() or not ((observed >= 1) & (observed <= block.max_steps) & (observed == observed.trunc())).all():
        raise ValueError("MFU reasoning depths must be integers within the configured limit")
    if block.reasoning == "fixed" and not observed.eq(block.max_steps).all():
        raise ValueError("fixed-depth MFU telemetry disagrees with gl_max_steps")
    calls = int(observed.sum().item())
    write_calls = calls
    if block.chunk_size > 1:
        length = model.cfg.get("seq_len")
        if length is None or length < 1 or tokens % length:
            raise ValueError("chunk MFU requires complete sequences and a positive seq_len")
        # The last (possibly partial) chunk has no outgoing-state consumer.
        prefix = ((length - 1) // block.chunk_size) * block.chunk_size
        write_calls = int(observed.reshape(-1, length)[:, :prefix].sum().item())
    return observed, {"reasoner_token_passes": calls, "write_token_passes": write_calls}


def utilization(model, tokens, seconds, peak, depth=None):
    if tokens < 1 or seconds <= 0:
        raise ValueError("positive token count and elapsed time required for MFU")
    counts = parameter_counts(model)
    extra = {}
    parameter_uses = counts["active_params"] * tokens
    if model.cfg["arch_type"] == "gl_sdm":
        observed, calls = gl_execution_counts(model, tokens, depth)
        groups = gl_parameter_groups(model)
        parameter_uses = (groups["once_params"] * tokens
                          + groups["reasoner_params"] * calls["reasoner_token_passes"]
                          + groups["write_params"] * calls["write_token_passes"])
        extra = {**model.blocks[0].reasoning_metrics(observed), **groups, **calls}
    flops = 6 * parameter_uses
    return {**counts, **extra, "tokens": tokens, "seconds": seconds, "tok_s": tokens / seconds,
            "effective_active_params": parameter_uses / tokens, "estimated_training_flops_6nd": flops,
            "mfu_6nd_pct": None if peak is None else 100 * flops / seconds / peak,
            "unique_parameter_6nd_pct": None if peak is None else 100 * 6 * counts["active_params"] * tokens / seconds / peak,
            "unique_parameter_6nd_interpretation": "capacity-normalized throughput proxy; does not count tied-weight reuse",
            "peak_tflops": None if peak is None else peak / 1e12,
            "mfu_interpretation": "execution-weighted 6ND estimate; preserves the parameter-count convention, excludes attention and sparse state FLOPs",
            "mfu_formula": "6 * (tokens * once_params + reasoner_token_passes * reasoner_params + write_token_passes * write_params) / seconds / peak_flops" if extra else "6 * active_params * tokens / seconds / peak_flops"}
