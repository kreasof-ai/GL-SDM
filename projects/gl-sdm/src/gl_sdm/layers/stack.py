"""Execute each physical layer once per chunk, then commit the shared bank.

Local KV history survives memory commits. All global layers read and propose
against the chunk-start snapshot; no writes become visible between layers.
"""
import torch
import torch.nn.functional as F
from gl_sdm.memory import merge, commit
from .global_layer import GlobalLayer


def make_layers(cfg):
    from gl_sdm.baselines.block import Block
    pattern = cfg["gl_layer_pattern"]
    count = cfg["num_hidden_layers"]
    if not pattern or any(kind not in {"local", "global"} for kind in pattern) or "global" not in pattern or count % len(pattern):
        raise ValueError("gl_layer_pattern must include global and repeat to cover num_hidden_layers")
    if cfg.get("gl_max_steps", 1) != 1 or cfg.get("gl_reasoning", "fixed") != "fixed":
        raise ValueError("the physical GL-SDM stack has no reasoning loops")
    if cfg.get("gl_chunk_size", 512) < 1 or cfg.get("gl_local_window", 512) < 1:
        raise ValueError("positive GL-SDM chunk and local window required")
    if cfg.get("gl_cuda_graph", False):
        raise ValueError("physical GL-SDM stack currently uses ordinary training, not legacy loop CUDA graphs")
    local_cfg = {**cfg, "arch_type": "transformer", "attention_window": cfg.get("gl_local_window", 512)}
    return [GlobalLayer(cfg) if pattern[i % len(pattern)] == "global" else Block(local_cfg, i)
            for i in range(count)]


def forward(model, inputs, cache=None):
    B, T, _ = inputs.shape
    if min(B, T) < 1:
        raise ValueError("GL-SDM requires a nonempty token batch")
    retain_state = cache is not None
    cache = model.bank.new_cache(B) if cache is None else cache
    model.bank.validate_cache(cache, B)
    outputs = []
    reg, align = inputs.new_zeros(()), inputs.new_zeros(())
    C = model.cfg.get("gl_chunk_size", 512)
    backend = model.cfg.get("gl_memory_backend", "torch")
    position = 0
    while position < T:
        offset = cache.tokens % C
        length = min(C - offset, T - position)
        snapshot = cache.view
        global_layer = next(b for b in model.blocks if isinstance(b, GlobalLayer))
        reference = global_layer.attn.reference
        # Reuse the existing URM vector read schedule across all global layers.
        # Padding selects URM's faster backward schedule. Serving has no
        # backward and reads width 64 directly, avoiding a second full bank.
        physical = F.pad(snapshot.values, (0, 64)) if (torch.is_grad_enabled() and backend == "urm"
            and snapshot.values.shape[-1] == 64 and not reference) else None
        writes_needed = retain_state or position + length < T
        proposals = list(cache.pending)
        x = inputs[:, position:position + length]
        for i, block in enumerate(model.blocks):
            if isinstance(block, GlobalLayer):
                x, r, a = block(x, snapshot, physical)
                if writes_needed:
                    proposals.append(block.propose(x, snapshot, offset, i, C, len(model.blocks), physical))
            else:
                x, r, a = block(x, cache.local.setdefault(i, {}))
            reg, align = reg + r * (length / T), align + a
        cache.tokens += length
        if cache.tokens % C == 0 and writes_needed:
            cache.view = commit(snapshot, merge(snapshot, proposals), backend="torch" if reference else backend)
            cache.pending = ()
        else:
            cache.pending = tuple(proposals)
        outputs.append(x)
        position += length
    return torch.cat(outputs, 1), reg / len(model.blocks), align / len(model.blocks)
