"""Whole-model training, prefill and decode timing, after explicit warmup."""
import statistics
import time
import torch
from .metrics import peak_flops, utilization
from .train import optimizers, update, synchronize
from .upstream import metadata


def run(model, cfg, device, iterations=10, warmup=3, peak=None):
    if iterations < 1 or warmup < 1:
        raise ValueError("positive warmup and iterations required")
    x = torch.randint(cfg["vocab_size"], (cfg["batch_size"] // cfg["seq_len"], cfg["seq_len"]), device=device)
    y = torch.randint(cfg["vocab_size"], x.shape, device=device)
    opts = optimizers(model, cfg, device)
    result = {"arch_type": cfg["arch_type"], "runtime": metadata(cfg["arch_type"]), "warmup": warmup, "iterations": iterations}
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for _ in range(warmup):
        update(model, opts, x, y, cfg)
    synchronize(device)
    times = []
    depths = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        loss = update(model, opts, x, y, cfg)
        synchronize(device)
        times.append(time.perf_counter() - t0)
        if cfg["arch_type"] == "gl_sdm":
            depths.append(model.last_training_depth)
    result["training"] = {**utilization(model, x.numel() * iterations, sum(times), peak_flops(peak) if torch.device(device).type == "cuda" else None, torch.cat(depths) if depths else None), "median_step_ms": 1000 * statistics.median(times), "loss": loss}
    del opts
    model.eval()
    for name in ("prefill", "decode"):
        times = []
        with torch.inference_mode():
            for i in range(warmup + iterations):
                # Rebuild outside the measurement so all samples start at the
                # same context length. Stateful decode workspaces are not reused
                # across independent requests.
                cache = model.new_cache(x.shape[0])
                if name == "decode":
                    _, cache = model.prefill(x, cache)
                synchronize(device)
                t0 = time.perf_counter()
                logits, _ = model.prefill(x, cache) if name == "prefill" else model.decode(y[:, :1], cache)
                synchronize(device)
                elapsed = time.perf_counter() - t0
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("non-finite benchmark logits")
                if i >= warmup:
                    times.append(elapsed)
            result[name] = {"median_ms": 1000 * statistics.median(times), "tokens_per_second": (x.numel() if name == "prefill" else x.shape[0]) * iterations / sum(times), "context_tokens": x.shape[1]}
            if cfg["arch_type"] == "gl_sdm":
                result[name].update(model.blocks[0].reasoning_metrics())
    if torch.device(device).type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    return result
