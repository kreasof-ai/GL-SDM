"""ATMA-style token batches, accumulation, validation curves and checkpoints."""
import math
import time
import torch
from . import checkpoint
from gl_sdm.experiments.data import available_steps, data_generator
from gl_sdm.experiments.evaluate import run_eval
from gl_sdm.experiments.metrics import peak_flops, utilization
from gl_sdm.model import create_model
from gl_sdm.experiments.provenance import metadata


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def optimizers(model, cfg, device):
    if cfg.get("optimizer", "adamw") == "atma_muon":
        from gl_sdm.experiments.muon import Muon
        scalar = [p for p in model.parameters() if p.ndim < 2]
        matrix = [p for p in model.blocks.parameters() if p.ndim >= 2]
        # SDM's learned memory is a sparse parameter bank, not a dense matrix
        # whose full SVD/polar update is justified. Keep it in AdamW.
        banks = [p for p in matrix if getattr(p, "_sdm_memory_bank", False)]
        matrix = [p for p in matrix if not getattr(p, "_sdm_memory_bank", False)]
        adam = torch.optim.AdamW([
            {"params": [model.embed.weight], "lr": 0.3},
            {"params": [model.proj.weight], "lr": 1 / 320},
            {"params": scalar + banks, "lr": 0.01},
        ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=torch.device(device).type == "cuda")
        result = [adam, Muon(matrix, lr=0.02, weight_decay=0.01)]
    else:
        result = [torch.optim.AdamW(model.parameters(), lr=cfg.get("adamw_lr", 3e-4),
                    betas=(cfg.get("adamw_beta1", 0.9), cfg.get("adamw_beta2", 0.95)),
                    eps=cfg.get("adamw_eps", 1e-15), weight_decay=cfg.get("adamw_weight_decay", 0.1),
                    fused=torch.device(device).type == "cuda")]
    assigned = [p for opt in result for group in opt.param_groups for p in group["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(model.parameters()):
        raise AssertionError("optimizer parameters must cover the model exactly once")
    for opt in result:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return result


def schedule(opts, cfg, step, steps):
    if cfg.get("optimizer") == "atma_muon":
        cooldown = cfg.get("cooldown_frac", 0.7)
        eta = min(1.0, (1 - step / steps) / cooldown)
    else:
        warmup = max(1, int(cfg.get("adamw_warmup_frac", 0.05) * steps))
        minimum = cfg.get("adamw_lr_min_frac", 0.1)
        eta = (step + 1) / warmup if step < warmup else minimum + (1 - minimum) * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1))) / 2
    for opt in opts:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


def update(model, opts, inputs, targets, cfg):
    if cfg.get("gl_cuda_graph", False):
        from gl_sdm.runtime.training_graph import TrainingGraph
        if not hasattr(model, "_training_graph"):
            model._training_graph = TrainingGraph(model, inputs, targets, cfg)
        return model._training_graph.update(opts, inputs, targets)
    mbs = cfg["mbs"]
    if inputs.shape[0] % mbs:
        raise ValueError("sequences per token batch must be divisible by mbs")
    total = 0.0
    depths = []
    for i in range(0, inputs.shape[0], mbs):
        loss, reg, align = model(inputs[i:i + mbs], targets[i:i + mbs])
        if model.cfg["arch_type"] == "gl_sdm":
            depths.append(model.blocks[0].last_depth)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        alpha = cfg.get("sigr_alpha", 0.0)
        # Keep ATMA's summed CE/backward scale identical across accumulation.
        objective = (1 - alpha) * loss + alpha * reg + cfg.get("auxiliary_loss_weight", cfg.get("dist_align_loss_weight", 0.0)) * align
        objective.backward()
        total += loss.detach().item()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite training gradient; run aborted")
    for opt in opts:
        opt.step()
    model.zero_grad(set_to_none=True)
    if depths:
        model.last_training_depth = torch.cat(depths).flatten()
    return total / targets.numel()


@torch.inference_mode()
def validate(model, cfg, device):
    model.eval()
    total, count = 0.0, 0
    gen = data_generator(cfg["val_data"], cfg["mbs"] * cfg["seq_len"], cfg["seq_len"], device)
    tokens = cfg["val_tokens"]
    if tokens % (cfg["mbs"] * cfg["seq_len"]):
        raise ValueError("val_tokens must be divisible by microbatch tokens")
    for _ in range(tokens // (cfg["mbs"] * cfg["seq_len"])):
        x, y = next(gen)
        loss, _, _ = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation loss")
        total, count = total + loss.item(), count + y.numel()
    model.train()
    return total / count


def run(cfg, device, directory, emit, resume=None, peak=None):
    seed = cfg.get("seed", 1234)
    torch.manual_seed(seed)
    steps = available_steps(cfg["train_data"], cfg["batch_size"], cfg.get("num_chunks", 1))
    steps = min(steps, cfg.get("max_steps", steps))
    if steps < 1:
        raise ValueError("no complete training batches")
    model, payload = checkpoint.load(resume, device) if resume else (create_model(cfg).to(device), None)
    if resume and model.cfg != cfg:
        raise ValueError("resume config must match the checkpoint exactly")
    opts = optimizers(model, cfg, device)
    start, data_batches = 0, 0
    if payload:
        for opt, state in zip(opts, payload["optimizers"], strict=True):
            opt.load_state_dict(state)
        start, data_batches = payload["step"], payload["data_batches"]
        torch.set_rng_state(payload["rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    else:
        # ATMA initializes the head and residual output projections to zero.
        with torch.no_grad():
            model.proj.weight.zero_()
            for block in model.blocks:
                block.mlp.proj.weight.zero_()
                output = getattr(block.attn, "o_proj", getattr(block.attn, "Wo", getattr(block.attn, "proj", None)))
                output.weight.zero_()
    emit("ABLATION_CONFIG_JSON", {**cfg, "attn_type": cfg["arch_type"], "runtime": metadata(cfg["arch_type"], cfg)})
    loader = data_generator(cfg["train_data"], cfg["batch_size"], cfg["seq_len"], device, cfg.get("num_chunks", 1))
    for _ in range(data_batches):
        next(loader)
    peak = peak_flops(peak) if torch.device(device).type == "cuda" else None
    curve = []
    interval = cfg.get("val_freq", max(1, min(125, steps // 4)))
    seconds, measured_tokens = 0.0, 0
    measured_depth = []
    training_time = 0.0
    for step in range(start, steps + 1):
        if step == start or step == steps or step % interval == 0:
            val_loss = validate(model, cfg, device)
            row = {"step": step, "val_loss": val_loss, "wall_s": training_time}
            if measured_tokens:
                row.update(utilization(model, measured_tokens, seconds, peak, torch.cat(measured_depth) if measured_depth else None))
                row["mfu"] = row["mfu_6nd_pct"]
                row["step_ms"] = 1000 * seconds / (measured_tokens / cfg["batch_size"])
            curve.append(row)
            emit("VALIDATION_JSON", row)
            checkpoint.save(model, directory, opts, step, data_batches)
            seconds, measured_tokens = 0.0, 0
            measured_depth = []
        if step == steps:
            break
        x, y = next(loader)
        data_batches += 1
        schedule(opts, cfg, step, steps)
        synchronize(device)
        t0 = time.perf_counter()
        train_loss = update(model, opts, x, y, cfg)
        synchronize(device)
        elapsed = time.perf_counter() - t0
        training_time += elapsed
        # First update includes cold kernel/optimizer initialization. Exclude it
        # from steady-state throughput, while recording its full duration.
        if step > start:
            seconds += elapsed
            measured_tokens += y.numel()
            if cfg["arch_type"] == "gl_sdm":
                measured_depth.append(model.last_training_depth)
        emit("TRAIN_STEP_JSON", {"step": step + 1, "train_loss": train_loss, "seconds": elapsed})
    emit("ABLATION_CURVE_JSON", curve)
    if cfg.get("eval_after_train", False):
        emit("ABLATION_EVAL_JSON", {**run_eval(model, cfg, device), "mfu_final": curve[-1].get("mfu"), "train_elapsed_s": training_time, "num_params": sum(p.numel() for p in model.parameters()), "train_steps": steps})
    return model
