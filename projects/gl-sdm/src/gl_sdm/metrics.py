"""Explicit 6ND utilization. No attention proxy is charged to recurrent models."""
import torch


def parameter_counts(model):
    memory = sum(p.numel() for p in model.parameters() if getattr(p, "_sdm_memory_bank", False))
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


def utilization(model, tokens, seconds, peak, depth=None):
    counts = parameter_counts(model)
    reasoning = model.blocks[0].reasoning_metrics(depth) if model.cfg["arch_type"] == "gl_sdm" else {}
    interpretation = {"mfu_interpretation": "unique-parameter 6ND; repeated applications of tied reasoner weights are not counted"} if reasoning else {}
    return {**counts, **reasoning, **interpretation, "tokens": tokens, "seconds": seconds, "tok_s": tokens / seconds,
            "mfu_6nd_pct": None if peak is None else 100 * 6 * counts["active_params"] * tokens / seconds / peak,
            "peak_tflops": None if peak is None else peak / 1e12,
            "mfu_formula": "6 * active_params * tokens / seconds / peak_flops; excludes sparse learned memory bank and state work"}
