"""Profile one explicit microbatch, without optimizer or throughput claims."""
import argparse
import json
from pathlib import Path
import time
import torch
from gl_sdm.model import create_model
from gl_sdm.experiments.provenance import metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("projects/gl-sdm/configs/gl_sdm.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.manual_seed(cfg.get("seed", 1234))
    model = create_model(cfg).cuda().train()
    x = torch.randint(cfg["vocab_size"], (cfg["mbs"], cfg["seq_len"]), device="cuda")
    y = torch.randint_like(x, cfg["vocab_size"])
    # Warm compilation and release gradients before instrumenting allocations.
    model(x, y)[0].backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as profile:
        model(x, y)[0].backward()
        torch.cuda.synchronize()
    kernels = [event for event in profile.key_averages() if str(event.device_type) == "DeviceType.CUDA"]
    total = sum(event.self_device_time_total for event in kernels)
    kernel_rows = [{"name": event.key, "calls": event.count,
                    "device_ms": event.self_device_time_total / 1000,
                    "device_time_pct": 100 * event.self_device_time_total / total}
                   for event in sorted(kernels, key=lambda e: e.self_device_time_total, reverse=True)[:20]]
    allocation_rows = [{"name": event.key, "input_shapes": event.input_shapes,
                        "calls": event.count, "cumulative_allocated_gib": event.self_device_memory_usage / 1024**3}
                       for event in sorted(profile.key_averages(group_by_input_shape=True),
                                           key=lambda e: e.self_device_memory_usage, reverse=True)[:12]]
    memory_names = {"aten::add", "aten::add_", "aten::fill_", "aten::zero_",
                    "aten::zeros_like", "aten::copy_", "aten::constant_pad_nd"}
    memory_rows = [{"name": event.key, "input_shapes": event.input_shapes,
                   "calls": event.count, "inclusive_device_ms": event.device_time_total / 1000}
                  for event in sorted(profile.key_averages(group_by_input_shape=True),
                                      key=lambda e: e.device_time_total, reverse=True)
                  if event.key in memory_names][:20]
    result = {"scope": "one training microbatch, forward/backward; no clipping or optimizer",
              "batch": x.shape[0], "sequence": x.shape[1], "config": cfg,
              "runtime": metadata(cfg["arch_type"], cfg),
              "profiled_wall_seconds": time.perf_counter() - start,
              "kernel_device_ms": total / 1000, "top_kernels": kernel_rows,
              "top_allocations": allocation_rows,
              "memory_operators_by_shape": memory_rows,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"kernel_device_ms": result["kernel_device_ms"], "top_kernels": kernel_rows[:5]}), flush=True)


if __name__ == "__main__":
    main()
