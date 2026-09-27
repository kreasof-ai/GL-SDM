"""Reproduce the historical SDM shape, timings, FLOP accounting and decay bug.

Run from projects/urm after provisioning CUDA, with a clean historical checkout:
  python extra/benchmark_sdm.py --historical-root /tmp/urm-sdm-history \
      --out results/sdm-optimization/historical.json
GPU measurements must run alone. Compilation and warmup are excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.profiler import ProfilerActivity, profile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))
from urm.backends.triton.k3.sparse_state import chunked_sparse_state_update
from extra.comparators.sdm.cuda import load_pinned_sdm, sdm_cuda_identity

HISTORICAL_COMMIT = "dd45e66a42fbaf4e570638e175ccac0599aeddfc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True,
                        help="clean repository checkout at dd45e66")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    checkout = args.historical_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                       text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"],
                                    text=True).strip()
    if revision != HISTORICAL_COMMIT or dirty:
        raise RuntimeError("historical reproduction requires a clean checkout at " + HISTORICAL_COMMIT)
    import importlib.util
    import types
    # Execute the untouched historical file with its original enum import in a
    # private namespace; no historical code is used by the current provider.
    semantic = types.ModuleType("urm.compiler.semantic")
    from urm.ir.program import SparseReadTiming
    semantic.SparseReadTiming = SparseReadTiming
    source = checkout / "projects/urm/src/urm/backends/dual_form_sdm.py"
    spec = importlib.util.spec_from_file_location("historical_dual_form_sdm", source)
    old_module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(semantic.__name__)
    sys.modules[semantic.__name__] = semantic
    try:
        spec.loader.exec_module(old_module)
    finally:
        if previous is None:
            sys.modules.pop(semantic.__name__)
        else:
            sys.modules[semantic.__name__] = previous
    chunked_dual_form_sdm = old_module.chunked_dual_form_sdm
    kernel = load_pinned_sdm()
    identity = sdm_cuda_identity()
    historical = torch.compile(chunked_dual_form_sdm, dynamic=False, fullgraph=True)
    p, t, s, d, w, r, c = 12, 1024, 4096, 64, 64, 64, 256
    torch.manual_seed(2026)
    wi = torch.rand(p, t, s, device="cuda").topk(w, largest=False).indices.sort(-1).values
    ri = torch.rand(p, t, s, device="cuda").topk(r, largest=False).indices.sort(-1).values
    memory = torch.randn(p, s, d, device="cuda", dtype=torch.bfloat16) * .1
    ww = torch.rand(p, t, w, device="cuda", dtype=torch.bfloat16)
    ww = ww / ww.sum(-1, keepdim=True)
    rw = torch.rand(p, t, r, device="cuda", dtype=torch.bfloat16)
    rw = rw / rw.sum(-1, keepdim=True)
    values = torch.randn(p, t, d, device="cuda", dtype=torch.bfloat16) * .1
    beta = torch.rand(p, t, 1, device="cuda", dtype=torch.bfloat16) * .5
    decay = -torch.rand(p, t, 1, device="cuda", dtype=torch.bfloat16) * .05
    leaves = [x.detach().clone().requires_grad_() for x in (memory, rw, ww, values, beta, decay)]
    memory, rw, ww, values, beta, decay = leaves
    kwargs = dict(write_indices=wi, write_weights=ww, values=values, beta=beta,
                  log_decay=decay, chunk_size=c)

    def old(compiled=True):
        fn = historical if compiled else chunked_dual_form_sdm
        return fn(memory, ri, rw, **kwargs)

    def corrected(compiled=True):
        return chunked_sparse_state_update(memory, ri, rw, **kwargs, compiled=compiled)

    # Historical T is already divisible by C; launch the original flattened API.
    offsets = torch.arange(p, device="cuda").view(p, 1, 1) * s
    wglobal = (wi + offsets).reshape(p * t, w)
    rglobal = (ri + offsets).reshape(p * t, r)

    def upstream():
        workspace = memory.clone().reshape(p * s, d)
        with torch.autocast("cuda", enabled=False):
            output, _ = kernel.apply(workspace, wglobal, ww.reshape(p * t, w),
                values.reshape(p * t, d), beta.reshape(p * t, 1), decay.reshape(p * t, 1),
                rglobal, rw.reshape(p * t, r), c, True, s, p, False, "none", None)
        return output.reshape(p, t, d), workspace

    def step(fn):
        output, state = fn()
        output.float().square().sum().backward()
        for leaf in leaves:
            leaf.grad = None
        del output, state

    results = {}
    for name, fn in (("historical_eager", lambda: old(False)),
                     ("historical_compiled", old), ("pinned_cuda", upstream),
                     ("corrected_compiled", corrected)):
        print("Warmup", name, flush=True)
        for _ in range(args.warmup):
            step(fn)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(args.iterations):
            step(fn)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iterations
        results[name] = {"forward_backward_ms": ms,
                        "historical_522_gflop_proxy_percent": 522 / ms / 66.166 * 100,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
        print(name, results[name], flush=True)

    for name, eager, compiled_name in (("historical_eager", lambda: old(False), "historical_compiled"),
                                       ("corrected_eager", lambda: corrected(False), "corrected_compiled")):
        with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=True) as prof:
            step(eager)
        flops = sum(event.flops for event in prof.key_averages() if event.flops)
        results[name + "_profiled_gemm_flops"] = flops
        results[compiled_name]["eager_gemm_work_percent_at_compiled_latency"] = (
            flops / 1e9 / results[compiled_name]["forward_backward_ms"] / 66.166 * 100)

    y_old, m_old = old(False)
    y_up, m_up = upstream()
    ignored = torch.autograd.grad(y_old.float().square().sum(), decay, allow_unused=True)[0]
    results["historical_decay_check"] = {
        "reading_max_abs_vs_cuda": float((y_old.detach() - y_up.detach()).abs().max()),
        "state_max_abs_vs_cuda": float((m_old.detach() - m_up.detach().reshape_as(m_old)).abs().max()),
        "decay_gradient": None if ignored is None else float(ignored.norm()),
    }
    y_new, m_new = corrected(False)
    results["corrected_forward_check"] = {
        "reading_max_abs_vs_cuda": float((y_new.detach() - y_up.detach()).abs().max()),
        "state_max_abs_vs_cuda": float((m_new.detach() - m_up.detach().reshape_as(m_new)).abs().max()),
        "decay_gradient_norm": float(torch.autograd.grad(y_new.float().square().sum(), decay)[0].norm()),
    }
    results.update(shapes=dict(parallel=p, sequence=t, slots=s, value_dim=d, reads=r, writes=w, chunk=c),
                   historical_commit=revision, upstream=identity,
                   torch=torch.__version__, device=torch.cuda.get_device_name(),
                   seed=2026, iterations=args.iterations, warmup=args.warmup,
                   timing="CUDA events, forward + reading-loss backward, all initial/learned operands require gradients",
                   flop_policy="522 GFLOP proxy is historical only; profiler GEMMs exclude solves, reductions and fused CUDA/Triton work")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
