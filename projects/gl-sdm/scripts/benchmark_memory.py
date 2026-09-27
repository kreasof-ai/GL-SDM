"""Compare the same GL-SDM token transaction at a larger bank geometry.

Includes snapshot reads, R colliding proposals, one commit and gradients of
initial state and every learned operand. This is an operator timing, not MFU
or a decoder measurement. Run alone on the GPU after installing GL-SDM + URM.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from gl_sdm.memory import MemoryView, read, propose_write, merge, commit
from gl_sdm.experiments.provenance import metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.slots < 8 or args.iterations < 1 or args.warmup < 1:
        parser.error("slots >=8 and positive warmup/iterations required")
    B, H, D, K, R = 2, 8, 64, 8, 8
    torch.manual_seed(1234)
    device = "cuda"
    memory = torch.randn(B, H, args.slots, D, device=device)
    weights = torch.randn(R, B, H, K, device=device).softmax(-1)
    targets = torch.randn(R, B, H, D, device=device)
    beta = 0.1 + 0.8 * torch.rand(R, B, H, 1, device=device)
    decay = -0.05 - 0.15 * torch.rand(R, B, H, 1, device=device)
    mass = torch.full((R, B), 1 / R, device=device)
    inputs = [t.requires_grad_() for t in (memory, weights, targets, beta, decay, mass)]
    requests = torch.arange(B, device=device)
    # Repeat the same unique within-step addresses across all R proposals.
    indices = torch.rand(B, H, args.slots, device=device).argsort(-1)[..., :K].sort(-1).values
    read_probe = torch.randn(R, B, H, D, device=device) * 1e-3
    state_probe = torch.randn_like(memory) * 1e-3

    def execute(backend):
        view = MemoryView(memory)
        readings, proposals = [], []
        for r in range(R):
            readings.append(read(view, requests, indices, weights[r], backend=backend))
            proposals.append(propose_write(view, requests, indices, weights[r], targets[r],
                                            beta[r], decay[r], mass[r], backend=backend))
        updated = commit(view, merge(view, proposals), backend=backend)
        return torch.stack(readings), updated.values

    def objective(readings, state):
        return (readings * read_probe).sum() + (state * state_probe).sum()

    checked = {}
    for backend in ("torch", "urm"):
        readings, state = execute(backend)
        gradients = torch.autograd.grad(objective(readings, state), inputs)
        checked[backend] = [readings.detach().cpu(), state.detach().cpu(), *(g.cpu() for g in gradients)]
    errors = {}
    for name, a, b in zip(("readings", "state", "initial_state_grad", "weights_grad", "targets_grad", "beta_grad", "decay_grad", "mass_grad"),
                           checked["urm"], checked["torch"], strict=True):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        errors[name] = (a - b).abs().max().item()
    del readings, state, gradients, checked
    result = {"config": {"batch": B, "heads": H, "slots": args.slots, "dim": D, "routes": K,
                          "reasoning_steps": R, "dtype": "float32", "seed": 1234,
                          "transaction_tokens": 1, "cross_step_collision": "all write addresses repeated"},
              "scope": "token transaction forward/backward; no model projections or optimizer",
              "warmup": args.warmup, "iterations": args.iterations, "correctness_max_abs": errors}
    for backend in ("torch", "urm"):
        saved = {}
        def pack(t):
            storage = t.untyped_storage()
            saved[storage.data_ptr()] = storage.nbytes()
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            readings, state = execute(backend)
        saved_bytes = sum(saved.values())
        retains_bank = memory.untyped_storage().data_ptr() in saved
        del readings, state
        times, allocations = [], []
        for i in range(args.warmup + args.iterations):
            for t in inputs:
                t.grad = None
            torch.cuda.synchronize()
            if i == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            readings, state = execute(backend)
            objective(readings, state).backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            del readings, state
            if i >= args.warmup:
                times.append(1000 * elapsed)
                allocations.append(torch.cuda.memory_allocated() / 1024**2)
        result[backend] = {"runtime": metadata("gl_sdm", {"gl_memory_backend": backend}),
                           "forward_backward_ms": statistics.median(times), "samples_ms": times,
                           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                           "step_end_allocated_mib": allocations, "saved_operator_storage_bytes": saved_bytes,
                           "retains_initializer_storage_for_backward": retains_bank}
    result["speedup"] = result["torch"]["forward_backward_ms"] / result["urm"]["forward_backward_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
