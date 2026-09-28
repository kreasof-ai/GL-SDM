"""Record strict BF16 differences and independently verify FP32 integration.

Keep a failed BF16 comparison visible: this diagnostic does not relax the
production verification tolerances or convert a failure into a passing result.
"""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch
import torch
from gl_sdm.model import create_model
from gl_sdm.experiments.provenance import metadata
from gl_sdm.experiments.verify import check, compare


def difference(actual, expected, label, atol, rtol):
    delta = actual.float() - expected.float()
    result = {"max_abs": delta.abs().max().item(),
              "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()}
    try:
        compare(actual, expected, label, atol=atol, rtol=rtol)
        result["status"] = "passed"
    except AssertionError as error:
        result.update(status="failed", failure=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("projects/gl-sdm/configs/gl_sdm_chunk_r4.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=1025)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg.update(vocab_size=args.vocab_size, gl_cuda_graph=False, dtype="bfloat16")
    if args.chunk_size is not None:
        cfg["gl_chunk_size"] = args.chunk_size
    torch.manual_seed(1234)
    model = create_model(cfg).cuda()
    oracle = copy.deepcopy(model)
    inputs = torch.randint(cfg["vocab_size"], (2, args.length), device="cuda")
    targets = torch.randint_like(inputs, cfg["vocab_size"])
    from gl_sdm.memory import routing
    original_read = routing.routed_read
    observed = []
    def observe(block, memory, scores, *args):
        result = original_read(block, memory, scores, *args)
        if scores.ndim == 4 and len(observed) < cfg["gl_max_steps"]:
            observed.append((scores.detach().clone(), result[0].detach().clone(), result[1].detach().clone()))
        return result
    with patch.object(routing, "routed_read", observe):
        actual = model.head(model.hidden(inputs)[0])
        native_reads, observed = observed, []
        with oracle.reference():
            expected = oracle.head(oracle.hidden(inputs)[0])
    first_chunk_routes = [
        {"step": step + 1, "score_max_abs": (a[0] - b[0]).abs().max().item(),
         "read_max_abs": (a[1] - b[1]).abs().max().item(),
         "selected_address_disagreements": (a[2] != b[2]).sum().item()}
        for step, (a, b) in enumerate(zip(native_reads, observed, strict=True))]
    output = difference(actual, expected, "reference logits", .03, .03)
    torch.nn.functional.cross_entropy(actual.flatten(0, 1), targets.flatten()).backward()
    torch.nn.functional.cross_entropy(expected.flatten(0, 1), targets.flatten()).backward()
    gradients = {}
    for (name, p), (other, q) in zip(model.named_parameters(), oracle.named_parameters(), strict=True):
        assert name == other and p.grad is not None and q.grad is not None
        gradients[name] = difference(p.grad, q.grad, name, .002, .12)
    del actual, expected, model, oracle
    torch.cuda.empty_cache()
    print("BF16 differences recorded; checking FP32 independently", flush=True)
    try:
        fp32 = check({**cfg, "dtype": "float32"}, "cuda", args.length)
    except AssertionError as error:
        fp32 = {"status": "failed", "failure": str(error)}
    passed = output["status"] == "passed" and all(g["status"] == "passed" for g in gradients.values()) and fp32["status"] == "passed"
    result = {"status": "passed" if passed else "reference_failed", "config": cfg,
              "runtime": metadata("gl_sdm", cfg), "batch": 2, "length": args.length,
              "bf16_output": output, "bf16_parameter_gradients": gradients, "fp32_check": fp32,
              "bf16_first_chunk_routes": first_chunk_routes,
              "note": "Strict BF16 and FP32 gates are unchanged. Vocabulary is reduced for this diagnostic; all effective dimensions are saved in config."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"], output, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
