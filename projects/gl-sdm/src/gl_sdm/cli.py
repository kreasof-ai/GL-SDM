import argparse
import json
from pathlib import Path
import traceback
import torch


def main():
    parser = argparse.ArgumentParser(description="GL-SDM and ATMA-compatible Transformer, upstream CUDA SDM and FLA GDN2 baselines")
    parser.add_argument("command", choices=("train", "infer", "eval", "verify", "benchmark"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--length", type=int, help="reference check token count; defaults to at least one full transaction plus a continuation")
    parser.add_argument("--verify-batch-size", type=int, default=2, help="explicit reference-check batch size")
    parser.add_argument("--memory-backend", choices=("torch", "urm"), help="explicit GL-SDM control for a config-based train/verify/benchmark run")
    parser.add_argument("--prompt", default="The research question is")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()
    if args.memory_backend and (args.command not in {"train", "verify", "benchmark"} or args.checkpoint):
        parser.error("--memory-backend requires config-based train, verify or benchmark")
    def read_config():
        cfg = json.loads(args.config.read_text())
        if args.memory_backend:
            if cfg["arch_type"] != "gl_sdm":
                parser.error("--memory-backend applies only to GL-SDM")
            cfg["gl_memory_backend"] = args.memory_backend
        return cfg
    fh = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        fh = args.log.open("w", buffering=1)
    def emit(name, value):
        block = f"\n==={name}===\n{json.dumps(value, allow_nan=False)}\n===END===\n"
        print(block, flush=True)
        if fh:
            fh.write(block)
    try:
        from gl_sdm.experiments.checkpoint import load
        from .model import create_model
        if args.command == "train":
            if args.config is None or args.output is None:
                parser.error("train requires --config and --output checkpoint directory")
            from gl_sdm.experiments.train import run
            cfg = read_config()
            run(cfg, args.device, args.output, emit, args.checkpoint, args.peak_tflops)
            return
        if args.command == "verify":
            if args.config is None:
                parser.error("verify requires --config")
            from gl_sdm.experiments.verify import check
            cfg = read_config()
            cfg["verify_batch_size"] = args.verify_batch_size
            length = args.length or max(65, cfg.get("gl_chunk_size", 1) + 1)
            result = check(cfg, args.device, length)
            emit("REFERENCE_CHECK_JSON", result)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            return
        if args.checkpoint:
            model, _ = load(args.checkpoint, args.device)
            cfg = model.cfg
        elif args.command == "benchmark" and args.config:
            cfg = read_config()
            torch.manual_seed(cfg.get("seed", 1234))
            model = create_model(cfg).to(args.device)
        else:
            parser.error("infer/eval require --checkpoint; benchmark requires --config or --checkpoint")
        if args.command == "benchmark":
            from gl_sdm.experiments.benchmark import run
            result = run(model, cfg, args.device, args.iterations, args.warmup, args.peak_tflops)
            emit("BENCHMARK_JSON", result)
        elif args.command == "eval":
            from gl_sdm.experiments.evaluate import run_eval
            if args.config:
                cfg = {**cfg, **json.loads(args.config.read_text())}
            result = run_eval(model, cfg, args.device)
            emit("ABLATION_EVAL_JSON", result)
        else:
            from gl_sdm.experiments.inference import generate
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer_name", "gpt2"))
            inputs = torch.tensor(tokenizer.encode(args.prompt), device=args.device)[None]
            ids = generate(model, inputs, args.tokens, args.temperature, args.top_k, tokenizer.eos_token_id)
            result = {"text": tokenizer.decode(ids[0].tolist()), "tokens": ids[0].tolist()}
            emit("INFERENCE_JSON", result)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except Exception:
        emit("ABLATION_ERROR_JSON", {"error": traceback.format_exc()})
        raise
    finally:
        if fh:
            fh.close()


if __name__ == "__main__":
    main()
