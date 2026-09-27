"""Compare identical chunk models with PyTorch/URM memory, without capture."""
import argparse
import gc
import json
from pathlib import Path
import torch
from gl_sdm.model import create_model
from gl_sdm.benchmark import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("projects/gl-sdm/configs/gl_sdm_chunk_r4.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg["gl_cuda_graph"] = False  # PyTorch unique-address reduction has dynamic output shape.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for backend in ("torch", "urm"):
        current = {**cfg, "gl_memory_backend": backend}
        torch.manual_seed(current.get("seed", 1234))
        model = create_model(current).cuda()
        result = run(model, current, "cuda", args.iterations, args.warmup)
        path = args.output_dir / f"{cfg['run_id']}_{backend}_control.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(backend, result["training"]["median_step_ms"], result["training"]["mfu_6nd_pct"], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
