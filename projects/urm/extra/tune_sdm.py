"""One isolated full-model SDM schedule/chunk-size diagnostic.

Run separate processes for different candidates; use train.sweep/train.upstream
for the canonical ten-step campaign. This tool never edits the registry.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT, PROJECT / "src"):
    sys.path.insert(0, str(path))

from architectures.sdm_memory import SparseDeltaMemoryLayer
from train.data import data_generator, get_data
from train.harness import TrainConfig, train
from train.registry import get_mixer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("native", "torch-chunked", "upstream-cuda"), required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def builder(width, heads, dim, intent, target="native", batch_size=None):
        return SparseDeltaMemoryLayer(width, heads, dim, 256, 8, 8, batch_size,
            target="native", intent=intent, execution=args.execution, chunk_size=args.chunk_size)

    cfg = TrainConfig(mixer="sdm", layers=9, batch_tokens=8192, microbatch_tokens=8192,
                      steps=args.steps)
    spec = replace(get_mixer("sdm"), builder=builder)
    get_data("finewebedu_train_000001.bin")
    data = data_generator("data/finewebedu10B/finewebedu_train_*.bin",
                          cfg.microbatch_tokens, cfg.sequence_length)
    result = train(cfg, spec, data).to_dict()
    result["diagnostic"] = True
    result["sdm_execution"] = {"schedule": args.execution, "chunk_size": args.chunk_size,
                               "routes": "native-public", "compile_state": args.execution == "torch-chunked"}
    if args.execution == "upstream-cuda":
        from extra.comparators.sdm.cuda import sdm_cuda_identity
        result["sdm_source_identity"] = sdm_cuda_identity()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
