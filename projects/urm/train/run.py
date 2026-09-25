"""Top-level driver for the URM training harness (ATMA-pattern entry point).

Trains one registered public-path mixer on the finewebedu shards (or a synthetic
stream with ``--synthetic``), reports loss/MFU, and runs the correctness gates
(checkpoint alignment, gradient alignment, KL divergence vs upstream). This is the
model-agnostic replacement for the SDM-tied ``train/loop.py`` runner.

Usage:
    PYTHONPATH=src:. python -m train.run --mixer dense_attention --steps 10
    PYTHONPATH=src:. python -m train.run --mixer gla --steps 10 --synthetic
"""

from __future__ import annotations

import argparse
import json

import torch

from train.data import data_generator, get_data, synthetic_generator
from train.harness import TrainConfig, train
from train.registry import MIXER_REGISTRY, get_mixer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URM model-agnostic training harness")
    p.add_argument("--mixer", required=True, choices=sorted(MIXER_REGISTRY))
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--synthetic", action="store_true",
                   help="use a fixed-seed random-token stream instead of finewebedu")
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--batch-tokens", type=int, default=512 * 8)
    p.add_argument("--microbatch-tokens", type=int, default=512 * 8)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--capture-gradients", action="store_true")
    p.add_argument("--out", default=None, help="write the TrainResult JSON here")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = TrainConfig(
        mixer=args.mixer, vocab_size=args.vocab_size,
        sequence_length=args.sequence_length, layers=args.layers, width=args.width,
        num_heads=args.num_heads, head_dim=args.head_dim,
        batch_tokens=args.batch_tokens, microbatch_tokens=args.microbatch_tokens,
        steps=args.steps, seed=args.seed, capture_gradients=args.capture_gradients,
    )
    mixer = get_mixer(args.mixer)
    if args.synthetic:
        data = synthetic_generator(cfg.microbatch_tokens, cfg.sequence_length,
                                   cfg.vocab_size, device="cuda", seed=cfg.seed)
    else:
        get_data("finewebedu_train_000001.bin")
        data = data_generator("finewebedu10B/finewebedu_train_*.bin",
                              cfg.microbatch_tokens, cfg.sequence_length)
    result = train(cfg, mixer, data, device="cuda" if torch.cuda.is_available() else "cpu")
    report = result.to_dict()
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
