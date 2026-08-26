"""Tiny dependency-light timing smoke test, not a publishable benchmark."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from urm.presets import DENSE_ATTENTION, GL_SDM_TRANSACTION, TOP2_MOE
from urm.reference import execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("attention", "moe", "memory"), default="attention"
    )
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--sources", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = {
        "attention": DENSE_ATTENTION,
        "moe": TOP2_MOE,
        "memory": GL_SDM_TRANSACTION,
    }[args.family]
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((args.queries, args.sources))
    values = rng.standard_normal((args.sources, args.value_dim))

    execute(scores, values, spec)
    started = perf_counter()
    for _ in range(args.iterations):
        execute(scores, values, spec)
    elapsed = perf_counter() - started

    print(
        json.dumps(
            {
                "backend": "numpy_reference",
                "family": args.family,
                "queries": args.queries,
                "sources": args.sources,
                "value_dim": args.value_dim,
                "iterations": args.iterations,
                "mean_ms": 1000.0 * elapsed / args.iterations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
