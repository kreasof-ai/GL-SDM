"""Representative impossible problems: run, diagnose, and commit compactly.

Every case must come back UNSAT with an unsat core that maps onto the named
constraints the catalog expects; raw solver formulas never appear in output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_OUTPUT = Path("results/compiler/solver/unsat-diagnostics.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from provenance import provenance, utc_now, write_artifact

    from urm.compiler.solver import FeasibilityPass
    from urm.compiler.unsat_catalog import REPRESENTATIVE_UNSAT_CASES, describe_unsat

    records: list[dict[str, object]] = []
    for case in REPRESENTATIVE_UNSAT_CASES:
        started = time.perf_counter()
        result = FeasibilityPass().run(case.build())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert result.status.value == "unsat", (
            f"{case.name} unexpectedly {result.status.value}"
        )
        record = describe_unsat(case, result.unsat_core_names)
        record["solve_ms"] = round(elapsed_ms, 3)
        assert record["core_mapped"], (case.name, result.unsat_core_names)
        records.append(record)

    configuration = {"cases": [c.name for c in REPRESENTATIVE_UNSAT_CASES]}
    artifact = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "provenance": {
            **provenance("python benchmarks/unsat_diagnostics.py", configuration),
            # The model hash covers every case model's summary.
            "constraint_model_hash": _catalog_model_hash(),
        },
        "cases": records,
        "summary": {
            "cases_run": len(records),
            "all_unsat": all(r["status"] == "unsat" for r in records),
            "all_cores_mapped": all(r["core_mapped"] for r in records),
        },
    }
    write_artifact(args.output, artifact)
    print(json.dumps(artifact["summary"], indent=2))


def _catalog_model_hash() -> str:
    import hashlib
    import json as _json

    from urm.compiler.unsat_catalog import REPRESENTATIVE_UNSAT_CASES

    summaries = [case.build().to_summary() for case in REPRESENTATIVE_UNSAT_CASES]
    canonical = _json.dumps(summaries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
