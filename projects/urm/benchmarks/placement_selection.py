"""Solver-guided simulated placement: Z3 versus baselines versus exhaustive.

Runs the committed 2x2 and 2x4 mesh instances, verifies every returned plan
independently (no solver in the verifier), and compares against round-robin,
greedy load balancing, and a brute-force optimum on tiny instances.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_OUTPUT = Path("results/compiler/solver/placement-selection.json")

INSTANCES = [
    {
        "name": "moe_experts_2x2",
        "mesh": "2x2",
        "rows": 2,
        "cols": 2,
        "items": [120, 140, 160, 180],
        "capacity": 340,
        "replication": 1,
        "edges": [(0, 1), (1, 2), (2, 3), (0, 2)],
        "payload_bytes": 4,
    },
    {
        "name": "hot_pair_2x2",
        "mesh": "2x2",
        "rows": 2,
        "cols": 2,
        # Two large experts dominate load; a small hot pair can colocate
        # without changing the optimal maximum load.
        "items": [150, 150, 20, 20],
        "capacity": 400,
        "replication": 1,
        "edges": [(2, 3), (2, 3)],
        "payload_bytes": 32,
    },
    {
        "name": "moe_experts_2x4_skewed",
        "mesh": "2x4",
        "rows": 2,
        "cols": 4,
        "items": [100, 110, 120, 130, 140, 150],
        "capacity": 260,
        "replication": 1,
        "edges": [(i, (i + 3) % 6) for i in range(6)],
        "payload_bytes": 8,
    },
    {
        "name": "pages_replicated_2x2_tiny",
        "mesh": "2x2",
        "rows": 2,
        "cols": 2,
        "items": [60, 70, 80],
        "capacity": 400,
        "replication": 1,
        "edges": [(0, 1), (1, 2)],
        "payload_bytes": 16,
        "exhaustive": True,
    },
]


def build_instance(spec: dict):
    from urm.compiler.placement_solver import (
        PlacementEdge,
        PlacementItem,
        PlacementProblem,
    )

    names = [f"e{i}" for i in range(len(spec["items"]))]
    return PlacementProblem(
        items=tuple(
            PlacementItem(name=name, size_bytes=size)
            for name, size in zip(names, spec["items"], strict=True)
        ),
        device_count=spec["rows"] * spec["cols"],
        device_capacity_bytes=spec["capacity"],
        mesh_rows=spec["rows"],
        mesh_cols=spec["cols"],
        edges=tuple(
            PlacementEdge(
                source_item=names[src],
                query_item=names[dst],
                payload_bytes=spec["payload_bytes"],
            )
            for src, dst in spec["edges"]
        ),
        replication_factor=spec.get("replication", 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from provenance import provenance, utc_now, write_artifact

    from urm.compiler.placement_solver import (
        build_placement_model,
        decode_placement,
        exhaustive_placement_optimum,
        feasible_owners,
        greedy_placement,
        placement_metrics,
        round_robin_placement,
    )
    from urm.compiler.solver import OptimizationPass
    from urm.compiler.verification import (
        AssignmentFacts,
        ModelVerifier,
        PlacementItemFacts,
    )

    rows: list[dict[str, object]] = []
    agreements = 0
    exhaustive_runs = 0
    all_verified = True
    savings: list[float] = []

    for spec in INSTANCES:
        problem = build_instance(spec)
        model = build_placement_model(problem)

        started = time.perf_counter()
        z3_result = OptimizationPass().run(model)
        solve_ms = (time.perf_counter() - started) * 1000.0

        record: dict[str, object] = {
            "name": spec["name"],
            "mesh": spec["mesh"],
            "item_count": len(problem.items),
            "edge_count": len(problem.edges),
            "replication_factor": problem.replication_factor,
            "device_capacity_bytes": problem.device_capacity_bytes,
        }

        if z3_result.status.value == "sat":
            owners = decode_placement(z3_result.assignment, problem)
            metrics = placement_metrics(owners, problem)
            facts = AssignmentFacts(
                devices=tuple(range(problem.device_count)),
                device_capacity_bytes={
                    d: problem.device_capacity_bytes for d in problem.devices()
                },
                items=tuple(
                    PlacementItemFacts(
                        name=item.name,
                        size_bytes=item.size_bytes,
                        owner_variable="unused",
                        one_hot_devices=tuple(problem.devices()),
                        replication_factor=problem.replication_factor,
                    )
                    for item in problem.items
                ),
            )
            report = ModelVerifier().verify(model, z3_result.assignment, facts)
            verified = report.ok
            all_verified = all_verified and verified
            rr = round_robin_placement(problem)
            rr_metrics = placement_metrics(rr, problem)
            savings_pct = (
                100.0
                * (rr_metrics["total_bytes"] - metrics["total_bytes"])
                / rr_metrics["total_bytes"]
                if rr_metrics["total_bytes"]
                else 0.0
            )
            savings.append(savings_pct)
            record["z3"] = {
                "status": "sat",
                "owners": {k: int(v) for k, v in owners.items()},
                "metrics": metrics,
                "solve_ms": round(solve_ms, 3),
                "verified": verified,
                "verification_failures": [f.to_dict() for f in report.failures],
            }
            record["baselines"] = {
                "round_robin": {
                    "owners": {k: int(v) for k, v in rr.items()},
                    "metrics": rr_metrics,
                    "feasible": feasible_owners(rr, problem),
                },
                "greedy": _baseline_record(greedy_placement(problem), problem),
            }
            record["communication_savings_vs_round_robin_pct"] = round(savings_pct, 2)

            wants_exhaustive = (
                bool(spec.get("exhaustive"))
                and len(problem.items) ** problem.device_count <= 200_000
            )
            if wants_exhaustive:
                exhaustive_runs += 1
                optimum = exhaustive_placement_optimum(problem)
                agree = optimum is not None and (
                    optimum[1] == metrics or optimum[0] == owners
                )
                agreements += int(agree)
                record["exhaustive_agreement"] = {
                    "run": True,
                    "agree": bool(agree),
                    "owners": (
                        {k: int(v) for k, v in optimum[0].items()} if optimum else None
                    ),
                    "metrics": optimum[1] if optimum else None,
                }
            else:
                record["exhaustive_agreement"] = {"run": False, "agree": False}
        else:
            record["z3"] = {
                "status": z3_result.status.value,
                "unsat_core_names": list(z3_result.unsat_core_names)[:12],
                "solve_ms": round(solve_ms, 3),
                "verified": False,
            }
            record["exhaustive_agreement"] = {"run": False, "agree": False}
        rows.append(record)

    configuration = {"instances": [s["name"] for s in INSTANCES]}
    artifact = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "provenance": {
            **provenance("python benchmarks/placement_selection.py", configuration),
            # Hash of every instance model's constraint summary.
            "constraint_model_hash": _model_hash(),
        },
        "instances": rows,
        "summary": {
            "instances_run": len(rows),
            "all_verified": all_verified,
            "exhaustive_agreements": agreements,
            "mean_comm_savings_vs_round_robin_pct": (
                round(sum(savings) / len(savings), 2) if savings else 0.0
            ),
        },
    }
    write_artifact(args.output, artifact)
    print(json.dumps(artifact["summary"], indent=2))


def _baseline_record(owners, problem):
    from urm.compiler.placement_solver import feasible_owners, placement_metrics

    return {
        "owners": {k: int(v) for k, v in owners.items()},
        "metrics": placement_metrics(owners, problem),
        "feasible": feasible_owners(owners, problem),
    }


def _model_hash() -> str:
    import hashlib

    from urm.compiler.placement_solver import build_placement_model

    summaries = [
        build_placement_model(build_instance(spec)).to_summary() for spec in INSTANCES
    ]
    canonical = json.dumps(summaries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
