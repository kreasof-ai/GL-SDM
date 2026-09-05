"""Observe real eager/Inductor Triton launches outside authoritative timing."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def observe_launches():
    from triton import knobs
    from triton.compiler.compiler import CompiledKernel

    original = CompiledKernel.launch_metadata
    previous_hook = knobs.runtime.launch_enter_hook
    records = []

    def observed(kernel, grid, stream, *args):
        name = kernel.name
        if "sparse_state_update" in name or "sparse_route_" in name:
            names = kernel.src.fn.arg_names
            constants = {
                names[key[0]] if isinstance(key, tuple) else str(key): value
                for key, value in kernel.src.constants.items()
            }
            records.append(
                {
                    "kernel": name,
                    "grid": list(grid),
                    "constants": constants,
                    "num_warps": kernel.metadata.num_warps,
                    "num_stages": kernel.metadata.num_stages,
                }
            )
        return original(kernel, grid, stream, *args)

    # Inductor specializes its launcher with the hook present. This audit must
    # run in its own process/cache, never in a timing child.
    knobs.runtime.launch_enter_hook = lambda metadata: None
    CompiledKernel.launch_metadata = observed
    try:
        yield records
    finally:
        CompiledKernel.launch_metadata = original
        knobs.runtime.launch_enter_hook = previous_hook


def verify_launches(records, plan, config):
    launch = plan["steps"][0]["launch_config"]
    categories = set()
    for record in records:
        constants = record["constants"]
        backward = "backward" in record["kernel"]
        if "sparse_state_update" in record["kernel"]:
            categories.add("state_backward" if backward else "state_forward")
            expected = {
                "SEQUENCE": config.sequence_length,
                "SLOTS": config.slots_per_partition,
                "WRITE_WIDTH": config.writes,
                "READ_WIDTH": config.reads,
                "VALUE_DIM": config.value_dim,
                "BLOCK_D": launch["state_block_d"],
                "READ_BEFORE_UPDATE": False,
            }
            if not backward:
                expected["SAVE_SELECTED"] = True
            expected_grid = [
                config.parallel,
                (config.value_dim + launch["state_block_d"] - 1)
                // launch["state_block_d"],
            ]
            warps, stages = launch["state_num_warps"], launch["state_num_stages"]
        else:
            categories.add("route_backward" if backward else "route_forward")
            width = constants["WIDTH"]
            role = "read" if width == config.reads else "write"
            block = launch[f"{role}_route_block"]
            expected = {
                "HALF": config.factor_extent,
                "WIDTH": width,
                "BLOCK_HALF": launch["route_block_half"],
                "BLOCK_ROUTE": block,
            }
            if not backward:
                expected["BLOCK_PAIR"] = block * block
            assert width in (config.reads, config.writes)
            expected_grid = [config.parallel * config.sequence_length]
            warps = (
                launch["route_backward_num_warps"]
                if backward
                else launch[f"{role}_route_num_warps"]
            )
            stages = launch["route_num_stages"]
        assert all(constants.get(key) == value for key, value in expected.items()), (
            record,
            expected,
        )
        assert record["grid"][: len(expected_grid)] == expected_grid, (
            record,
            expected_grid,
        )
        assert all(value == 1 for value in record["grid"][len(expected_grid) :])
        assert (record["num_warps"], record["num_stages"]) == (warps, stages), record
    assert categories == {
        "state_forward",
        "state_backward",
        "route_forward",
        "route_backward",
    }, categories


def main():
    import torch
    from pretraining_step import load_frozen_config
    from provenance import provenance, utc_now, write_artifact

    from urm.pretraining import SparseMemoryMixer

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "compile_fullgraph"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, config = load_frozen_config()
    torch.manual_seed(1701)
    mixer = SparseMemoryMixer(config, "urm_native").cuda().to(torch.bfloat16)
    x = torch.randn(
        config.microbatch,
        config.sequence_length,
        config.width,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with observe_launches() as records:
        graph = (
            torch.compile(mixer, fullgraph=True, dynamic=False)
            if args.mode == "compile_fullgraph"
            else mixer
        )
        graph(x).float().square().mean().backward()
        torch.cuda.synchronize()
    plan = mixer._executor.serialized_plan()
    verify_launches(records, plan, config)
    write_artifact(
        args.output,
        {
            "schema_version": 1,
            "artifact_kind": "sparse_memory_production_launch_audit",
            "generated_utc": utc_now(),
            "provenance": provenance(
                "audit_sparse_memory_launches", payload, include_gpu=True
            ),
            "mode": args.mode,
            "scope": "one_production_shape_layer_forward_and_backward_separate_process_no_timing",
            "plan": plan,
            "observed_launches": records,
            "passed": True,
        },
    )


if __name__ == "__main__":
    main()
