"""Capture predeclared operands from a separate unchanged frozen-model replay.

This diagnostic is never timed and never replaces the model's state operator.
Selection is fixed before measurement: step 0, microbatch 3, layer 5 (zero based).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pretraining_step as baseline
import torch
from provenance import write_artifact
from route_parallel_experiment import base_artifact, identity

from urm.pretraining import FP32AdamW, URMDecoderLM
from urm.triton_kernels import sparse_state_mixer as native

SELECTION = {"optimizer_step": 0, "microbatch": 3, "layer": 5}


def capture(seed):
    frozen, config = baseline.load_frozen_config()
    spec, measurement = frozen["optimizer"], frozen["measurement"]
    total = 5 + int(measurement["warmup_steps"]) + int(measurement["measured_steps"])
    batches = baseline.prefetched_batches(config, seed, total, torch)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = URMDecoderLM(config, "urm_native").cuda().to(torch.bfloat16)
    optimizer = FP32AdamW(
        model.named_parameters(),
        lr=float(spec["learning_rate"]),
        betas=(float(spec["beta1"]), float(spec["beta2"])),
        eps=float(spec["epsilon"]),
        weight_decay=float(spec["weight_decay"]),
    )
    model.reset_state()
    matched = {
        "parameters_sha256": baseline._tensor_digest(model.named_parameters(), torch),
        "optimizer_sha256": baseline._tensor_digest(
            enumerate(optimizer.state_tensors()), torch
        ),
        "optimizer_step": optimizer.step_count,
        "batches_sha256": baseline._tensor_digest(
            (
                (f"{i}/{j}", value)
                for i, batch in enumerate(batches)
                for j, value in enumerate(batch)
            ),
            torch,
        ),
        "persistent_state_sha256": baseline._tensor_digest(
            ((i, x.persistent_memory) for i, x in enumerate(model.sparse_mixers())),
            torch,
        ),
    }
    original_fwd = native._sparse_state_update_forward
    original_bwd = native._sparse_state_update_backward
    counts, selected = {"forward": 0, "backward": 0}, {}
    # Every microbatch forwards layers 0..11 and backwards layers 11..0.
    target_fwd, target_bwd = 3 * 12 + 5, 3 * 12 + (11 - 5)

    def forward(*args, **kwargs):
        ordinal = counts["forward"]
        counts["forward"] += 1
        if ordinal == target_fwd:
            assert kwargs["read_before_update"] is False and kwargs["save_selected"]
            # Original mutates its working state, so snapshot BEFORE the call.
            selected["operands"] = [x.detach().cpu().clone() for x in args[:8]]
        result = original_fwd(*args, **kwargs)
        if ordinal == target_fwd:
            selected["history_pointer"] = result[2].data_ptr()
        return result

    def backward(*args, **kwargs):
        ordinal = counts["backward"]
        counts["backward"] += 1
        if ordinal == target_bwd:
            assert args[2].data_ptr() == selected["history_pointer"]
            selected["cotangents"] = [x.detach().cpu().clone() for x in args[:2]]
        return original_bwd(*args, **kwargs)

    native._sparse_state_update_forward, native._sparse_state_update_backward = (
        forward,
        backward,
    )
    try:
        baseline._one_step(
            model,
            optimizer,
            batches[: config.gradient_accumulation],
            config,
            gradient_clip=float(spec["gradient_clip_norm"]),
            record_correctness=False,
            torch=torch,
        )
        torch.cuda.synchronize()
    finally:
        native._sparse_state_update_forward, native._sparse_state_update_backward = (
            original_fwd,
            original_bwd,
        )
    assert counts == {"forward": 48, "backward": 48}
    m, wi, w, v, b, g, ri, q = selected["operands"]
    dy, df = selected["cotangents"]
    sample = (wi, ri, [m, w, v, b, g, q], dy, df)
    # Diagnostics outside all timing. Membership matrix is only 64x64 per token.
    overlap = (wi[..., :, None] == ri[..., None, :]).any(-1).float().mean().item()
    collision = (
        (wi[:, 1:, :, None] == wi[:, :-1, None, :]).any(-1).float().mean().item()
    )
    metadata = {
        "seed": seed,
        "selection": SELECTION,
        "matched_initial": matched,
        "observed_calls": counts,
        "forward_ordinal": target_fwd,
        "backward_ordinal": target_bwd,
        "matched_saved_history_pointer": True,
        "sample_sha256": identity(sample),
        "read_write_route_overlap_fraction": overlap,
        "adjacent_token_write_collision_fraction": collision,
        "initial_state_nonzero_elements": int(m.count_nonzero()),
        "final_cotangent_nonzero_elements": int(df.count_nonzero()),
        "tensor_shapes_and_dtypes": [
            {"shape": list(x.shape), "dtype": str(x.dtype)}
            for x in (wi, ri, *sample[2], dy, df)
        ],
    }
    assert metadata["initial_state_nonzero_elements"] > 0
    return sample, metadata


def load_sample(manifest_path, seed, *, joint_loss=False):
    manifest = json.loads(Path(manifest_path).read_text())
    row = next(x for x in manifest["captures"] if x["seed"] == seed)
    path = Path(manifest_path).parent / row["fixture"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["fixture_sha256"]:
        raise ValueError("captured fixture checksum mismatch")
    with gzip.open(path, "rb") as handle:
        wi, ri, data, cy, cf = torch.load(handle, weights_only=True)
    sample = (wi, ri, data, cy, cf)
    if identity(sample) != row["sample_sha256"]:
        raise ValueError("captured operand identity mismatch")
    if joint_loss:
        # Actual training normally has zero final cotangent after detach. Add an
        # explicitly separate joint-loss check, without changing timing operands.
        generator = torch.Generator().manual_seed(seed + 100000)
        cf = torch.randn(cf.shape, generator=generator) / cf.numel()
    return (wi.cuda(), ri.cuda(), [x.cuda() for x in data], cy.cuda(), cf.cuda())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = base_artifact(
        "ordered_route_parallel_frozen_operand_capture",
        {"selection": SELECTION, "seeds": [1701, 2903, 4409]},
        False,
    )
    artifact["captures"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in artifact["configuration"]["seeds"]:
        sample, metadata = capture(seed)
        fixture = args.output.with_name(f"{args.output.stem}-{seed}.pt.gz")
        with gzip.open(fixture, "wb", compresslevel=6) as handle:
            torch.save(sample, handle)
        metadata["fixture"] = fixture.name
        metadata["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        metadata["fixture_bytes"] = fixture.stat().st_size
        artifact["captures"].append(metadata)
        artifact["complete"] = len(artifact["captures"]) == 3
        write_artifact(args.output, artifact)
        print(f"captured frozen operands for seed {seed}", flush=True)
        del sample
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
