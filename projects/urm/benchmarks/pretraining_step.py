"""Frozen model-level optimizer-step benchmark for Sparse Memory mixers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).with_name("pretraining_step.toml")
SCHEMA_PATH = Path(__file__).with_name("pretraining-step-authority-schema.json")
DEFAULT_OUTPUT = Path("results/pretraining-step/confirmation-authority-v2.json")
UPSTREAM_COMMIT = "183e7df809131b80ad4393741029d0f20fc3640b"
FINEWEB_SHA256 = "6bb7ce7bcac8e11463433767ec3402311c7527c3d8d766e7d65ef86dc4546bb2"


def load_frozen_config(*, diagnostic: bool = False):
    from urm.pretraining import PretrainingConfig

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1 or payload["freeze_status"] != "pre_measurement":
        raise RuntimeError("pretraining-step configuration is not frozen")
    values = dict(payload["model"])
    for key in (
        "architecture",
        "activation_checkpointing",
        "parameter_dtype",
        "optimizer_state_dtype",
    ):
        values.pop(key)
    if diagnostic:
        values.update(payload["diagnostic"])
        values.pop("enabled_only_after_primary_failure")
    config = PretrainingConfig(**values)
    return payload, config


def _fineweb_tokens(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(
            f"cached FineWeb shard missing: {path}; see docs/pretraining-step.md"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FINEWEB_SHA256:
        raise RuntimeError(f"FineWeb cache hash mismatch: {digest}")
    header = np.memmap(path, dtype=np.int32, mode="r", shape=(256,))
    if tuple(int(x) for x in header[:3]) != (20240520, 1, 100000000):
        raise RuntimeError("unexpected FineWeb token-shard header")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024)


def prefetched_batches(config, seed: int, count: int, torch):
    tokens = _fineweb_tokens(Path("data/fineweb_edu_gpt2_uint16.bin"))
    needed = config.microbatch * (config.sequence_length + 1)
    rng = random.Random(seed)
    batches = []
    for _ in range(count * config.gradient_accumulation):
        start = rng.randrange(0, len(tokens) - needed)
        host = np.array(tokens[start : start + needed], dtype=np.int64).reshape(
            config.microbatch, config.sequence_length + 1
        )
        device = torch.from_numpy(host).to("cuda", non_blocking=False)
        batches.append((device[:, :-1], device[:, 1:]))
    return batches


def _data_loading_lane(config, seed: int, torch) -> dict[str, object]:
    """Separate pageable-host FineWeb-to-device throughput diagnostic."""
    tokens = _fineweb_tokens(Path("data/fineweb_edu_gpt2_uint16.bin"))
    needed = config.microbatch * (config.sequence_length + 1)
    rng = random.Random(seed + 91_337)
    raw = []
    for _ in range(10):
        start = rng.randrange(0, len(tokens) - needed)
        torch.cuda.synchronize()
        began = time.perf_counter_ns()
        host = np.array(tokens[start : start + needed], dtype=np.int64).reshape(
            config.microbatch, config.sequence_length + 1
        )
        _device = torch.from_numpy(host).to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        raw.append((time.perf_counter_ns() - began) / 1e6)
    stats = _stats(raw)
    stats["tokens_per_second"] = needed / (stats["median_ms"] / 1000.0)
    stats["scope"] = "memmap_slice_plus_pageable_host_materialization_plus_cuda_copy"
    return stats


def _stats(values):
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": p95,
        "minimum_ms": ordered[0],
        "raw_ms": values,
    }


def _validate_confirmation(artifact) -> None:
    from jsonschema import validate

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate(artifact, schema)


def _finite_model(model, optimizer, torch):
    counters = {
        "parameters": 0,
        "gradients": 0,
        "optimizer_state": 0,
        "persistent_state": 0,
    }
    for parameter in model.parameters():
        counters["parameters"] += int((~torch.isfinite(parameter)).sum().item())
        if parameter.grad is not None:
            counters["gradients"] += int((~torch.isfinite(parameter.grad)).sum().item())
    for tensor in optimizer.state_tensors():
        counters["optimizer_state"] += int((~torch.isfinite(tensor)).sum().item())
    for mixer in model.sparse_mixers():
        counters["persistent_state"] += int(
            (~torch.isfinite(mixer.persistent_memory)).sum().item()
        )
    return counters


def _one_step(
    model,
    optimizer,
    batches,
    config,
    *,
    gradient_clip: float,
    record_correctness: bool,
    torch,
    capture_internal_gradients: bool = False,
    state_snapshots=None,
):
    from urm.pretraining import gradient_norms

    execution = model
    model = getattr(model, "_orig_mod", model)
    model.reset_state()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    sampled_logits = []
    mixer_gradients = []
    microbatch_states = []
    model.capture_mixer_input_gradients(capture_internal_gradients)
    profile_ranges = getattr(model, "profile_ranges", False)
    for tokens, targets in batches:
        logits, loss = execution(tokens, targets)
        if loss is None:
            raise RuntimeError("language-model loss was not produced")
        if profile_ranges:
            with torch.autograd.profiler.record_function("pretraining::backward"):
                (loss / config.gradient_accumulation).backward()
        else:
            (loss / config.gradient_accumulation).backward()
        if record_correctness:
            losses.append(float(loss.detach().item()))
            positions = (0, config.sequence_length // 2, config.sequence_length - 1)
            sampled_logits.append(
                logits[0, positions, :8].detach().float().cpu().tolist()
            )
        if capture_internal_gradients:
            mixer_gradients.append(
                [gradient.float().cpu() for gradient in model.mixer_input_gradients()]
            )
        model.detach_state()
        if record_correctness:
            microbatch_states.append(model.state_checksums())
            if state_snapshots is not None:
                state_snapshots.append(
                    [
                        mixer.persistent_memory.detach().cpu().clone()
                        for mixer in model.sparse_mixers()
                    ]
                )
    if profile_ranges:
        with torch.autograd.profiler.record_function("pretraining::gradient_clip"):
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    norms = gradient_norms(model) if record_correctness else {}
    if profile_ranges:
        with torch.autograd.profiler.record_function("pretraining::optimizer"):
            updates = optimizer.step(record_updates=record_correctness)
    else:
        updates = optimizer.step(record_updates=record_correctness)
    model.capture_mixer_input_gradients(False)
    if not record_correctness:
        # No host scalar reads, checksums, or finite scans in the timed boundary.
        return {}, []
    return {
        "loss_before": statistics.mean(losses),
        "sampled_logits": sampled_logits,
        "gradient_norms": norms,
        "gradient_clip_input_norm": float(total_norm.item()),
        "parameter_updates": updates,
        "persistent_state_checksums": microbatch_states[-1],
        "microbatch_persistent_state_checksums": microbatch_states,
        "nonfinite": _finite_model(model, optimizer, torch),
    }, mixer_gradients


def _tensor_digest(tensors, torch):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _evaluation_loss(model, batch, torch):
    model.reset_state()
    with torch.no_grad():
        _logits, loss = model(*batch)
    model.discard_pending_state()
    return float(loss.item())


def _saved_activation_bytes(model, batch, torch):
    seen = set()
    total = 0

    def pack(tensor):
        nonlocal total
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        if key not in seen:
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
        return tensor

    model.reset_state()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        _logits, loss = model(*batch)
    model.discard_pending_state()
    del loss
    return total


def _child(args) -> None:
    process_started = time.perf_counter_ns()
    import torch

    torch_import_ms = (time.perf_counter_ns() - process_started) / 1e6
    module_import_started = time.perf_counter_ns()
    from provenance import provenance, utc_now, write_artifact

    from urm.pretraining import (
        FP32AdamW,
        URMDecoderLM,
        model_memory_ledger,
        model_parameter_count,
        semantic_training_flops,
    )

    urm_module_import_ms = (time.perf_counter_ns() - module_import_started) / 1e6

    payload, config = load_frozen_config(diagnostic=args.diagnostic)
    measurement = payload["measurement"]
    optimizer_spec = payload["optimizer"]
    total_steps = (
        5 + int(measurement["warmup_steps"]) + int(measurement["measured_steps"])
    )
    batches = prefetched_batches(config, args.seed, total_steps, torch)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    backend_dependency_import_build_ms = None
    backend_dependency_import_build_scope = "not_applicable"
    upstream_support = None
    if args.backend == "upstream_sdm":
        dependency_started = time.perf_counter_ns()
        from urm.adapters.sparse_delta_memory import probe_sdm_support

        upstream_support = probe_sdm_support()
        backend_dependency_import_build_ms = (
            time.perf_counter_ns() - dependency_started
        ) / 1e6
        backend_dependency_import_build_scope = (
            "pinned_sdm_import_revision_probe_and_extension_build_if_cold"
        )
        if not upstream_support.supported:
            raise RuntimeError(
                "pinned upstream comparator unavailable "
                f"[{upstream_support.code}]: {upstream_support.reason}"
            )
    construct_started = time.perf_counter_ns()
    model = URMDecoderLM(config, args.backend).cuda().to(torch.bfloat16)
    optimizer = FP32AdamW(
        model.named_parameters(),
        lr=float(optimizer_spec["learning_rate"]),
        betas=(float(optimizer_spec["beta1"]), float(optimizer_spec["beta2"])),
        eps=float(optimizer_spec["epsilon"]),
        weight_decay=float(optimizer_spec["weight_decay"]),
    )
    construction_ms = (time.perf_counter_ns() - construct_started) / 1e6
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    matched_initial = {
        "parameters_sha256": _tensor_digest(model.named_parameters(), torch),
        "optimizer_sha256": _tensor_digest(enumerate(optimizer.state_tensors()), torch),
        "optimizer_step": optimizer.step_count,
        "batches_sha256": _tensor_digest(
            (
                (f"{i}/{j}", value)
                for i, batch in enumerate(batches)
                for j, value in enumerate(batch)
            ),
            torch,
        ),
    }

    def restore_initial():
        model.load_state_dict(initial_state)
        model.reset_state()
        optimizer.zero_grad(set_to_none=True)
        optimizer.step_count = 0
        with torch.no_grad():
            for parameter, master, mean, variance in zip(
                optimizer.parameters,
                optimizer.master,
                optimizer.exp_avg,
                optimizer.exp_avg_sq,
                strict=True,
            ):
                master.copy_(parameter.float())
                mean.zero_()
                variance.zero_()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = model_parameter_count(config, args.backend)
    if parameter_count != expected_parameters:
        raise RuntimeError(
            f"parameter ledger mismatch: actual={parameter_count}, ledger={expected_parameters}"
        )
    if args.backend == "urm_native":
        plans = [mixer._executor.serialized_plan() for mixer in model.sparse_mixers()]
        if not plans or any(plan != plans[0] for plan in plans[1:]):
            raise RuntimeError("native model layers did not receive one frozen plan")
        execution = {
            "kind": "compiler_produced_native_plan",
            "layer_count": len(plans),
            "compiler_plan": plans[0],
            "compiled_boundary": [],
            "benchmark_direct_backend_instantiation": False,
        }
    elif args.backend == "upstream_sdm":
        execution = {
            "kind": "pinned_external_comparator",
            "layer_count": config.layers,
            "compiler_plan": None,
            "compiled_boundary": [
                "urm::compiled_upstream_sdm_update",
                "urm::compiled_upstream_sdm_backward",
            ],
            "benchmark_direct_backend_instantiation": False,
        }
    else:
        execution = {
            "kind": "sdpa_contextual_control",
            "layer_count": config.layers,
            "compiler_plan": None,
            "compiled_boundary": [],
            "benchmark_direct_backend_instantiation": False,
        }
    graph = model
    compile_ms = 0.0
    if args.mode == "compile_fullgraph":
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch._dynamo.config.error_on_recompile = True
        started = time.perf_counter_ns()
        graph = torch.compile(model, fullgraph=True, dynamic=False)
        compile_ms = (time.perf_counter_ns() - started) / 1e6

    cursor = 0
    correctness = []
    gradient_files = []
    state_files = []
    first_step_ms = None
    first_step_peak = None
    # Cold timing uses the same diagnostic-free boundary as settled timing.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    first_call_started = time.perf_counter_ns()
    _one_step(
        graph,
        optimizer,
        batches[: config.gradient_accumulation],
        config,
        gradient_clip=float(optimizer_spec["gradient_clip_norm"]),
        record_correctness=False,
        torch=torch,
    )
    torch.cuda.synchronize()
    first_step_ms = (time.perf_counter_ns() - first_call_started) / 1e6
    first_step_peak = torch.cuda.max_memory_allocated()
    restore_initial()
    if (
        _tensor_digest(model.named_parameters(), torch)
        != matched_initial["parameters_sha256"]
        or _tensor_digest(enumerate(optimizer.state_tensors()), torch)
        != matched_initial["optimizer_sha256"]
    ):
        raise RuntimeError("cold-step reset did not restore matched initial state")
    for index in range(5):
        step_batches = batches[cursor : cursor + config.gradient_accumulation]
        cursor += config.gradient_accumulation
        state_snapshots = []
        report, gradients = _one_step(
            graph,
            optimizer,
            step_batches,
            config,
            gradient_clip=float(optimizer_spec["gradient_clip_norm"]),
            record_correctness=True,
            state_snapshots=state_snapshots,
            torch=torch,
        )
        # Auxiliary post-update evaluation is explicitly eager in both modes.
        report["loss_after"] = _evaluation_loss(model, step_batches[0], torch)
        state_path = args.output.with_suffix(f".states-{index}.pt")
        torch.save(state_snapshots, state_path)
        state_files.append(str(state_path))
        del state_snapshots
        correctness.append(report)

    first_report = correctness[0]
    for warmup_index in range(int(measurement["warmup_steps"])):
        step_batches = batches[cursor : cursor + config.gradient_accumulation]
        cursor += config.gradient_accumulation
        _one_step(
            graph,
            optimizer,
            step_batches,
            config,
            gradient_clip=float(optimizer_spec["gradient_clip_norm"]),
            record_correctness=False,
            torch=torch,
        )
    torch.cuda.synchronize()
    raw = []
    peaks_allocated = []
    peaks_reserved = []
    temporary_allocated = []
    for _ in range(int(measurement["measured_steps"])):
        step_batches = batches[cursor : cursor + config.gradient_accumulation]
        cursor += config.gradient_accumulation
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        started = time.perf_counter_ns()
        _one_step(
            graph,
            optimizer,
            step_batches,
            config,
            gradient_clip=float(optimizer_spec["gradient_clip_norm"]),
            record_correctness=False,
            torch=torch,
        )
        torch.cuda.synchronize()
        raw.append((time.perf_counter_ns() - started) / 1e6)
        peaks_allocated.append(torch.cuda.max_memory_allocated())
        peaks_reserved.append(torch.cuda.max_memory_reserved())
        temporary_allocated.append(torch.cuda.max_memory_allocated() - allocated_before)

    # This accounting probe is eager and outside all authoritative timing.
    saved_activations = _saved_activation_bytes(model, batches[0], torch)
    memory = model_memory_ledger(model, optimizer)
    memory.update(
        {
            "saved_activation_bytes": saved_activations,
            "steady_peak_allocated_bytes": max(peaks_allocated),
            "steady_peak_reserved_bytes": max(peaks_reserved),
            "temporary_peak_allocated_bytes": max(temporary_allocated),
            "first_step_compilation_peak_bytes": first_step_peak,
            "maximum_common_microbatch": config.microbatch,
            "maximum_backend_microbatch": config.microbatch,
            "maximum_backend_microbatch_status": "bounded_by_native_parallel_envelope_not_oom_search",
        }
    )
    dynamo = (
        dict(torch._dynamo.utils.counters) if args.mode == "compile_fullgraph" else {}
    )
    unique_graphs = (
        int(dynamo.get("stats", {}).get("unique_graphs", 0)) if dynamo else 0
    )
    graph_breaks = (
        sum(int(value) for value in dynamo.get("graph_break", {}).values())
        if dynamo
        else 0
    )
    if args.mode == "compile_fullgraph" and (graph_breaks or unique_graphs != 1):
        raise RuntimeError(
            f"compiled lane graph invariant failed: graph_breaks={graph_breaks}, unique_graphs={unique_graphs}"
        )
    # Separate eager-only internal-gradient evidence, replayed from the same
    # initial model/AdamW state and the first five batches, after all timing.
    restore_initial()
    eager_diagnostics = []
    for index in range(5):
        step_batches = batches[
            index * config.gradient_accumulation : (index + 1)
            * config.gradient_accumulation
        ]
        report, gradients = _one_step(
            model,
            optimizer,
            step_batches,
            config,
            gradient_clip=float(optimizer_spec["gradient_clip_norm"]),
            record_correctness=True,
            capture_internal_gradients=True,
            torch=torch,
        )
        report["loss_after"] = _evaluation_loss(model, step_batches[0], torch)
        gradient_path = args.output.with_suffix(f".gradients-{index}.pt")
        torch.save(gradients, gradient_path)
        gradient_files.append(str(gradient_path))
        eager_diagnostics.append(report)
    tokens_per_step = (
        config.microbatch * config.sequence_length * config.gradient_accumulation
    )
    timing = _stats(raw)
    useful_flops = semantic_training_flops(config, args.backend)
    upstream = None
    if upstream_support is not None:
        upstream = upstream_support.details
    result = {
        "schema_version": 1,
        "artifact_kind": "pretraining_backend_process",
        "generated_utc": utc_now(),
        "provenance": provenance(
            " ".join(sys.argv),
            {
                "config": config.to_dict(),
                "backend": args.backend,
                "mode": args.mode,
                "seed": args.seed,
            },
            include_gpu=True,
        ),
        "backend": args.backend,
        "mode": args.mode,
        "seed": args.seed,
        "diagnostic": args.diagnostic,
        "upstream": upstream,
        "configuration": config.to_dict(),
        "execution": execution,
        "fineweb": {
            "repository": "karpathy/fineweb-edu-100B-gpt2-token-shards",
            "revision": "a33f75d78c7f74236fb03754ec1eb3cc77507d64",
            "file": "edu_fineweb_train_000001.bin",
            "sha256": FINEWEB_SHA256,
            "cache_path": "data/fineweb_edu_gpt2_uint16.bin",
        },
        "cold": {
            "process_to_torch_import_ms": torch_import_ms,
            "urm_module_import_ms": urm_module_import_ms,
            "backend_dependency_import_build_ms": backend_dependency_import_build_ms,
            "backend_dependency_import_build_scope": (
                backend_dependency_import_build_scope
            ),
            "model_optimizer_construction_ms": construction_ms,
            "torch_compile_wrapper_ms": compile_ms,
            "first_optimizer_step_ms": first_step_ms,
        },
        "correctness": correctness,
        "correctness_execution": args.mode,
        "matched_initial": matched_initial,
        "eager_internal_gradient_diagnostics": eager_diagnostics,
        "compiled_or_eager_first_step": first_report,
        "gradient_files": gradient_files,
        "state_files": state_files,
        "timing": timing,
        "tokens_per_second": tokens_per_step / (timing["median_ms"] / 1000.0),
        "optimizer_steps_per_second": 1000.0 / timing["median_ms"],
        "data_loading_lane": _data_loading_lane(config, args.seed, torch),
        "memory": memory,
        "flops": {
            **useful_flops.to_dict(),
            "achieved_useful_tflops": useful_flops.useful_total
            / (timing["median_ms"] / 1000.0)
            / 1e12,
            "model_mfu": useful_flops.useful_total
            / (timing["median_ms"] / 1000.0)
            / (float(payload["flops"]["measured_bfloat16_tensor_core_tflops"]) * 1e12),
            "denominator": payload["flops"],
        },
        "compile": {
            "fullgraph": args.mode == "compile_fullgraph",
            "graph_break_count": graph_breaks,
            "unique_graphs": unique_graphs,
            "recompilations": max(0, unique_graphs - 1),
        },
    }
    write_artifact(args.output, result)


def _compare_correctness(upstream, native, torch, tolerances, *, internal=False):
    if upstream["matched_initial"] != native["matched_initial"]:
        raise RuntimeError(
            "correctness comparison requires matched parameters, optimizer state, and batches"
        )
    key = "eager_internal_gradient_diagnostics" if internal else "correctness"
    reports = []
    for index, (left, right) in enumerate(zip(upstream[key], native[key], strict=True)):
        left_gradients = (
            torch.load(upstream["gradient_files"][index], weights_only=True)
            if internal
            else []
        )
        right_gradients = (
            torch.load(native["gradient_files"][index], weights_only=True)
            if internal
            else []
        )
        mixer = []
        state_tensors = []
        if not internal and "state_files" in upstream and "state_files" in native:
            left_states = torch.load(
                upstream["state_files"][index], weights_only=True, mmap=True
            )
            right_states = torch.load(
                native["state_files"][index], weights_only=True, mmap=True
            )
            for microbatch, (micro_left, micro_right) in enumerate(
                zip(left_states, right_states, strict=True)
            ):
                for block, (a, b) in enumerate(
                    zip(micro_left, micro_right, strict=True)
                ):
                    difference = a.float() - b.float()
                    state_tensors.append(
                        {
                            "microbatch": microbatch,
                            "block": block,
                            "max_abs": float(difference.abs().max()),
                            "rms": float(difference.square().mean().sqrt()),
                        }
                    )
        for micro_left, micro_right in zip(
            left_gradients, right_gradients, strict=True
        ):
            for block, (grad_left, grad_right) in enumerate(
                zip(micro_left, micro_right, strict=True)
            ):
                flat_left = grad_left.flatten()
                flat_right = grad_right.flatten()
                mixer.append(
                    {
                        "block": block,
                        "cosine": float(
                            torch.nn.functional.cosine_similarity(
                                flat_left, flat_right, dim=0
                            ).item()
                        ),
                        "max_abs": float((flat_left - flat_right).abs().max().item()),
                    }
                )
        logit_max = 0.0
        for a, b in zip(left["sampled_logits"], right["sampled_logits"], strict=True):
            logit_max = max(
                logit_max, float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            )
        loss_error = max(
            abs(left["loss_before"] - right["loss_before"]),
            abs(left["loss_after"] - right["loss_after"]),
        )
        gradient_norm_error = max(
            abs(left["gradient_norms"][key] - right["gradient_norms"][key])
            for key in left["gradient_norms"]
        )
        update_error = max(
            abs(
                left["parameter_updates"][key][field]
                - right["parameter_updates"][key][field]
            )
            for key in left["parameter_updates"]
            for field in ("l2", "max_abs")
        )
        state_pairs = [
            (a, b)
            for micro_left, micro_right in zip(
                left["microbatch_persistent_state_checksums"],
                right["microbatch_persistent_state_checksums"],
                strict=True,
            )
            for a, b in zip(micro_left, micro_right, strict=True)
        ]
        state_sum_error = max(
            (abs(a["sum"] - b["sum"]) for a, b in state_pairs), default=0.0
        )
        state_normalized_error = max(
            (abs(a["mean"] - b["mean"]) for a, b in state_pairs), default=0.0
        )
        finite = not any(
            value for report in (left, right) for value in report["nonfinite"].values()
        )
        passed = bool(
            finite
            and loss_error <= tolerances["loss_max_abs"]
            and logit_max <= tolerances["logit_max_abs"]
            and gradient_norm_error <= tolerances["gradient_norm_max_abs"]
            and update_error <= tolerances["parameter_update_max_abs"]
            and state_normalized_error
            <= tolerances["persistent_state_checksum_normalized_max_abs"]
            and min((item["cosine"] for item in mixer), default=1.0)
            >= tolerances["mixer_input_gradient_cosine_min"]
            and max((item["max_abs"] for item in mixer), default=0.0)
            <= tolerances["mixer_input_gradient_max_abs"]
        )
        reports.append(
            {
                "step": index,
                "loss_max_abs": loss_error,
                "logit_max_abs": logit_max,
                "gradient_norm_max_abs": gradient_norm_error,
                "parameter_update_max_abs": update_error,
                "persistent_state_checksum_sum_max_abs": state_sum_error,
                "persistent_state_checksum_normalized_max_abs": state_normalized_error,
                "mixer_input_gradients": mixer,
                "persistent_state_tensor_diagnostics": state_tensors,
                "finite": finite,
                "passed": passed,
            }
        )
    return {"steps": reports, "passed": all(report["passed"] for report in reports)}


def _ratio_summary(pairs):
    from measurement import bootstrap_ci, hierarchical_bootstrap_paired_slowdown

    log_by_seed = []
    medians = []
    p95_ratios = []
    for upstream, native in pairs:
        raw = [
            n / u
            for n, u in zip(
                native["timing"]["raw_ms"], upstream["timing"]["raw_ms"], strict=True
            )
        ]
        log_by_seed.append([math.log(value) for value in raw])
        medians.append(native["timing"]["median_ms"] / upstream["timing"]["median_ms"])
        p95_ratios.append(native["timing"]["p95_ms"] / upstream["timing"]["p95_ms"])
    median_pct, lower_pct, upper_pct = hierarchical_bootstrap_paired_slowdown(
        log_by_seed, num_resamples=10_000, seed=9413
    )
    all_ratios = [math.exp(value) for run in log_by_seed for value in run]
    flat_lower, flat_upper = bootstrap_ci(all_ratios, num_resamples=10_000, seed=9949)
    geometric = math.exp(sum(map(math.log, medians)) / len(medians))
    memory_passed = all(
        native["memory"]["steady_peak_allocated_bytes"]
        <= max(
            int(upstream["memory"]["steady_peak_allocated_bytes"] * 1.02),
            upstream["memory"]["steady_peak_allocated_bytes"] + 64 * 1024 * 1024,
        )
        for upstream, native in pairs
    )
    return {
        "geometric_mean_optimizer_step_ratio": geometric,
        "hierarchical_ratio_ci95": {
            "median": 1 + median_pct / 100,
            "lower": 1 + lower_pct / 100,
            "upper": 1 + upper_pct / 100,
        },
        "flat_diagnostic_ratio_ci95": {"lower": flat_lower, "upper": flat_upper},
        "per_seed_median_ratios": medians,
        "per_seed_p95_ratios": p95_ratios,
        "memory_passed": memory_passed,
        "passed": bool(
            geometric < 1.0
            and 1 + upper_pct / 100 < 1.0
            and max(medians) <= 1.05
            and max(p95_ratios) <= 1.10
            and memory_passed
        ),
    }


def _confirmation(args) -> None:
    import torch
    from provenance import provenance, utc_now, write_artifact

    from urm.adapters.sparse_delta_memory import probe_sdm_support

    payload, _config = load_frozen_config(diagnostic=args.diagnostic)
    modes = payload["execution"]["modes"]
    seeds = payload["measurement"]["paired_seeds"]
    results = {}
    controls = {}
    with tempfile.TemporaryDirectory(prefix="urm-pretraining-step-") as root:
        root_path = Path(root)
        launch_audits = {}
        for mode in modes:
            audit_output = args.output.with_name(
                f"{args.output.stem}-launch-{mode}.json"
            )
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = str(root_path / f"audit-cache-{mode}")
            subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__)
                        .with_name("audit_sparse_memory_launches.py")
                        .resolve()
                    ),
                    "--mode",
                    mode,
                    "--output",
                    str(audit_output),
                ],
                env=env,
                check=True,
            )
            launch_audits[mode] = json.loads(audit_output.read_text())
        for mode in modes:
            pairs = []
            correctness = []
            internal_correctness = []
            orders = []
            for seed in seeds:
                order = ["upstream_sdm", "urm_native"]
                random.Random(seed + 77).shuffle(order)
                orders.append(order)
                rows = {}
                for backend in order:
                    output = root_path / f"{mode}-{seed}-{backend}.json"
                    cache = root_path / f"cache-{mode}-{seed}-{backend}"
                    env = os.environ.copy()
                    env["TRITON_CACHE_DIR"] = str(cache / "triton")
                    env["TORCH_EXTENSIONS_DIR"] = str(cache / "extensions")
                    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        "--backend",
                        backend,
                        "--mode",
                        mode,
                        "--seed",
                        str(seed),
                        "--output",
                        str(output),
                    ]
                    if args.diagnostic:
                        command.append("--diagnostic")
                    completed = subprocess.run(command, env=env, check=False)
                    if completed.returncode:
                        raise RuntimeError(
                            f"pretraining child failed: mode={mode}, seed={seed}, backend={backend}"
                        )
                    rows[backend] = json.loads(output.read_text(encoding="utf-8"))
                pair = (rows["upstream_sdm"], rows["urm_native"])
                pairs.append(pair)
                correctness.append(
                    _compare_correctness(
                        *pair, torch, payload["correctness"]["bfloat16"]
                    )
                )
                internal_correctness.append(
                    _compare_correctness(
                        *pair, torch, payload["correctness"]["bfloat16"], internal=True
                    )
                )
                # Full state snapshots remain until cross-mode comparison;
                # internal-gradient tensors have now been reduced to durable
                # metrics and need not occupy disk for the rest of the grid.
                for row in pair:
                    for path in row.pop("gradient_files", []):
                        Path(path).unlink()
            ratio = _ratio_summary(pairs)
            graph_passed = all(
                row["compile"]["graph_break_count"] == 0
                and row["compile"]["recompilations"] == 0
                and (
                    mode != "compile_fullgraph" or row["compile"]["unique_graphs"] == 1
                )
                for pair in pairs
                for row in pair
            )
            results[mode] = {
                "orders": orders,
                "pairs": [{"upstream": pair[0], "native": pair[1]} for pair in pairs],
                "correctness": correctness,
                "eager_internal_gradient_diagnostics": internal_correctness,
                "performance": ratio,
                "graph_passed": graph_passed,
                "passed": ratio["passed"]
                and graph_passed
                and all(item["passed"] for item in correctness + internal_correctness),
            }
            # Contextual throughput control only: attention is not a semantic
            # comparator and therefore does not enter sparse correctness or
            # superiority ratios.
            control_rows = []
            for seed in seeds:
                output = root_path / f"{mode}-{seed}-sdpa.json"
                cache = root_path / f"cache-{mode}-{seed}-sdpa"
                env = os.environ.copy()
                env["TRITON_CACHE_DIR"] = str(cache / "triton")
                env["TORCH_EXTENSIONS_DIR"] = str(cache / "extensions")
                env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--backend",
                    "sdpa",
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                ]
                if args.diagnostic:
                    command.append("--diagnostic")
                completed = subprocess.run(command, env=env, check=False)
                if completed.returncode:
                    raise RuntimeError(
                        f"SDPA contextual control failed: mode={mode}, seed={seed}"
                    )
                control_rows.append(json.loads(output.read_text(encoding="utf-8")))
            controls[mode] = {
                "semantic_role": "contextual_control_not_sparse_equivalence_comparator",
                "runs": control_rows,
                "graph_passed": all(
                    row["compile"]["graph_break_count"] == 0
                    and row["compile"]["recompilations"] == 0
                    and (
                        mode != "compile_fullgraph"
                        or row["compile"]["unique_graphs"] == 1
                    )
                    for row in control_rows
                ),
            }
            # Raw gradient tensors are consumed above; the durable artifact
            # retains all requested cosine/error metrics without dangling
            # temporary file paths.
            for pair in pairs:
                pair[0].pop("gradient_files", None)
                pair[1].pop("gradient_files", None)
            for row in control_rows:
                for path in row.pop("gradient_files", []):
                    Path(path).unlink()
        support = probe_sdm_support()
        if not support.supported:
            raise RuntimeError(
                f"pinned comparator unavailable [{support.code}]: {support.reason}"
            )
        artifact_provenance = provenance(" ".join(sys.argv), payload, include_gpu=True)
        artifact_provenance["upstream"] = support.details
        correctness_passed = all(
            result["graph_passed"]
            and all(
                item["passed"]
                for item in result["correctness"]
                + result["eager_internal_gradient_diagnostics"]
            )
            for result in results.values()
        )
        mode_comparisons = {}
        for backend in ("upstream", "native"):
            mode_comparisons[backend] = [
                _compare_correctness(
                    eager[backend],
                    compiled[backend],
                    torch,
                    payload["correctness"]["bfloat16"],
                )
                for eager, compiled in zip(
                    results["eager"]["pairs"],
                    results["compile_fullgraph"]["pairs"],
                    strict=True,
                )
            ]
        correctness_passed = (
            correctness_passed
            and all(audit["passed"] for audit in launch_audits.values())
            and all(row["passed"] for rows in mode_comparisons.values() for row in rows)
        )
        for result in results.values():
            for pair in result["pairs"]:
                for row in pair.values():
                    row.pop("state_files", None)
        for control in controls.values():
            for row in control["runs"]:
                row.pop("state_files", None)
        acceptance_passed = (
            all(result["passed"] for result in results.values()) and correctness_passed
        )
        if acceptance_passed:
            decision = "superiority"
        elif correctness_passed and any(
            result["performance"]["geometric_mean_optimizer_step_ratio"] > 1.0
            for result in results.values()
        ):
            decision = "regression"
        elif correctness_passed:
            decision = "parity_without_superiority"
        else:
            decision = "correctness_failure"
        artifact = {
            "schema_version": 1,
            "artifact_kind": "pretraining_step_confirmation",
            "generated_utc": utc_now(),
            "provenance": artifact_provenance,
            "frozen_configuration": payload,
            "modes": results,
            "contextual_controls": controls,
            "eager_vs_compiled_correctness": mode_comparisons,
            "production_launch_audits": launch_audits,
            "correctness_passed": correctness_passed,
            "decision": decision,
            "passed": acceptance_passed,
        }
        _validate_confirmation(artifact)
        write_artifact(args.output, artifact)
        if not artifact["passed"]:
            raise RuntimeError(
                f"model-level completion gates did not pass; evidence retained at {args.output}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=["upstream_sdm", "urm_native", "sdpa"])
    parser.add_argument("--mode", choices=["eager", "compile_fullgraph"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.child:
        if args.backend is None or args.mode is None or args.seed is None:
            raise ValueError("child mode requires backend, mode, and seed")
        _child(args)
    else:
        _confirmation(args)


if __name__ == "__main__":
    main()
