"""Numerical-only comparison of triangular chunks to two distinct references."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from provenance import provenance, utc_now, write_artifact

from urm.adapters.sparse_delta_memory import (
    MODE_READ_ONLY,
    MODE_TRAINING,
    UrmSparseDeltaMemoryAdapter,
    probe_sdm_support,
)
from urm.backends.sparse_state_reference import torch_sparse_state_mixer
from urm.backends.sparse_state_triangular import torch_selected_slot_triangular
from urm.compiler.semantic import SparseReadTiming

NAMES = (
    "initial_memory",
    "write_weights",
    "values",
    "beta",
    "log_decay",
    "read_weights",
)
# Frozen verbatim from test_sparse_state_mixer_gpu.py and
# test_sparse_state_mixer_upstream_gpu.py. No model or E2E gates are replaced.
ORACLE_FORWARD = {
    "float32": {"atol": 2e-5, "rtol": 2e-5},
    "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
}
UPSTREAM_FORWARD = {
    "float32": {"atol": 2e-2, "rtol": 3e-3},
    "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
}
BACKWARD = {
    "float32": {"atol": 3e-5, "rtol": 3e-4},
    "bfloat16": {"atol": 3e-2, "rtol": 3e-2},
}


def pad_upstream_partition(wi, w, v, b, g, ri, q):
    """Pinned forward/backward require a multiple of 16 even for P=1.

    Dummy writes have E=I, k=0, beta=0, hence M_next=M; dummy queries
    have zero weight. Only the comparator's physical call is padded.
    """
    padding = (-wi.shape[1]) % 16
    if not padding:
        return wi, w, v, b, g, ri, q

    def extend(tensor, *, indices=False):
        extra = (
            tensor[:, -1:].expand(-1, padding, -1)
            if indices
            else tensor.new_zeros((1, padding, tensor.shape[-1]))
        )
        return torch.cat((tensor, extra), 1).contiguous()

    return (
        extend(wi, indices=True),
        extend(w),
        extend(v),
        extend(b),
        extend(g),
        extend(ri, indices=True),
        extend(q),
    )


def make_case(dtype, sequence, pattern, width):
    p, slots, dim = 2, 4096, 64
    generator = torch.Generator(device="cuda").manual_seed(8221 + sequence + width)
    token = torch.arange(sequence, device="cuda")[:, None]
    route = torch.arange(width, device="cuda")[None]
    if pattern in ("repeated", "rounding_stress"):
        wi = route.expand(sequence, -1)
        ri = wi if pattern == "rounding_stress" else (wi + width // 2) % slots
    elif pattern == "disjoint":
        # Every token's selected slots are distinct, including across chunks.
        assert sequence * width <= slots
        wi = token * width + route
        ri = wi
    else:
        wi = (token * 3 + route * 7) % slots
        ri = (token * 3 + route * 7 + 1) % slots
    wi = wi.sort(-1).values[None].expand(p, -1, -1).contiguous()
    ri = ri.sort(-1).values[None].expand(p, -1, -1).contiguous()
    memory = (
        torch.randn(p, slots, dim, device="cuda", generator=generator).to(dtype) * 0.05
    )
    values = (
        torch.randn(p, sequence, dim, device="cuda", generator=generator).to(dtype)
        * 0.05
    )
    ww = torch.softmax(
        torch.randn(p, sequence, width, device="cuda", generator=generator).to(dtype),
        -1,
    )
    rw = torch.softmax(
        torch.randn(p, sequence, width, device="cuda", generator=generator).to(dtype),
        -1,
    )
    beta = torch.rand(p, sequence, 1, device="cuda", generator=generator).to(dtype)
    decay = (
        -torch.rand(p, sequence, 1, device="cuda", generator=generator).to(dtype) * 0.1
    )
    if pattern == "rounding_stress":
        memory[:, :width] = 1
        values.zero_()
        ww.fill_(1 / width)
        rw.fill_(1 / width)
        beta.fill_(0.001)
        decay.fill_(-0.001)
    reading_ct = (
        torch.randn(values.shape, device="cuda", generator=generator) / values.numel()
    )
    final_ct = (
        torch.randn(memory.shape, device="cuda", generator=generator) / memory.numel()
    )
    return wi, ri, [memory, ww, values, beta, decay, rw], reading_ct, final_ct


def run_reference(kind, sample, chunk, timing):
    wi, ri, data, reading_ct, final_ct = sample
    leaves = [x.detach().clone().requires_grad_() for x in data]
    m, w, v, b, g, q = leaves
    if kind == "pinned_upstream":
        # Keep the frozen model comparator chunk=16 for all candidate chunks.
        # Per-partition calls make partial chunks independent: upstream's
        # flattened P*T path assumes partition/chunk alignment.
        adapter = UrmSparseDeltaMemoryAdapter(
            slots_per_partition=m.shape[1],
            value_dim=m.shape[2],
            num_writes=w.shape[-1],
            num_reads=q.shape[-1],
            chunk_size=16,
            mode=MODE_TRAINING,
            dtype=m.dtype,
        )
        outputs, finals = [], []
        for p in range(m.shape[0]):
            read_idx, read_w = ri[p : p + 1], q[p : p + 1]
            if timing is SparseReadTiming.BEFORE_UPDATE:
                # Before(t) is After(t-1) evaluated with query(t). The first
                # query reads the initial memory directly. No kernel is changed.
                read_idx = torch.cat(
                    (read_idx[:, 1:], read_idx[:, -1:]), 1
                ).contiguous()
                read_w = torch.cat(
                    (read_w[:, 1:], torch.zeros_like(read_w[:, -1:])), 1
                ).contiguous()
            padded = pad_upstream_partition(
                wi[p : p + 1],
                w[p : p + 1],
                v[p : p + 1],
                b[p : p + 1],
                g[p : p + 1],
                read_idx,
                read_w,
            )
            output, final = adapter.direct_calls["update"](
                m[p] + 0,
                *padded,
                grad_final_memory=final_ct[p].to(m.dtype).contiguous(),
            )
            output = output[:, : v.shape[1]]
            finals.append(final.detach().clone())  # backward restores working memory
            if timing is SparseReadTiming.BEFORE_UPDATE:
                first = (
                    (m[p, ri[p, 0]].float() * q[p, 0].float()[:, None])
                    .sum(0)
                    .to(m.dtype)
                )
                output = torch.cat((first[None, None], output[:, :-1]), 1)
            outputs.append(output)
        y, final = torch.cat(outputs), torch.stack(finals)
        # The final-memory VJP is injected via the pinned upstream API. The
        # detached term below records the same scalar joint loss without double
        # counting that cotangent through an unrelated autograd path.
    else:
        fn = (
            torch_sparse_state_mixer
            if kind == "per_token_cast_oracle"
            else torch_selected_slot_triangular
        )
        options = {"chunk_size": chunk} if kind == "triangular_candidate" else {}
        y, final = fn(
            m,
            ri,
            q,
            write_indices=wi,
            write_weights=w,
            values=v,
            beta=b,
            log_decay=g,
            read_timing=timing,
            **options,
        )
    loss = (y.float() * reading_ct).sum() + (final.float() * final_ct).sum()
    gradients = torch.autograd.grad(loss, leaves)
    return {
        "readings": y.detach().float().cpu(),
        "final_memory": final.detach().float().cpu(),
        "gradients": {
            name: grad.detach().float().cpu()
            for name, grad in zip(NAMES, gradients, strict=True)
        },
        "joint_loss": float(loss.detach()),
    }


def discrepancy(candidate, reference, tolerance):
    delta = (candidate - reference).abs()
    allowance = tolerance["atol"] + tolerance["rtol"] * reference.abs()
    finite = bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())
    return {
        "max_abs": float(delta.max()),
        "rms": float(delta.square().mean().sqrt()),
        "max_tolerance_ratio": float((delta / allowance).max()),
        "violating_elements": int((delta > allowance).sum()),
        "elements": delta.numel(),
        "finite": finite,
        "passed": finite and bool((delta <= allowance).all()),
    }


def compare(candidate, reference, dtype, reference_name):
    forward = (
        ORACLE_FORWARD[dtype]
        if reference_name == "per_token_cast_oracle"
        else UPSTREAM_FORWARD[dtype]
    )
    tensors = {
        name: discrepancy(candidate[name], reference[name], forward)
        for name in ("readings", "final_memory")
    }
    gradients = {
        name: discrepancy(
            candidate["gradients"][name], reference["gradients"][name], BACKWARD[dtype]
        )
        for name in candidate["gradients"]
    }
    return {
        "forward": tensors,
        "gradients": gradients,
        "joint_loss_abs": abs(candidate["joint_loss"] - reference["joint_loss"]),
        "passed": all(
            row["passed"] for row in (*tensors.values(), *gradients.values())
        ),
    }


def read_only_case(dtype, chunk):
    wi, ri, data, ct, mt = make_case(dtype, chunk + 3, "mixed", 64)
    del wi
    rows = {}
    for kind in ("per_token_cast_oracle", "pinned_upstream", "triangular_candidate"):
        m, q = (x.detach().clone().requires_grad_() for x in (data[0], data[5]))
        if kind == "pinned_upstream":
            adapter = UrmSparseDeltaMemoryAdapter(
                slots_per_partition=m.shape[1],
                value_dim=m.shape[2],
                num_writes=1,
                num_reads=q.shape[-1],
                mode=MODE_READ_ONLY,
                dtype=dtype,
            )
            offsets = (
                torch.arange(m.shape[0], device=m.device)[:, None, None] * m.shape[1]
            )
            y = adapter.direct_calls["read"](
                m.flatten(0, 1), q, (ri + offsets).contiguous()
            )
            final = m.clone()
        else:
            fn = (
                torch_sparse_state_mixer
                if kind == "per_token_cast_oracle"
                else torch_selected_slot_triangular
            )
            options = {"chunk_size": chunk} if kind == "triangular_candidate" else {}
            y, final = fn(
                m, ri, q, read_timing=SparseReadTiming.CURRENT_STATE, **options
            )
        loss = (y.float() * ct).sum() + (final.float() * mt).sum()
        grads = torch.autograd.grad(loss, (m, q))
        rows[kind] = {
            "readings": y.detach().float().cpu(),
            "final_memory": final.detach().float().cpu(),
            "gradients": {
                name: grad.detach().float().cpu()
                for name, grad in zip((NAMES[0], NAMES[-1]), grads, strict=True)
            },
            "joint_loss": float(loss.detach()),
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pretraining-step/triangular-numerical-v1.json"),
    )
    args = parser.parse_args()
    support = probe_sdm_support()
    if not support.supported:
        raise RuntimeError(f"pinned upstream unavailable: {support}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    configuration = {
        "chunks": [16, 32, 64],
        "dtypes": ["float32", "bfloat16"],
        "patterns": ["repeated", "disjoint", "mixed", "rounding_stress"],
        "read_timings": ["before_update", "after_update", "current_state_read_only"],
        "oracle_forward_tolerances": ORACLE_FORWARD,
        "upstream_forward_tolerances": UPSTREAM_FORWARD,
        "backward_tolerances": BACKWARD,
        "upstream_chunk_size": 16,
        "upstream_partial_chunk_policy": "identity_update_and_zero_query_padding_to_multiple_of_16; trim_dummy_readings; retain_final_state_cotangent",
        "loss": "sum(readings * random_cotangent / readings.numel) + sum(final_memory * random_cotangent / memory.numel)",
        "cast_policy": "candidate_casts_readings_and_chunk_boundaries_only; oracle_retains_every_token_state_cast",
        "model_integration": False,
        "custom_gpu_kernels": False,
    }
    rows = []
    for chunk, dtype_name, pattern, timing in itertools.product(
        configuration["chunks"],
        configuration["dtypes"],
        configuration["patterns"],
        (SparseReadTiming.BEFORE_UPDATE, SparseReadTiming.AFTER_UPDATE),
    ):
        dtype = getattr(torch, dtype_name)
        width = 64
        partial_length = (
            129
            if pattern == "rounding_stress"
            else min(2 * chunk + 3, 63)
            if pattern == "disjoint"
            else 2 * chunk + 3
        )
        for length in (chunk, partial_length):
            sample = make_case(dtype, length, pattern, width)
            results = {
                kind: run_reference(kind, sample, chunk, timing)
                for kind in (
                    "per_token_cast_oracle",
                    "pinned_upstream",
                    "triangular_candidate",
                )
            }
            comparison = {
                ref: compare(
                    results["triangular_candidate"], results[ref], dtype_name, ref
                )
                for ref in ("per_token_cast_oracle", "pinned_upstream")
            }
            row = {
                "chunk_size": chunk,
                "dtype": dtype_name,
                "pattern": pattern,
                "sequence": length,
                "parallel": 2,
                "slots": 4096,
                "value_dim": 64,
                "writes": width,
                "reads": width,
                "initial_state": "nonzero",
                "read_timing": timing.value,
                "partial_chunk": length % chunk != 0,
                "upstream_physical_sequence": ((length + 15) // 16) * 16,
                "differentiable_inputs": list(NAMES),
                "references": comparison,
                "upstream_vs_oracle_diagnostic": compare(
                    results["pinned_upstream"],
                    results["per_token_cast_oracle"],
                    dtype_name,
                    "per_token_cast_oracle",
                ),
                "passed": all(value["passed"] for value in comparison.values()),
            }
            rows.append(row)
            print(
                f"chunk={chunk} {dtype_name} {pattern} T={length} {timing.value}: "
                + ", ".join(
                    f"{name}={'pass' if result['passed'] else 'FAIL'}"
                    for name, result in comparison.items()
                ),
                flush=True,
            )
    for chunk, dtype_name in itertools.product(
        configuration["chunks"], configuration["dtypes"]
    ):
        results = read_only_case(getattr(torch, dtype_name), chunk)
        comparison = {
            ref: compare(results["triangular_candidate"], results[ref], dtype_name, ref)
            for ref in ("per_token_cast_oracle", "pinned_upstream")
        }
        rows.append(
            {
                "chunk_size": chunk,
                "dtype": dtype_name,
                "pattern": "read_only",
                "sequence": chunk + 3,
                "read_timing": "current_state",
                "initial_state": "nonzero",
                "partial_chunk": True,
                "differentiable_inputs": [NAMES[0], NAMES[-1]],
                "references": comparison,
                "passed": all(value["passed"] for value in comparison.values()),
            }
        )
    accepted = all(row["passed"] for row in rows)
    artifact = {
        "schema_version": 1,
        "artifact_kind": "selected_slot_triangular_numerical_prototype",
        "generated_utc": utc_now(),
        "provenance": provenance(
            "sparse_state_triangular.py", configuration, include_gpu=True
        ),
        "upstream": support.details,
        "configuration": configuration,
        "cases": rows,
        "reference_passed": {
            ref: all(row["references"][ref]["passed"] for row in rows)
            for ref in ("per_token_cast_oracle", "pinned_upstream")
        },
        "numerically_accepted": accepted,
        "gpu_prototype_authorized_by_numerics": accepted,
        "decision": "numerically_accepted_pending_cost_projection"
        if accepted
        else "rejected_before_kernel_integration",
    }
    from jsonschema import validate

    validate(
        artifact,
        json.loads(
            Path(__file__)
            .with_name("sparse-state-triangular-result-schema.json")
            .read_text()
        ),
    )
    write_artifact(args.output, artifact)
    if not accepted:
        raise SystemExit(
            "Numerical contract failed; evidence retained. No kernel integration."
        )


if __name__ == "__main__":
    main()
