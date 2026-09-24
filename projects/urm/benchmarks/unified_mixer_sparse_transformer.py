"""Pinned OpenAI dense Sparse Transformer mask equations vs unified K1."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from provenance import provenance, utc_now, write_artifact
from benchmarks.comparators.sparse_transformer import fixed_mode_mask, load_dense_attention
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe


SOURCE_MODES = {
    "sparse_all": ("all", None),
    "sparse_local": ("local", 32),
    "sparse_strided": ("strided", 8),
    "sparse_fixed": ("fixed", 64),
}


def _error(left, right):
    return float(np.max(np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))))


def _profile_pair(source_forward, compiled_forward, source_backward, compiled_backward, pairs, warmup):
    for _ in range(warmup):
        source_forward()
        compiled_forward()
        source_backward()
        compiled_backward()
    measurements = {}
    for mode in ("forward", "forward_backward"):
        source_samples, compiled_samples, overheads, order = [], [], [], []
        for index in range(pairs):
            names = ("source", "compiled") if index % 2 == 0 else ("compiled", "source")
            order.append("".join(names))
            elapsed = {}
            for name in names:
                start = time.perf_counter()
                if mode == "forward":
                    source_forward() if name == "source" else compiled_forward()
                else:
                    source_backward() if name == "source" else compiled_backward()
                if name == "compiled":
                    torch.cuda.synchronize()
                else:
                    # TensorFlow synchronous execution is enabled for this process.
                    pass
                elapsed[name] = time.perf_counter() - start
            source_samples.append(elapsed["source"] * 1000)
            compiled_samples.append(elapsed["compiled"] * 1000)
            overheads.append((elapsed["compiled"] - elapsed["source"]) / elapsed["source"])
        median_overhead = statistics.median(overheads)
        measurements[mode] = {
            "upstream_wall": {
                "sample_count": pairs,
                "median_ms": statistics.median(source_samples),
                "raw_samples_ms": source_samples,
            },
            "compiled_wall": {
                "sample_count": pairs,
                "median_ms": statistics.median(compiled_samples),
                "raw_samples_ms": compiled_samples,
            },
            "paired_compiled_overhead_fraction": {
                "median": median_overhead,
                "raw_samples": overheads,
                "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
            },
            "pair_order": order,
        }
    return measurements


def _run(pairs, warmup):
    if not torch.cuda.is_available():
        raise RuntimeError("Sparse Transformer profiles require CUDA")
    import tensorflow as tf

    tf.config.experimental.set_synchronous_execution(True)
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("the pinned TensorFlow source comparator requires a GPU")
    try:
        tf.config.experimental.set_memory_growth(devices[0], True)
    except RuntimeError:
        pass
    source, identity = load_dense_attention()
    batch, sequence, heads, head_dim = 1, 128, 4, 32
    width = heads * head_dim
    generator = np.random.default_rng(72072)
    base = tuple(
        generator.normal(0.0, 0.1, size=(batch, sequence, width)).astype(np.float32)
        for _ in range(3)
    )
    source_values = tuple(tf.Variable(value) for value in base)
    compiled_values = tuple(
        torch.tensor(value, device="cuda", requires_grad=True) for value in base
    )
    plan = compile_mixer(
        named_mixer_recipe("sparse_attention_core"),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    results = {}
    for case_name, (mode, context) in SOURCE_MODES.items():
        if mode == "fixed":
            expanded_mask = fixed_mode_mask(
                source,
                n_ctx=sequence,
                heads=heads,
                block_size=32,
                local_attn_ctx=context,
                num_verts=2,
                vertsize=1,
            )
            source_mask = expanded_mask[None].astype(np.float32)
        else:
            source_mask = source.get_attn_mask(sequence, mode, context).numpy()
        torch_mask = torch.as_tensor(source_mask, dtype=torch.bool, device="cuda")

        def source_forward():
            if mode == "fixed":
                q, k, v = (source.split_heads(tensor, heads) for tensor in source_values)
                scores = tf.matmul(q, k, transpose_b=True) * (head_dim ** -0.5)
                mask = tf.convert_to_tensor(source_mask, dtype=tf.float32)
                weights = scores * mask + -1e9 * (1.0 - mask)
                weights = tf.nn.softmax(weights)
                return source.merge_heads(tf.matmul(weights, v))
            return source.attention_impl(
                *source_values, heads=heads, attn_mode=mode, local_attn_ctx=context
            )

        def compiled_forward():
            q, k, v = (
                tensor.reshape(batch, sequence, heads, head_dim)
                for tensor in compiled_values
            )
            return plan.execute(
                query=q, key=k, value=v, attention_mask=torch_mask
            ).output.reshape(batch, sequence, width)

        def source_backward():
            with tf.GradientTape() as tape:
                output = source_forward()
                loss = tf.reduce_mean(tf.square(output))
            return tape.gradient(loss, source_values)

        def compiled_backward():
            output = compiled_forward()
            return torch.autograd.grad(output.square().mean(), compiled_values)

        expected = source_forward()
        expected_grads = source_backward()
        actual = compiled_forward()
        actual_grads = compiled_backward()
        output_error = _error(actual.detach().cpu().numpy(), expected.numpy())
        gradient_errors = [
            _error(actual_grad.detach().cpu().numpy(), expected_grad.numpy())
            for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True)
        ]
        if output_error > 2e-4 or max(gradient_errors) > 3e-4:
            raise AssertionError(
                f"{case_name} source parity failed: output={output_error}, grads={gradient_errors}"
            )
        measurements = _profile_pair(
            source_forward,
            compiled_forward,
            source_backward,
            compiled_backward,
            pairs,
            warmup,
        )
        results[case_name] = {
            "mode": mode,
            "context": context,
            "upstream_callable": (
                "dense transcription of blocksparse_attention_impl(fixed) using pinned get_blocksparse_obj/get_callback"
                if mode == "fixed"
                else f"attention_impl(attn_mode={mode!r}, local_attn_ctx={context!r})"
            ),
            "shape": {
                "batch": batch,
                "sequence": sequence,
                "heads": heads,
                "head_dim": head_dim,
                "dtype": "float32",
            },
            "output_error": output_error,
            "gradient_errors": gradient_errors,
            "measurements": measurements,
        }
    return identity, plan, results


def run(pairs: int, warmup: int, output_path: Path):
    identity, plan, results = _run(pairs, warmup)
    cases = {}
    for case_name, result in results.items():
        cases[case_name] = {
            "architecture_ids": ["arch-072"],
            "semantic_scope": f"Pinned OpenAI dense attention mask mode={result['mode']} from attention_impl or the exact fixed-mode BlocksparseTransformer layout/callback; optimized block traversal is not included",
            "shape": result["shape"],
            "upstream_callable": result["upstream_callable"],
            "compiled_anchor": plan.anchor,
            "parity": {
                "status": "pass",
                "output_max_abs_error": result["output_error"],
                "input_and_parameter_gradient_max_abs_errors": result["gradient_errors"],
                "tolerances": {
                    "output_atol": 2e-4,
                    "gradient_atol": 3e-4,
                    "relative_tolerance": 5e-4,
                },
                "independent_reference": (
                    "tests/test_unified_mixer.py compares the fixed-mode mask generated by pinned get_blocksparse_obj/get_callback and a dense TensorFlow equation against the unified K1 plan"
                    if result["mode"] == "fixed"
                    else "tests/test_unified_mixer.py compares pinned TensorFlow attention_impl outputs and gradients against the unified K1 plan"
                ),
            },
            "performance": {
                "comparison": "pinned TensorFlow 2 eager attention_impl for all/local/strided, or dense fixed-mask equation from pinned layout/callback functions, vs PyTorch SDPA K1; the legacy custom blocksparse kernel is not timed",
                "measurements": result["measurements"],
            },
        }
    payload = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "purpose": "compare unified sparse-mask K1 against pinned OpenAI Sparse Transformer dense source equations",
        "upstream": {
            "repository": identity["repository"],
            "revision": identity["revision"],
            "loaded_module": identity["source_path"],
            "kernel_source_sha256": {"attention.py": identity["source_sha256"]},
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/sparse_attention-checkout:src python benchmarks/unified_mixer_sparse_transformer.py",
            {"pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tensorflow": __import__("tensorflow").__version__,
        },
        "methodology": {
            "timed_work": "pinned source attention_impl including all/local/strided mask generation, or dense fixed-mask equation using the pinned block layout/callback, vs unified K1 with the equivalent caller-supplied mask; output-only and input-gradient modes",
            "sampling": "interleaved alternating order with synchronized TensorFlow eager execution and CUDA synchronization for PyTorch",
            "warmup": warmup,
            "pairs": pairs,
            "overhead_gate_fraction": 0.10,
        },
        "cases": cases,
    }
    write_artifact(output_path, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/sparse-transformer-k1.json"))
    args = parser.parse_args()
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
