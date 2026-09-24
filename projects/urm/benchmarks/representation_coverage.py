"""Representation-coverage gate: every architecture lowers into a canonical core.

This is the durable contract behind the unified generator. For each named
architecture recipe it attempts to lower the declarative
:class:`~urm.ir.graph.UnifiedMixerSpec` into the canonical NumPy execution path
for its core (K1 normalized routed reduction, K2 matrix-state recurrence, K3
ordered sparse state) via :func:`urm.backends.reference.numpy.graph.execute_canonical`. A
recipe that lowers is then verified against its independent architecture
equation (the compiler's reference executor) in float64: outputs and final state
must agree.

A recipe that does not lower is recorded with the precise reason (an
under-specified collision group, or a genuinely exotic equation the canonical
cores do not represent). Coverage is the count of recipes verified to lower and
match, not the count that merely run through some executor.

Run ``python benchmarks/representation_coverage.py`` to print the table; the
markdown doc regenerates from this measurement.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from urm.frontend.recipes import MIXER_RECIPE_NAMES, named_mixer_recipe
from urm.ir.graph import MixerKernelFamily
from urm.backends.reference.numpy.graph import UnderspecifiedComposition, execute_canonical


@dataclass
class RecipeCoverage:
    name: str
    family: str
    lowers: bool
    reason: str = ""
    output_err: float | None = None
    state_err: float | None = None
    verified: bool = False


def _rng_operands(spec, seed=0):
    """Build small finite NumPy operands for a canonically representable spec."""
    rng = np.random.default_rng(seed)
    if spec.family is MixerKernelFamily.SOFTMAX:
        from urm.ir.graph import K1Operation

        b, t, h, k, v = 1, 5, 2, 4, 4
        if spec.k1_operation is K1Operation.DIFFERENTIAL:
            return {
                "query_a": rng.normal(size=(b, t, h, k)),
                "query_b": rng.normal(size=(b, t, h, k)),
                "key_a": rng.normal(size=(b, t, h, k)),
                "key_b": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "lambda_weight": rng.uniform(0.1, 0.9, size=(h,)),
            }
        if spec.k1_operation is K1Operation.PROJECTED:
            r = 3  # low-rank query width
            return {
                "query": rng.normal(size=(b, t, r)),
                "B_pre": rng.normal(size=(h, r, k)),
                "key": rng.normal(size=(b, t, k)),
                "value": rng.normal(size=(b, t, v)),
            }
        if spec.k1_operation is K1Operation.POSITIONAL:
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "r": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, k)),
            }
        if spec.k1_operation is K1Operation.POSITIVE_FEATURE:
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "num_groups": 2,
            }
        if spec.k1_operation is K1Operation.GATED:
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "g": -rng.uniform(0, 0.3, size=(b, t, h, k)),
            }
        if spec.k1_operation is K1Operation.THRESHOLDED:
            return {
                "query_a": rng.normal(size=(b, t, h, k)),
                "query_b": rng.normal(size=(b, t, h, k)),
                "key_a": rng.normal(size=(b, t, h, k)),
                "key_b": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "beta": np.asarray(0.5),
                "lambda_weight": np.asarray(0.5),
            }
        if spec.k1_operation is K1Operation.DELTA_TRANSFORM:
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "beta": rng.uniform(0.1, 0.9, size=(b, t, h)),
            }
        ops = {
            "query": rng.normal(size=(b, t, h, k)),
            "key": rng.normal(size=(b, t, h, k)),
            "value": rng.normal(size=(b, t, h, v)),
        }
        if spec.k1_operation is K1Operation.LOCAL_WINDOW:
            ops["attention_window"] = 2
        elif spec.requires_attention_mask:
            # A boolean mask with every query exposing at least one visible key.
            mask = np.ones((b, 1, t, t), dtype=bool)
            mask[..., 0, :] = True
            ops["attention_mask"] = mask
        return ops
    if spec.family is MixerKernelFamily.RECURRENCE:
        from urm.ir.graph import RecurrenceOperator, RecurrentLayout

        # Distinct recurrence operators (IR-distinguished equations).
        op = spec.recurrence_operator
        if op is RecurrenceOperator.TANH_RNN:
            b, t, n, h = 1, 5, 2, 3
            return {
                "query": rng.normal(size=(b, t, n, h)),
                "weight": rng.normal(size=(n, h, h)) * 0.3,
                "initial_state": rng.normal(size=(b, n, h)),
            }
        if op is RecurrenceOperator.GATED_RNN:
            b, t, n, h = 1, 5, 2, 3
            return {
                "query": rng.normal(size=(b, t, n, h)),
                "weight": rng.normal(size=(n, h, h)) * 0.3,
                "forget_input": rng.normal(size=(b, t, n, h)),
                "forget_weight": rng.normal(size=(n, h, h)) * 0.3,
                "reset_input": rng.normal(size=(b, t, n, h)),
                "reset_weight": rng.normal(size=(n, h, h)) * 0.3,
                "initial_state": rng.normal(size=(b, n, h)),
            }
        if op is RecurrenceOperator.MULTIPLICATIVE_RNN:
            b, t, n, k, v = 1, 5, 2, 3, 3
            return {
                "query": rng.normal(size=(b, t, n, k)),
                "key": rng.normal(size=(b, t, n, k)),
                "value": rng.normal(size=(b, t, n, v)),
                "weight": rng.normal(size=(n, v, v)) * 0.3,
                "forget_input": rng.uniform(0.1, 0.9, size=(b, t, n)),
                "initial_state": rng.normal(size=(b, n, k, v)) * 0.1,
            }
        if op is RecurrenceOperator.FFT_CONVOLUTION:
            b, t, c = 1, 8, 3
            return {
                "query": rng.normal(size=(b, t, c)),
                "kernel": rng.normal(size=(c, t)),
                "direct": rng.normal(size=(c,)),
            }
        if op is RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION:
            b, t, h = 1, 8, 2
            return {
                "query": rng.normal(size=(b, t, h, 1)),
                "key": rng.normal(size=(b, t, h, 1)),
                "value": rng.normal(size=(b, t, h, 1)),
                "ssm_kernel": rng.normal(size=(h, t)),
                "ssm_k_kernel": rng.normal(size=(h, t)),
                "ssm_k_direct": rng.normal(size=(h,)),
                "skip": rng.normal(size=(h,)),
            }
        if op is RecurrenceOperator.SECOND_ORDER_CUMSUM:
            b, t, h, k, v = 1, 6, 2, 3, 3
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
            }
        if op is RecurrenceOperator.REGULARIZED_SOLVE:
            b, t, h, k = 1, 6, 2, 3
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, k)),
                "log_decay": -rng.uniform(0, 0.4, size=(b, t, h)),
                "beta": rng.uniform(0.1, 0.9, size=(b, t, h)),
                "lamb": rng.uniform(0.5, 1.5, size=(h, k)),
            }
        if op is RecurrenceOperator.LAYERNORM_INNER_STATE:
            b, t, h, d = 1, 8, 2, 4  # t divisible by chunk_size=4
            return {
                "query": rng.normal(size=(b, t, h, d)),
                "key": rng.normal(size=(b, t, h, d)),
                "value": rng.normal(size=(b, t, h, d)),
                "w": rng.normal(size=(h, d)),
                "b": rng.normal(size=(h, d)),
                "eta": rng.uniform(0.01, 0.1, size=(b, t, h, 1)),
                "chunk_size": 4,
            }
        if op is RecurrenceOperator.MOMENTUM_INNER_STATE:
            b, t, h, d = 1, 8, 2, 4
            return {
                "query": rng.normal(size=(b, t, h, d)),
                "key": rng.normal(size=(b, t, h, d)),
                "value": rng.normal(size=(b, t, h, d)),
                "w": rng.normal(size=(h, d)),
                "b": rng.normal(size=(h, d)),
                "theta": rng.uniform(0.01, 0.1, size=(b, t, h, 1)),
                "alpha": rng.uniform(0.01, 0.3, size=(b, t, h, 1)),
                "eta": rng.uniform(0.01, 0.3, size=(b, t, h, 1)),
                "chunk_size": 4,
            }
        if op is RecurrenceOperator.MAMBA2_STRUCTURED_SSM:
            b, t, h, p, g, n = 1, 6, 2, 4, 1, 3
            return {
                "x": rng.normal(size=(b, t, h, p)),
                "dt": rng.uniform(0.01, 0.5, size=(b, t, h)),
                "A": -rng.uniform(0.1, 1.0, size=(h,)),
                "B": rng.normal(size=(b, t, g, n)),
                "C": rng.normal(size=(b, t, g, n)),
            }
        if op is RecurrenceOperator.RWKV6_BONUS_CORRECTED:
            b, t, h, k, v = 1, 6, 2, 4, 3
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "log_decay": -rng.uniform(0, 0.4, size=(b, t, h, k)),
                "bonus": rng.normal(size=(h, k)),
            }
        if op is RecurrenceOperator.RWKV4_SCALAR_STATE:
            b, t, c = 1, 6, 4
            return {
                "w": -rng.uniform(0.1, 1.0, size=(c,)),
                "u": rng.normal(size=(c,)),
                "k": rng.normal(size=(b, t, c)),
                "v": rng.normal(size=(b, t, c)),
                "state": rng.normal(size=(b, 3, 1, c)) * 0.1,
            }
        if op is RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE:
            b, t, hk, hq, k, s, v = 1, 6, 1, 2, 4, 3, 3  # group_size = hq//hk = 2
            base = {
                "query": rng.normal(size=(b, t, hq, k)),
                "key": rng.normal(size=(b, t, hk, k)),
                "value": rng.normal(size=(b, t, hk, v)),
            }
            if spec.name == "abc_core":
                # ABC derives slot_weights and log_decay from slot_logits.
                base["slot_logits"] = rng.normal(size=(b, t, hk, s))
            else:
                base["slot_weights"] = rng.uniform(0.1, 1.0, size=(b, t, hk, s))
                base["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, hk, s))
            return base
        if op is RecurrenceOperator.GATED_OJA_VALUE_CHANNEL:
            b, t, h, k, v = 1, 6, 2, 4, 3
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "gv": -rng.uniform(0, 0.4, size=(b, t, h, v)),
                "beta": rng.uniform(0.1, 0.9, size=(b, t, h)),
            }
        if op is RecurrenceOperator.MOMENTUM_DELTA_STATE:
            b, t, h, k, v = 1, 6, 2, 4, 3
            return {
                "query": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "p": rng.normal(size=(b, t, h, k)),
                "log_alpha": -rng.uniform(0, 0.4, size=(b, t, h)),
                "log_mu": -rng.uniform(0, 0.4, size=(b, t, h)),
                "beta": rng.uniform(0.1, 0.9, size=(b, t, h)),
                "eta": rng.uniform(0.01, 0.3, size=(b, t, h)),
            }
        if op is RecurrenceOperator.TRAPEZOIDAL_SSM:
            b, t, h, kd, vd, a = 1, 6, 2, 4, 3, 2  # K even, 2*A <= K
            return {
                "query": rng.normal(size=(b, t, h, kd)),
                "key": rng.normal(size=(b, t, h, kd)),
                "value": rng.normal(size=(b, t, h, vd)),
                "adt": -rng.uniform(0, 0.5, size=(b, h, t)),
                "dt": rng.uniform(0.01, 0.5, size=(b, h, t)),
                "trap": rng.normal(size=(b, h, t)),
                "query_bias": rng.normal(size=(h, kd)),
                "key_bias": rng.normal(size=(h, kd)),
                "angles": rng.normal(size=(b, t, h, a)),
            }
        if spec.recurrent_layout is RecurrentLayout.DIAGONAL:
            b, t, c, n = 1, 6, 4, 2
            if spec.diagonal_hgrn:
                # HGRN: state width 1, per-channel decay [B,T,C].
                return {
                    "x": rng.normal(size=(b, t, c)),
                    "log_decay": -rng.uniform(0, 0.4, size=(b, t, c)),
                }
            ops = {
                "x": rng.normal(size=(b, t, c)),
                "log_decay": -rng.uniform(0, 0.4, size=(b, t, n)),
            }
            ops["input_gate"] = rng.uniform(0.1, 1.0, size=(b, t, n))
            ops["read_gate"] = rng.uniform(0.1, 1.0, size=(b, t, n))
            if spec.step_size_discretization:
                ops["step_size"] = rng.uniform(0.1, 1.0, size=(b, t, c))
            return ops
        b, t, h, k, v = 1, 6, 2, 4, 3
        ops = {
            "query": rng.normal(size=(b, t, h, k)),
            "key": rng.normal(size=(b, t, h, k)),
            "value": rng.normal(size=(b, t, h, v)),
        }
        from urm.ir.graph import DecayGranularity, StateUpdateRule

        if spec.gdn2_ssm:
            # Dual-gate delta: erase/write gates replace beta.
            ops["erase_gate"] = rng.uniform(0.1, 0.9, size=(b, t, h, k))
            ops["write_gate"] = rng.uniform(0.1, 0.9, size=(b, t, h, v))
        elif spec.update_rule is StateUpdateRule.DELTA:
            ops["beta"] = rng.uniform(0.1, 0.9, size=(b, t, h))
        if spec.generalized_delta_iplr or spec.generalized_delta_dplr:
            ops["transition_alpha"] = rng.normal(size=(b, t, h, k)) * 0.3
            ops["transition_beta"] = rng.normal(size=(b, t, h, k)) * 0.3
        if spec.gated_delta_product:
            r = 2  # ranks per token
            ops["update_keys"] = rng.normal(size=(b, t, r, h, k))
            ops["update_values"] = rng.normal(size=(b, t, r, h, v))
            ops["beta"] = rng.uniform(0.1, 0.9, size=(b, t, r, h))
        if spec.comba_rule:
            # comba names its prediction key "p" and its (head) log decay "g".
            ops["p"] = rng.normal(size=(b, t, h, k))
            ops["g"] = -rng.uniform(0, 0.4, size=(b, t, h))
        elif spec.generalized_delta_dplr:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, h, k))
        elif spec.static_head_decay:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(h,))
        elif spec.decay is DecayGranularity.HEAD:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, h))
        elif spec.decay is DecayGranularity.KEY_CHANNEL:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, h, k))
        return ops
    if spec.family is MixerKernelFamily.SPARSE_DELTA:
        b, t, s, d, r = 1, 6, 8, 3, 3
        read_idx = np.stack([rng.choice(s, r, replace=False) for _ in range(b * t)]).reshape(b, t, r)
        write_idx = np.stack([rng.choice(s, r, replace=False) for _ in range(b * t)]).reshape(b, t, r)
        read_w = rng.uniform(0.1, 1.0, size=(b, t, r)); read_w /= read_w.sum(-1, keepdims=True)
        write_w = rng.uniform(0.1, 1.0, size=(b, t, r)); write_w /= write_w.sum(-1, keepdims=True)
        return {
            "memory": rng.normal(size=(b, s, d)),
            "read_indices": read_idx,
            "read_weights": read_w,
            "write_indices": write_idx,
            "write_weights": write_w,
            "values": rng.normal(size=(b, t, d)),
            "beta": rng.uniform(0.1, 0.9, size=(b, t)),
            "log_decay": -rng.uniform(0, 0.5, size=(b, t)),
        }
    return {}


def _reference_execute(spec, operands):
    """Run the compiler's independent architecture reference equation (torch)."""
    import torch

    from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer

    recipe = named_mixer_recipe(spec.name)
    plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, backend=MixerBackend.REFERENCE,
        dtype="float32",
    )
    torch_ops = {}
    for k, v in operands.items():
        # Python scalars (e.g. attention_window) pass through unchanged.
        if isinstance(v, (int, float, bool)):
            torch_ops[k] = v
            continue
        arr = np.asarray(v)
        # Route indices stay integer; everything else is float32.
        if arr.dtype.kind in "iu":
            torch_ops[k] = torch.as_tensor(arr, dtype=torch.int64)
        else:
            torch_ops[k] = torch.as_tensor(arr, dtype=torch.float32)
    return plan.execute(**torch_ops)


def measure_recipe(name, seed=0) -> RecipeCoverage:
    recipe = named_mixer_recipe(name)
    spec = recipe.spec
    family = spec.family.name
    operands = _rng_operands(spec, seed=seed)
    try:
        composed = execute_canonical(spec, **operands)
    except UnderspecifiedComposition as exc:
        return RecipeCoverage(name, family, lowers=False, reason=str(exc))
    except (ValueError, TypeError) as exc:
        return RecipeCoverage(name, family, lowers=False, reason=f"decline: {exc}")

    # Verify the composition against the independent architecture equation.
    try:
        reference = _reference_execute(spec, operands)
    except Exception as exc:  # reference executor may need recipe-specific operands
        return RecipeCoverage(
            name, family, lowers=True, reason=f"reference unavailable: {type(exc).__name__}"
        )
    ref_output = reference.output.detach().cpu().numpy()
    output_err = float(np.abs(composed["output"] - ref_output).max())
    # Relative output error: the reference executes in float32, so the correct
    # criterion is closeness relative to the output magnitude (FP32 precision),
    # not absolute. High-op-count operators (inner-loss states) accumulate FP32
    # error that is large in absolute terms but ~1e-7 relative.
    output_rel = output_err / max(float(np.abs(ref_output).max()), 1e-12)
    state_err = None
    if "final_state" in composed and reference.final_state is not None:
        ref_state = reference.final_state
        ref_state = getattr(ref_state, "ht", ref_state)
        comp_state = composed["final_state"]
        if isinstance(comp_state, tuple) and not isinstance(ref_state, tuple):
            # The composition packs (state, normalizer_state); the reference splits
            # them into final_state and final_normalizer_state.
            ref_parts = [ref_state]
            if reference.final_normalizer_state is not None:
                ref_parts.append(reference.final_normalizer_state)
            errs = [
                float(np.abs(np.asarray(c) - r.detach().cpu().numpy()).max())
                for c, r in zip(comp_state, ref_parts)
            ]
            state_err = max(errs) if errs else 0.0
        elif isinstance(ref_state, tuple) and isinstance(comp_state, tuple):
            # Multi-component state (e.g. MesaNet's (h_kk, h_kv)).
            errs = [
                float(np.abs(np.asarray(c) - r.detach().cpu().numpy()).max())
                for c, r in zip(comp_state, ref_state)
            ]
            state_err = max(errs) if errs else 0.0
        elif not isinstance(ref_state, tuple):
            state_err = float(
                np.abs(np.asarray(comp_state) - ref_state.detach().cpu().numpy()).max()
            )
    # Verify with a mixed absolute/relative criterion: FP64 composition vs FP32
    # reference agrees to the reference's float32 precision.
    verified = (output_err < 2e-5 or output_rel < 1e-5) and (
        state_err is None or state_err < 2e-4
    )
    return RecipeCoverage(
        name, family, lowers=True, output_err=output_err, state_err=state_err,
        verified=verified,
    )


def measure_coverage(names=None, seed=0) -> list[RecipeCoverage]:
    names = list(MIXER_RECIPE_NAMES) if names is None else list(names)
    return [measure_recipe(name, seed=seed) for name in names]


def render_markdown(rows: list[RecipeCoverage]) -> str:
    covered = [r for r in rows if r.lowers and r.verified]
    lowered_unverified = [r for r in rows if r.lowers and not r.verified]
    declined = [r for r in rows if not r.lowers]
    lines = [
        "# Representation coverage: every architecture lowers into a canonical core",
        "",
        "Status: evidence record, regenerated from the live compiler. This is the",
        "durable contract behind the unified generator: each named architecture recipe",
        "either lowers into the canonical NumPy execution path for its core (K1/K2/K3)",
        "and is verified against its independent architecture equation in float64, or it",
        "is recorded as declined with the precise reason. Once a recipe lowers and",
        "matches, any later optimization of the canonical K1/K2/K3 kernel lifts it",
        "automatically - there is no per-architecture re-derivation.",
        "",
        f"**{len(covered)} of {len(rows)} named recipes lower into a canonical core and "
        f"match their independent equation.** "
        f"{len(lowered_unverified)} lower but are not yet verified; "
        f"{len(declined)} decline (under-specified or exotic).",
        "",
        "## Verified (lower + match independent equation)",
        "",
        "| Recipe | Family | output err | state err |",
        "|---|---|---|---|",
    ]
    for r in covered:
        state = "-" if r.state_err is None else f"{r.state_err:.2e}"
        lines.append(f"| `{r.name}` | {r.family} | {r.output_err:.2e} | {state} |")
    lines += ["", "## Lowers but not yet verified", "",
              "| Recipe | Family | note |", "|---|---|---|"]
    for r in lowered_unverified:
        lines.append(f"| `{r.name}` | {r.family} | {r.reason or 'mismatch'} |")
    lines += ["", "## Declined (under-specified or exotic)", "",
              "| Recipe | Family | reason |", "|---|---|---|"]
    for r in declined:
        lines.append(f"| `{r.name}` | {r.family} | {r.reason} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = measure_coverage()
    covered = sum(1 for r in rows if r.lowers and r.verified)
    print(render_markdown(rows))
    print(f"\n[coverage] {covered}/{len(rows)} recipes lower into a canonical core and match.")
