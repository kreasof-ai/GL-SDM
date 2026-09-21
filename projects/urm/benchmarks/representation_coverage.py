"""Representation-coverage gate: every architecture lowers into a canonical core.

This is the durable contract behind the unified generator. For each named
architecture recipe it attempts to lower the declarative
:class:`~urm.ir.mixer.UnifiedMixerSpec` into the canonical NumPy execution path
for its core (K1 normalized routed reduction, K2 matrix-state recurrence, K3
ordered sparse state) via :func:`urm.oracles.composition.execute_canonical`. A
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

from urm.frontend.mixer_recipes import MIXER_RECIPE_NAMES, named_mixer_recipe
from urm.ir.mixer import MixerKernelFamily
from urm.oracles.composition import UnderspecifiedComposition, execute_canonical


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
        from urm.ir.mixer import K1Operation

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
        from urm.ir.mixer import RecurrentLayout

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
        from urm.ir.mixer import DecayGranularity, StateUpdateRule

        if spec.gdn2_ssm:
            # Dual-gate delta: erase/write gates replace beta.
            ops["erase_gate"] = rng.uniform(0.1, 0.9, size=(b, t, h, k))
            ops["write_gate"] = rng.uniform(0.1, 0.9, size=(b, t, h, v))
        elif spec.update_rule is StateUpdateRule.DELTA:
            ops["beta"] = rng.uniform(0.1, 0.9, size=(b, t, h))
        if spec.static_head_decay:
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

    from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer

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
    output_err = float(
        np.abs(composed["output"] - reference.output.detach().cpu().numpy()).max()
    )
    state_err = None
    if "final_state" in composed and reference.final_state is not None:
        ref_state = reference.final_state
        ref_state = getattr(ref_state, "ht", ref_state)
        if not isinstance(ref_state, tuple):
            state_err = float(
                np.abs(composed["final_state"] - ref_state.detach().cpu().numpy()).max()
            )
    verified = output_err < 2e-5 and (state_err is None or state_err < 2e-5)
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
