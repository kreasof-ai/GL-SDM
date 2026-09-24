"""Measure true native-generation coverage versus dispatch-only coverage.

For most recipes the compiled plan can invoke the pinned upstream kernel through
a library adapter, which answers "how much does URM's dispatch add?" rather than
"does URM compute this natively?". That distinction is the whole value
proposition: URM is a unified generator that should cover trivial and exotic
operations with its own kernels, not a wrapper that dispatches to
FLA/FlashAttention/etc.

This report compiles every named recipe with the native backend and records
whether URM emits a native kernel (it computes the operation itself) or declines
(it can only run through the reference or an upstream adapter). Native coverage
- not dispatch parity - is the honest measure of the unified generator's reach.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTER = PROJECT_ROOT / "benchmarks" / "architecture-coverage.json"

# Native anchor -> human label for the reusable generator template.
NATIVE_ANCHOR_LABELS = {
    "urm_native_k1_online_softmax_v1": "K1 online softmax",
    "urm_native_diagonal_recurrence_v1": "K2 diagonal recurrence",
    "urm_native_sparse_state_mixer_v0": "K3 sparse state",
}


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def measure_native_coverage() -> dict[str, object]:
    """Compile every named recipe natively; return the coverage breakdown."""
    from urm.compiler.pipeline import MixerBackend, compile_mixer
    from urm.frontend.recipes import MIXER_RECIPE_NAMES, named_mixer_recipe

    native: dict[str, list[str]] = {}
    declined: list[str] = []
    for name in sorted(MIXER_RECIPE_NAMES):
        try:
            spec = named_mixer_recipe(name).spec
        except Exception:  # noqa: BLE001 - recipe construction failure = not compilable
            declined.append(name)
            continue
        try:
            plan = compile_mixer(spec, backend=MixerBackend.NATIVE, dtype="float32")
        except Exception:  # noqa: BLE001 - any decline means no native kernel
            declined.append(name)
            continue
        native.setdefault(plan.anchor, []).append(name)
    return {"native": native, "declined": sorted(declined)}


def render_markdown(coverage: dict[str, object]) -> str:
    native: dict[str, list[str]] = coverage["native"]  # type: ignore[assignment]
    declined: list[str] = coverage["declined"]  # type: ignore[assignment]
    native_count = sum(len(v) for v in native.values())
    total = native_count + len(declined)
    lines = [
        "# Native generation coverage",
        "",
        "This is the honest measure of URM's unified-generator reach: the recipes",
        "URM computes with its **own** generated kernels, not by dispatching to an",
        "upstream library. Dispatching a pinned upstream kernel through a library",
        "adapter does not establish that URM computes these operations natively. A",
        "unified generator that merely dispatches would be a thin wrapper - native",
        "coverage is what distinguishes a generator from a wrapper.",
        "",
        f"**{native_count} of {total} named recipes compile to a native kernel "
        f"({native_count * 100 // total}%).** The remaining {len(declined)} decline to",
        "the reference or an upstream adapter.",
        "",
        "## Natively generated (URM computes these)",
        "",
        "Each row is one reusable native generator template and the recipes it covers",
        "- one semantic kernel covering trivial and exotic variants, which is the",
        "unified-generator value proposition.",
        "",
        "| Native generator | Recipes covered | Count |",
        "|---|---|---|",
    ]
    for anchor, recipes in sorted(native.items(), key=lambda kv: -len(kv[1])):
        label = NATIVE_ANCHOR_LABELS.get(anchor, anchor)
        lines.append(f"| {label} (`{anchor}`) | {', '.join(f'`{r}`' for r in recipes)} | {len(recipes)} |")
    lines += [
        "",
        "## Dispatch/reference only (no native kernel yet)",
        "",
        "These recipes compile only through the reference oracle or an upstream",
        "library adapter. They are the native-generation backlog, grouped here as a",
        "flat list; the production matrix orders the mandatory subset.",
        "",
        "`" + "`, `".join(declined) + "`.",
        "",
        "## Reading this table",
        "",
        "- A recipe is **native** only when the compiler emits a URM kernel that",
        "  computes the operation; parity for native kernels is established against",
        "  the reference oracle and the pinned upstream comparator.",
        "- A recipe is **dispatch/reference only** when the compiler cannot yet",
        "  generate a native kernel for its full semantics and must run the",
        "  reference or call upstream. Expanding native coverage means lowering more",
        "  of these to native generation - not adding dispatch adapters.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print(render_markdown(measure_native_coverage()))


if __name__ == "__main__":
    main()
