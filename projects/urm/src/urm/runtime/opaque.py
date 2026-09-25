"""Compile-opacity for backend kernels: the generic custom-op wrapper.

The ATMA ``external_baselines/custom_ops.py`` pattern, generalized: a kernel's forward
is wrapped as a pair of ``torch.library`` custom ops (fused forward + fused backward)
so a ``torch.compile``'d model treats the op as opaque nodes in BOTH the forward and
the compiled backward graph — no trace into the kernel closure, no graph break.

**Ownership.** This module lives in ``runtime/`` because opacity is an *invocation*
mechanism (how the compiler sees the call), not a kernel property (what equation the
kernel computes) and not a compile decision (whether to opacify — that is the plan's).
Kernel files carry zero ``torch.library`` knowledge; a provider (or the binder) calls
:func:`opacify` with the kernel and a small :class:`OpaqueOpSpec` describing its
operand schema. The compiler's config decides whether a plan step is invoked through
the opaque wrapper or eagerly.

**Applicability.** The backward re-runs the forward under ``enable_grad`` (ctx cannot
cross the op boundary). That requires the kernel to be a deterministic function of its
tensor operands — true for every admitted native kernel today. A future kernel that
must *save* intermediates for the backward (rather than recompute) does not fit this
generic pattern and gets a bespoke wrapper next to its provider, never inside the
kernel file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class OpaqueOpSpec:
    """The operand schema an opaque wrapper needs — the only op-specific information.

    ``tensor_args``: the forward's tensor operands, in call order. ``optional_args``:
    the subset that may be None (their backward gradient is returned as None).
    ``scalar_args``: non-tensor knobs (floats/strings/bools), in call order — they
    appear in the custom-op schema but receive no gradient. ``n_outputs``: how many
    tensors the forward returns.
    """

    name: str                       # the ``urm::<name>`` custom-op base name
    tensor_args: tuple[str, ...]
    n_outputs: int
    optional_args: tuple[str, ...] = ()
    scalar_args: tuple[str, ...] = ()


_WRAPPED: dict[str, Callable] = {}


def _scalar_schema_type(name: str) -> str:
    if name == "scale":
        return "float?"
    if name.startswith(("is_", "has_", "read_before")):
        return "bool"
    return "str"


def _schema_bwd(arg_names, tensor_args, optional_set, scalar_args, flag_args) -> str:
    parts = []
    for k in arg_names:
        if k in flag_args:
            parts.append(f"bool {k}")
        elif k in tensor_args or k.startswith("grad_out"):
            parts.append(("Tensor?" if k in optional_set else "Tensor") + f" {k}")
        elif k in scalar_args:
            parts.append(f"{_scalar_schema_type(k)} {k}")
    outs = ", ".join("Tensor" for _ in range(len(tensor_args)))
    return f"({', '.join(parts)}) -> ({outs})"


def opacify(
    spec: OpaqueOpSpec,
    forward: Callable[..., tuple],
    *,
    output_meta: Callable[..., list[tuple[tuple[int, ...], Any]]],
) -> Callable[..., tuple]:
    """Wrap ``forward`` as an opaque custom-op pair; return the opaque entry point.

    ``forward(**operands)`` returns ``spec.n_outputs`` tensors.
    ``output_meta(**operands)`` returns ``(shape, dtype)`` per output for the fake
    (meta) impl. The returned callable has the same keyword signature as ``forward``
    and is safe under ``torch.compile``. Memoized by ``spec.name`` — custom-op
    registration is process-global.
    """
    import torch

    if spec.name in _WRAPPED:
        return _WRAPPED[spec.name]

    tensor_args = list(spec.tensor_args)
    scalar_args = list(spec.scalar_args)
    optional = list(spec.optional_args)
    optional_set = set(optional)
    flag_args = [f"has_{k}" for k in optional]
    n_outputs = spec.n_outputs

    def _schema(args: list[str], n_out: int) -> str:
        parts = []
        for k in args:
            if k in flag_args:
                parts.append(f"bool {k}")
            elif k in tensor_args:
                parts.append(("Tensor?" if k in optional_set else "Tensor") + f" {k}")
            elif k in scalar_args:
                parts.append(f"{_scalar_schema_type(k)} {k}")
        outs = ", ".join("Tensor" for _ in range(n_out))
        return f"({', '.join(parts)}) -> ({outs})"

    # --- the opaque forward op -----------------------------------------------------
    def fwd_impl(*args):
        kwargs = dict(zip(tensor_args + scalar_args, args))
        return forward(**kwargs)

    fwd_op = torch.library.custom_op(
        f"urm::{spec.name}_fwd", mutates_args=(),
        schema=_schema(tensor_args + scalar_args, n_outputs),
    )(fwd_impl)

    def fwd_fake(*args):
        kwargs = dict(zip(tensor_args + scalar_args, args))
        metas = output_meta(**kwargs)
        probe = next(kwargs[k] for k in tensor_args if kwargs[k] is not None)
        return tuple(
            torch.empty(shape, device=probe.device, dtype=dtype) for shape, dtype in metas
        )

    fwd_op.register_fake(fwd_fake)

    # --- the opaque backward op ----------------------------------------------------
    def bwd_impl(*args):
        grad_outs = list(args[:n_outputs])
        vals = list(args[n_outputs:])
        tensors = dict(zip(tensor_args, vals[: len(tensor_args)]))
        scalars = dict(zip(scalar_args, vals[len(tensor_args): len(tensor_args) + len(scalar_args)]))
        flags = dict(zip(flag_args, vals[len(tensor_args) + len(scalar_args):]))
        with torch.enable_grad():
            leaves = {}
            for k in tensor_args:
                if k in optional_set and not flags[f"has_{k}"]:
                    leaves[k] = None
                else:
                    leaves[k] = tensors[k].detach().requires_grad_(True)
            outs = forward(**{**leaves, **scalars})
            targets = [leaves[k] for k in tensor_args if leaves[k] is not None]
            grads = torch.autograd.grad(list(outs), targets, list(grad_outs),
                                        allow_unused=True)
        it = iter(grads)
        result = []
        for k in tensor_args:
            if k in optional_set and not flags[f"has_{k}"]:
                result.append(torch.zeros_like(tensors[k]))
            else:
                result.append(next(it).contiguous())
        return result

    bwd_arg_names = ([f"grad_out{i}" for i in range(n_outputs)] + tensor_args
                     + scalar_args + flag_args)
    bwd_op = torch.library.custom_op(
        f"urm::{spec.name}_bwd", mutates_args=(),
        schema=_schema_bwd(bwd_arg_names, tensor_args, optional_set, scalar_args, flag_args),
    )(bwd_impl)

    def bwd_fake(*args):
        vals = list(args[n_outputs:])
        tensors = dict(zip(tensor_args, vals[: len(tensor_args)]))
        return [torch.empty_like(tensors[k]) for k in tensor_args]

    bwd_op.register_fake(bwd_fake)

    # --- the autograd bridge -------------------------------------------------------
    def setup(ctx, inputs, output):
        kwargs = dict(zip(tensor_args + scalar_args, inputs))
        probe = next(kwargs[k] for k in tensor_args if kwargs[k] is not None)
        ctx.save_for_backward(*[
            kwargs[k] if kwargs[k] is not None else torch.empty(0, device=probe.device)
            for k in tensor_args
        ])
        ctx.scalars = {k: kwargs[k] for k in scalar_args}
        ctx.flags = {k: kwargs[k] is not None for k in optional}

    def bwd_bridge(ctx, *grad_outs):
        tensors = ctx.saved_tensors
        flags = ctx.flags
        op = getattr(torch.ops.urm, f"{spec.name}_bwd")
        grads = op(*grad_outs, *tensors,
                   *[ctx.scalars[k] for k in scalar_args],
                   *[flags[k] for k in optional])
        it = iter(grads)
        out = []
        for k in tensor_args:
            g = next(it)  # the bwd op returns one gradient per tensor arg (zeros for absent)
            if k in optional_set and not flags[k]:
                out.append(None)
            else:
                out.append(g)
        out.extend([None] * len(scalar_args))
        return tuple(out)

    fwd_op.register_autograd(bwd_bridge, setup_context=setup)

    def opaque_entry(**kwargs):
        return fwd_op(*[kwargs[k] for k in tensor_args + scalar_args])

    _WRAPPED[spec.name] = opaque_entry
    return opaque_entry


__all__ = ["OpaqueOpSpec", "opacify"]
