"""Gated delta-rule adapter: URM dispatch around the pinned FLA operation.

Comparison levels for this family (docs/baselines.md):

1. semantic oracle - explicit fp32 recurrence loop (correctness only);
2. framework baseline - transparent eager PyTorch recurrence;
3. optimized upstream - ``fla.ops.gated_delta_rule`` (pinned 0.5.2) called
   directly: ``chunk_gated_delta_rule`` for prefill and
   ``fused_recurrent_gated_delta_rule`` for token-by-token decode;
4. this adapter - the SAME pinned upstream call behind URM spec validation,
   capability selection, and dispatch. The difference from level 3 is URM
   integration only; there is no native URM recurrent kernel yet.

The frozen semantic contract is documented in docs/fla-gated-delta-rule.md.
The adapter deliberately does not wrap unrelated FLA operations: unrelated
gate fusions, sigmoid-beta variants, varlen packing, and context parallelism
raise instead of being silently forced through this typed boundary.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

import torch

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
MODE_PREFILL = "prefill"
MODE_DECODE = "decode"

# Frozen upstream pin for this comparator contract
# (docs/fla-gated-delta-rule.md). Recorded SEPARATELY from whatever version is
# actually installed: an incompatible installation must be rejected, never
# relabeled as the expected pin.
EXPECTED_FLA_VERSION = "0.5.2"


def _version_compatible(installed: str) -> bool:
    """Exact-match contract: 0.5.2 == 0.5.2; any other version is rejected."""

    return installed == EXPECTED_FLA_VERSION


@dataclass(frozen=True, slots=True)
class GatedDeltaRuleSpec:
    """Validated static description of one gated delta-rule call.

    Boundary layout: q/k are ``[B, T, H, K]``, v is ``[B, T, HV, V]``,
    ``g`` and ``beta`` are ``[B, T, HV]``, states are ``[B, HV, K, V]``
    float32. Grouped value attention (GVA) requires ``HV % H == 0``.
    """

    batch: int
    sequence: int
    heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    dtype: torch.dtype
    mode: str = MODE_PREFILL
    has_initial_state: bool = False
    output_final_state: bool = False
    scale_is_default: bool = True

    @classmethod
    def from_tensors(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        mode: str = MODE_PREFILL,
        scale: float | None = None,
    ) -> GatedDeltaRuleSpec:
        if mode not in (MODE_PREFILL, MODE_DECODE):
            raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError(
                "gated delta rule v1 requires rank-4 [B,T,H,K]/[B,T,HV,V] tensors"
            )
        # Boundary layout is [B, T, H, K] (FLA convention), NOT [B, H, T, K].
        b, t_q, h, k_dim = query.shape
        b_k, t_k, _h_k, _ = key.shape
        b_v, t_v, hv, v_dim = value.shape
        if (b_k, t_k, _h_k) != (b, t_q, h):
            raise ValueError("key must share [B,T,H,K] shape with query")
        if (b_v, t_v) != (b, t_q):
            raise ValueError("value must share batch and sequence with query")
        if hv < h or hv % h != 0:
            raise ValueError("value_heads must satisfy HV >= H and HV % H == 0 (GVA)")
        if k_dim < 1 or v_dim < 1:
            raise ValueError("key_dim and value_dim must be positive")
        if gate.shape != (b, t_q, hv):
            raise ValueError(f"gate must have shape [B,T,HV]={b, t_q, hv}")
        if beta.shape != (b, t_q, hv):
            raise ValueError(f"beta must have shape [B,T,HV]={b, t_q, hv}")
        if not (
            gate.is_floating_point()
            and beta.is_floating_point()
            and query.is_floating_point()
        ):
            raise ValueError("q/k/v/g/beta must be floating tensors")
        if query.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"gated delta-rule comparator supports {SUPPORTED_DTYPES}")
        if not (query.dtype == key.dtype == value.dtype):
            raise ValueError("q/k/v dtypes must match")
        devices = {query.device, key.device, value.device, gate.device, beta.device}
        if len(devices) != 1:
            raise ValueError("q/k/v/g/beta must share a device")
        if initial_state is not None:
            if initial_state.shape != (b, hv, k_dim, v_dim):
                raise ValueError(
                    f"initial_state must have shape [B,HV,K,V]={(b, hv, k_dim, v_dim)}"
                )
            if initial_state.dtype != torch.float32:
                raise ValueError("initial_state must be float32")
        return cls(
            batch=b,
            sequence=t_q,
            heads=h,
            value_heads=hv,
            key_dim=k_dim,
            value_dim=v_dim,
            dtype=query.dtype,
            mode=mode,
            has_initial_state=initial_state is not None,
            output_final_state=output_final_state,
            scale_is_default=scale is None,
        )


def fla_version() -> dict[str, object]:
    """Identity of the pinned FLA upstream, resolved dynamically.

    ``expected_version`` is the frozen contract pin; ``installed_version`` is
    what importlib.metadata actually resolves. They are recorded separately and
    ``version_compatible`` compares them: the adapter refuses to run against an
    incompatible installation instead of labeling it with the expected pin.
    """

    try:
        distribution = importlib.metadata.distribution("flash-linear-attention")
        installed_version = distribution.metadata["Version"]
        import fla.ops.gated_delta_rule as gdr_module

        return {
            "package": "flash-linear-attention",
            "expected_version": EXPECTED_FLA_VERSION,
            "installed_version": installed_version,
            "version_compatible": _version_compatible(installed_version),
            "version": installed_version,
            "helper_package": {
                name: importlib.metadata.version(name)
                for name in ("fla-core",)
                if _installed(name)
            },
            "prefill_entry_point": f"{gdr_module.chunk_gated_delta_rule.__module__}"
            ".chunk_gated_delta_rule",
            "decode_entry_point": f"{gdr_module.fused_recurrent_gated_delta_rule.__module__}"
            ".fused_recurrent_gated_delta_rule",
            "repository": "https://github.com/fla-org/flash-linear-attention",
            "pin": f"release {EXPECTED_FLA_VERSION} (GitHub tag "
            f"v{EXPECTED_FLA_VERSION}); installed version recorded separately",
            "license": "MIT",
            "usage": "URM calls the installed package externally; no FLA "
            "source is vendored into URM",
        }
    except Exception as error:  # noqa: BLE001 - identity probing is best-effort
        return {"status": "not_applicable", "reason": repr(error)}


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


class UrmGatedDeltaRuleAdapter:
    """URM dispatch boundary around one pinned FLA gated delta-rule call.

    ``mode="prefill"`` dispatches to ``chunk_gated_delta_rule`` (chunk-parallel
    prefill with full backward support); ``mode="decode"`` dispatches to
    ``fused_recurrent_gated_delta_rule`` (recurrent decode, forward-only
    upstream). Both paths run the identical kernel as a direct upstream call
    on identical preallocated inputs.
    """

    name = "urm_gated_delta_rule"

    def __init__(self, mode: str = MODE_PREFILL) -> None:
        if mode not in (MODE_PREFILL, MODE_DECODE):
            raise ValueError(f"unknown adapter mode: {mode}")
        self.mode = mode
        try:
            from fla.ops.gated_delta_rule import (
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )

            self._chunk_fn = chunk_gated_delta_rule
            self._fused_fn = fused_recurrent_gated_delta_rule
            self._import_error = None
        except Exception as error:  # noqa: BLE001 - deferred to support_status
            self._import_error = repr(error)

    def support_status(self, spec: GatedDeltaRuleSpec) -> str | None:
        if self._import_error is not None:
            return f"flash-linear-attention unavailable: {self._import_error}"
        identity = fla_version()
        if identity.get("status") == "not_applicable":
            return f"flash-linear-attention identity unresolved: {identity['reason']}"
        if not identity["version_compatible"]:
            return (
                "installed flash-linear-attention "
                f"{identity['installed_version']} does not satisfy the frozen "
                f"contract pin =={EXPECTED_FLA_VERSION}; reinstall the pinned "
                "release instead of running unverified semantics"
            )
        if spec.mode != self.mode:
            return f"adapter built for mode {self.mode!r}, got {spec.mode!r}"
        if self.mode == MODE_DECODE and spec.sequence != 1:
            return "decode mode requires sequence == 1 (token-by-token calls)"
        return None

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        scale: float | None = None,
        return_info: bool = False,
    ):
        """Run the pinned upstream operation after validating the contract.

        Returns ``(output, final_state)`` where ``final_state`` is None unless
        ``output_final_state=True``. With ``return_info=True`` also returns
        metadata recording backend selection evidence.
        """
        spec = GatedDeltaRuleSpec.from_tensors(
            query,
            key,
            value,
            gate,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            mode=self.mode,
            scale=scale,
        )
        reason = self.support_status(spec)
        if reason is not None:
            raise RuntimeError(f"{self.name} unsupported configuration: {reason}")

        resolved_scale = scale if scale is not None else spec.key_dim**-0.5
        if self.mode == MODE_PREFILL:
            output, final_state = self._chunk_fn(
                query,
                key,
                value,
                gate,
                beta,
                scale=resolved_scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                state_v_first=False,
            )
        else:
            output, final_state = self._fused_fn(
                query,
                key,
                value,
                g=gate,
                beta=beta,
                scale=resolved_scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                state_v_first=False,
            )

        if return_info:
            info: dict[str, object] = {
                "adapter_backend": self.name,
                "mode": self.mode,
                "spec": {
                    "batch": spec.batch,
                    "sequence": spec.sequence,
                    "heads": spec.heads,
                    "value_heads": spec.value_heads,
                    "key_dim": spec.key_dim,
                    "value_dim": spec.value_dim,
                    "dtype": str(spec.dtype),
                    "has_initial_state": spec.has_initial_state,
                    "output_final_state": spec.output_final_state,
                },
                "upstream": fla_version(),
                "backward_supported": self.mode == MODE_PREFILL,
            }
            return output, final_state, info
        return output, final_state
