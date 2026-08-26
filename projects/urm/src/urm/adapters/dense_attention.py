"""Dense causal attention adapter: URM dispatch around a pinned upstream kernel.

Comparison levels (see docs/baselines.md):

1. semantic oracle - explicit fp32 softmax-reduce, correctness only;
2. framework baseline - ``torch.nn.functional.scaled_dot_product_attention``
   forced to the math backend;
3. optimized upstream - FlashAttention ``flash_attn_func`` (pinned package) or
   SDPA forced to the flash backend;
4. this adapter - the SAME pinned upstream call behind URM spec validation and
   dispatch. The adapter exists to measure dispatch overhead; it does not
   claim routed reduction competes with attention kernels.

Semantics frozen for the comparison: causal alignment, BHSD layout at the
boundary (transposed views only), optional GQA with ``heads % kv_heads == 0``,
scale ``1/sqrt(head_dim)`` unless overridden, dropout disabled, fp16/bf16.
"""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DenseAttentionSpec:
    """Validated static description of a dense causal attention call."""

    batch: int
    heads: int
    kv_heads: int
    queries: int
    keys: int
    head_dim: int
    dtype: torch.dtype
    causal: bool = True
    scale_is_default: bool = True

    @classmethod
    def from_tensors(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
    ) -> DenseAttentionSpec:
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("dense attention v1 requires rank-4 BHSD tensors")
        b, h, s_q, d = query.shape
        b_k, h_kv, s_k, d_k = key.shape
        if b_k != b or d_k != d:
            raise ValueError("key must share batch and head_dim with query")
        if value.shape != key.shape:
            raise ValueError("value must have the same shape as key")
        if h % h_kv != 0:
            raise ValueError("heads must be divisible by kv_heads (GQA grouping)")
        if d < 1:
            raise ValueError("head_dim must be positive")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dense attention comparator supports fp16/bf16")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("q/k/v dtypes must match")
        if not (query.device == key.device == value.device):
            raise ValueError("q/k/v must share a device")
        return cls(
            batch=b,
            heads=h,
            kv_heads=h_kv,
            queries=s_q,
            keys=s_k,
            head_dim=d,
            dtype=query.dtype,
            causal=causal,
            scale_is_default=scale is None,
        )


def flash_attn_version() -> dict[str, object]:
    """Identity of the pinned FlashAttention upstream, when installed."""
    try:
        version = importlib.metadata.version("flash-attn")  # noqa: F841
        import flash_attn

        return {
            "package": "flash-attn",
            "version": flash_attn.__version__,
            "wheel_tag": "cu12torch2.8cxx11abiTRUE-cp312",
            "repository": "https://github.com/Dao-AILab/flash-attention",
            "pin": "v2.8.3 release wheel",
        }
    except Exception as error:  # noqa: BLE001 - identity probing is best-effort
        return {"status": "not_applicable", "reason": repr(error)}


class UrmDenseCausalAttentionAdapter:
    """URM dispatch boundary around one pinned dense-attention implementation.

    ``backend="flash_attn"`` calls the upstream package directly;
    ``backend="sdpa_flash"`` calls PyTorch SDPA restricted to its flash
    backend. Both paths execute the same mathematical contract as the direct
    calls in benchmark level 3, so any latency difference is adapter overhead.
    """

    name = "urm_dense_causal_attention"

    def __init__(self, backend: str = "flash_attn") -> None:
        if backend not in ("flash_attn", "sdpa_flash"):
            raise ValueError(f"unknown adapter backend: {backend}")
        self.backend = backend
        # Resolve the pinned upstream entry point once; steady-state calls
        # must not pay import or attribute-lookup cost.
        self._flash_func = None
        self._sdpa_ctx = None
        if backend == "flash_attn":
            try:
                from flash_attn import flash_attn_func

                self._flash_func = flash_attn_func
            except Exception as error:  # noqa: BLE001 - deferred to support_status
                self._flash_import_error = repr(error)
        else:
            try:
                backends = torch.backends.cuda
                self._saved_sdpa_flags = (
                    backends.flash_sdp_enabled(),
                    backends.math_sdp_enabled(),
                    backends.mem_efficient_sdp_enabled(),
                    getattr(backends, "cudnn_sdp_enabled", lambda: False)(),
                )
            except Exception as error:  # noqa: BLE001
                self._sdpa_import_error = repr(error)

    def support_status(self, spec: DenseAttentionSpec) -> str | None:
        if self.backend == "flash_attn":
            if getattr(self, "_flash_import_error", None) is not None:
                return f"flash-attn unavailable: {self._flash_import_error}"
            if self._flash_func is None:
                try:
                    from flash_attn import flash_attn_func

                    self._flash_func = flash_attn_func
                except Exception as error:  # noqa: BLE001
                    return f"flash-attn unavailable: {error}"
            return None
        else:
            if not torch.cuda.is_available():
                return "SDPA flash backend requires CUDA"
            major, minor = (
                torch.cuda.get_device_capability()
                if torch.cuda.is_available()
                else (0, 0)
            )
            if (major, minor) < (8, 0):
                return "SDPA flash backend requires SM80+"
        if spec.head_dim > 256:
            return "head_dim > 256 unsupported"
        return None

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
        return_info: bool = False,
    ):
        """Return the attention output in BHSD layout.

        With ``return_info=True`` also returns a metadata dictionary so that
        benchmarks can record backend-selection evidence without paying the
        cost inside steady-state timing loops.
        """
        spec = DenseAttentionSpec.from_tensors(
            query, key, value, causal=causal, scale=scale
        )
        reason = self.support_status(spec)
        if reason is not None:
            raise RuntimeError(f"{self.name} unsupported configuration: {reason}")

        resolved_scale = scale if scale is not None else 1.0 / math.sqrt(spec.head_dim)

        if self.backend == "flash_attn":
            # flash_attn_func consumes (B, S, H, D) with last dim contiguous;
            # transposing BHSD-contiguous tensors yields exactly that view.
            output = self._flash_func(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                causal=causal,
                softmax_scale=resolved_scale,
            ).transpose(1, 2)
        else:
            if getattr(self, "_sdpa_import_error", None) is not None:
                raise RuntimeError("SDPA flash backend unavailable")
            backends = torch.backends.cuda
            saved = self._saved_sdpa_flags
            backends.enable_flash_sdp(True)
            backends.enable_math_sdp(False)
            backends.enable_mem_efficient_sdp(False)
            disable_cudnn = hasattr(backends, "enable_cudnn_sdp")
            if disable_cudnn:
                backends.enable_cudnn_sdp(False)
            try:
                output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=causal,
                    scale=resolved_scale,
                    enable_gqa=spec.kv_heads != spec.heads,
                )
            finally:
                backends.enable_flash_sdp(saved[0])
                backends.enable_math_sdp(saved[1])
                backends.enable_mem_efficient_sdp(saved[2])
                if disable_cudnn:
                    backends.enable_cudnn_sdp(saved[3])

        if return_info:
            info: dict[str, object] = {
                "adapter_backend": self.backend,
                "spec": {
                    "batch": spec.batch,
                    "heads": spec.heads,
                    "kv_heads": spec.kv_heads,
                    "queries": spec.queries,
                    "keys": spec.keys,
                    "head_dim": spec.head_dim,
                    "dtype": str(spec.dtype),
                    "causal": spec.causal,
                },
                "upstream": (
                    flash_attn_version()
                    if self.backend == "flash_attn"
                    else {"package": "torch-sdpa-flash"}
                ),
            }
            return output, info
        return output
