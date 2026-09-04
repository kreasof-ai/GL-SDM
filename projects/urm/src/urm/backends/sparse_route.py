"""Capability-checked native score-to-route backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from urm.compiler.semantic import DType, SparseRouteSelectionSpec

NATIVE_SPARSE_ROUTE_NAME = "urm_native_sparse_route_selection_v0"
_SCORE_CERTIFICATE = object()
_ROUTE_OUTPUT_CERTIFICATE = object()


@dataclass(frozen=True, slots=True)
class SparseRouteSupportStatus:
    supported: bool
    code: str
    reason: str | None = None

    @classmethod
    def yes(cls):
        return cls(True, "supported")

    @classmethod
    def no(cls, code: str, reason: str):
        return cls(False, code, reason)

    def require(self) -> None:
        if not self.supported:
            raise ValueError(
                f"{NATIVE_SPARSE_ROUTE_NAME} declined [{self.code}]: {self.reason}"
            )


@dataclass(frozen=True, slots=True)
class CertifiedSparseRouteScores:
    """Tie-free finite score input certified outside steady-state dispatch."""

    spec: SparseRouteSelectionSpec
    scores: object
    _version: int = field(default=-1, repr=False, compare=False)
    _certificate: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _SCORE_CERTIFICATE:
            raise ValueError("route scores must be created by certify()")

    @classmethod
    def certify(cls, spec: SparseRouteSelectionSpec, scores: object):
        import torch

        expected = (spec.parallel, spec.sequence, spec.score_width)
        if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected:
            raise ValueError(f"scores must have shape {expected}")
        expected_dtype = {
            DType.FLOAT32: torch.float32,
            DType.BFLOAT16: torch.bfloat16,
        }[spec.dtype]
        if scores.dtype != expected_dtype:
            raise ValueError("score dtype must match route semantics")
        if not scores.is_cuda or not scores.is_contiguous():
            raise ValueError("scores must be contiguous CUDA storage")
        if not bool(torch.isfinite(scores).all().item()):
            raise ValueError("scores must be finite")
        half = spec.factor_extent
        k_sub = min(spec.route_width, half)
        # Chunk certification to bound temporary pair-score storage.
        rows = scores.reshape(-1, spec.score_width)
        for start in range(0, rows.shape[0], 256):
            chunk = rows[start : start + 256]
            chunk_left, chunk_right = chunk.chunk(2, dim=-1)
            left_sorted = chunk_left.sort(dim=-1, descending=True).values
            right_sorted = chunk_right.sort(dim=-1, descending=True).values
            if k_sub < half and (
                bool((left_sorted[..., k_sub - 1] == left_sorted[..., k_sub]).any())
                or bool(
                    (right_sorted[..., k_sub - 1] == right_sorted[..., k_sub]).any()
                )
            ):
                raise ValueError("route scores contain a factor-selection boundary tie")
            left_top = left_sorted[..., :k_sub]
            right_top = right_sorted[..., :k_sub]
            pairs = (left_top.unsqueeze(-1) + right_top.unsqueeze(-2)).flatten(-2)
            if spec.route_width < pairs.shape[-1]:
                ordered = pairs.sort(dim=-1, descending=True).values
                if bool(
                    (
                        ordered[..., spec.route_width - 1]
                        == ordered[..., spec.route_width]
                    ).any()
                ):
                    raise ValueError(
                        "route scores contain a pair-selection boundary tie"
                    )
        return cls(spec, scores, scores._version, _SCORE_CERTIFICATE)

    def require_intact(self) -> None:
        if self.scores._version != self._version:
            raise ValueError("certified route scores were mutated")


@dataclass(frozen=True, slots=True)
class NativeSparseRouteOutput:
    """Trusted result produced only by the URM route kernel."""

    spec: SparseRouteSelectionSpec
    addresses: object
    weights: object
    _versions: tuple[int, int] = field(default=(-1, -1), repr=False, compare=False)
    _certificate: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _ROUTE_OUTPUT_CERTIFICATE:
            raise ValueError("native routes must be produced by generate_certified()")

    def require_intact(self) -> None:
        if (self.addresses._version, self.weights._version) != self._versions:
            raise ValueError("native generated routes were mutated")


class TritonSparseRouteBackend:
    """URM-native lowering for the frozen factorized additive top-k route."""

    name = NATIVE_SPARSE_ROUTE_NAME

    def __init__(self, spec: SparseRouteSelectionSpec) -> None:
        self.spec = spec
        self.support_status(spec).require()

    @staticmethod
    def support_status(spec: SparseRouteSelectionSpec) -> SparseRouteSupportStatus:
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("triton") is None
        ):
            return SparseRouteSupportStatus.no(
                "missing_dependency", "PyTorch and Triton are required"
            )
        if spec.factor_extent > 256 or spec.route_width > 64:
            return SparseRouteSupportStatus.no(
                "unsupported_shape", "v0 requires factor extent <=256 and width <=64"
            )
        import torch

        if not torch.cuda.is_available():
            return SparseRouteSupportStatus.no(
                "unsupported_hardware", "CUDA is unavailable"
            )
        return SparseRouteSupportStatus.yes()

    def generate(self, certified: CertifiedSparseRouteScores):
        import torch

        if certified.spec != self.spec:
            raise ValueError("certified scores do not match route semantics")
        certified.require_intact()
        device = certified.scores.device
        if torch.cuda.get_device_capability(device) < (8, 0):
            raise ValueError(
                f"{NATIVE_SPARSE_ROUTE_NAME} declined [unsupported_hardware]: "
                "v0 requires SM80 or newer"
            )
        from urm.triton_kernels.sparse_route import sparse_route_selection

        index_dtype = {
            DType.INT32: torch.int32,
            DType.INT64: torch.int64,
        }[self.spec.output_index_dtype]
        return sparse_route_selection(
            certified.scores,
            self.spec.source_extent,
            self.spec.route_width,
            index_dtype=index_dtype,
        )

    def generate_certified(
        self, certified: CertifiedSparseRouteScores
    ) -> NativeSparseRouteOutput:
        addresses, weights = self.generate(certified)
        return NativeSparseRouteOutput(
            self.spec,
            addresses,
            weights,
            (addresses._version, weights._version),
            _ROUTE_OUTPUT_CERTIFICATE,
        )

    def launch_schedule(self) -> dict[str, int | str]:
        import triton

        half = self.spec.factor_extent
        route = max(2, triton.next_power_of_2(self.spec.route_width))
        return {
            "schedule_family": "row_owned_factor_topk_canonical_softmax",
            "block_half": triton.next_power_of_2(half),
            "block_route": route,
            "block_pair": route * route,
            "num_warps": 8 if route * route >= 1024 else 4,
            "num_stages": 2,
        }


__all__ = [
    "NATIVE_SPARSE_ROUTE_NAME",
    "CertifiedSparseRouteScores",
    "NativeSparseRouteOutput",
    "SparseRouteSupportStatus",
    "TritonSparseRouteBackend",
]
