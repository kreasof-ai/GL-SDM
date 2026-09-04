"""Fully native score-to-state Sparse Memory pipeline."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from urm.backends.sparse_route import (
    CertifiedSparseRouteScores,
    TritonSparseRouteBackend,
)
from urm.backends.sparse_state_mixer import (
    CertifiedSparseStateRoutes,
    SparseState,
    TritonSparseStateMixerBackend,
)
from urm.compiler.semantic import (
    DType,
    MergePolicy,
    ScoreNormalization,
    SDMExecutionMode,
    SparseAddressingKind,
    SparseMemoryMixerSpec,
    SparseRouteSelectionSpec,
    SparseStateExecutionMode,
    SparseStateMixerSpec,
    SparseStateOperation,
    SparseStatePolicy,
    SparseUpdateRule,
)
from urm.sparse_state_mixer import SparseStateSupportStatus, sparse_state_spec_status

NATIVE_SPARSE_MEMORY_NAME = "urm_native_sparse_memory_e2e_v0"
_PIPELINE_CERTIFICATE = object()


@dataclass(frozen=True, slots=True)
class CertifiedSparseMemoryInputs:
    spec: SparseMemoryMixerSpec
    read_scores: CertifiedSparseRouteScores
    write_scores: CertifiedSparseRouteScores | None
    values: object | None
    beta: object | None
    log_decay: object | None
    _versions: tuple[int, ...] = field(default=(), repr=False, compare=False)
    _certificate: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _PIPELINE_CERTIFICATE:
            raise ValueError("sparse memory inputs must be created by prepare()")

    def require_intact(self) -> None:
        self.read_scores.require_intact()
        if self.write_scores is not None:
            self.write_scores.require_intact()
        tensors = tuple(
            x for x in (self.values, self.beta, self.log_decay) if x is not None
        )
        if tuple(x._version for x in tensors) != self._versions:
            raise ValueError("certified sparse memory operands were mutated")


@dataclass(frozen=True, slots=True)
class SparseMemoryResult:
    readings: object
    state: SparseState
    read_addresses: object
    read_weights: object
    write_addresses: object | None = None
    write_weights: object | None = None


class TritonSparseMemoryBackend:
    """Compose two native route lowerings with the native state lowering."""

    name = NATIVE_SPARSE_MEMORY_NAME

    def __init__(self, spec: SparseMemoryMixerSpec) -> None:
        self.spec = spec
        self.support_status(spec).require()
        self.read_spec = SparseRouteSelectionSpec(
            spec.parallel,
            spec.sequence,
            spec.slots_per_partition,
            spec.reads,
            spec.dtype,
        )
        self.read_backend = TritonSparseRouteBackend(self.read_spec)
        self.write_spec = None
        self.write_backend = None
        if spec.operation is SparseStateOperation.UPDATE:
            self.write_spec = SparseRouteSelectionSpec(
                spec.parallel,
                spec.sequence,
                spec.slots_per_partition,
                spec.writes,
                spec.dtype,
            )
            self.write_backend = TritonSparseRouteBackend(self.write_spec)
        mode = (
            SparseStateExecutionMode.TRAINING
            if spec.mode is SDMExecutionMode.TRAINING
            else SparseStateExecutionMode.INFERENCE
        )
        self.state_spec = SparseStateMixerSpec(
            spec.parallel,
            spec.sequence,
            spec.slots_per_partition,
            spec.value_dim,
            spec.writes,
            spec.reads,
            spec.dtype,
            spec.operation,
            spec.read_timing,
            mode=mode,
            update_rule=spec.update_rule,
            collision_policy=spec.collision_policy,
            within_token_collision_policy=spec.within_token_collision_policy,
            state_policy=spec.state_policy,
            page_size=spec.page_size,
        )
        self.state_backend = TritonSparseStateMixerBackend(self.state_spec)

    @staticmethod
    def support_status(spec: SparseMemoryMixerSpec) -> SparseStateSupportStatus:
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("triton") is None
        ):
            return SparseStateSupportStatus.no(
                "missing_dependency", "PyTorch and Triton are required"
            )
        import torch

        exact = (
            spec.addressing is SparseAddressingKind.PRODUCT_KEY_TOP_K
            and spec.normalization is ScoreNormalization.SOFTMAX
            and spec.update_rule is SparseUpdateRule.DECAYED_DELTA
            and spec.collision_policy is MergePolicy.ORDERED
            and spec.within_token_collision_policy is MergePolicy.REJECT
            and spec.state_policy is SparseStatePolicy.PERSISTENT_IN_PLACE
            and spec.page_size == 1
        )
        if not exact:
            return SparseStateSupportStatus.no(
                "unsupported_semantics", "operation is outside native E2E v0"
            )
        if spec.reads > 64 or spec.writes > 64:
            return SparseStateSupportStatus.no(
                "unsupported_shape", "native route v0 supports widths <=64"
            )
        half = round(spec.slots_per_partition**0.5)
        if half > 256:
            return SparseStateSupportStatus.no(
                "unsupported_shape", "native route v0 supports factor extent <=256"
            )
        state_spec = SparseStateMixerSpec(
            spec.parallel,
            spec.sequence,
            spec.slots_per_partition,
            spec.value_dim,
            spec.writes,
            spec.reads,
            spec.dtype,
            spec.operation,
            spec.read_timing,
            mode=(
                SparseStateExecutionMode.TRAINING
                if spec.mode is SDMExecutionMode.TRAINING
                else SparseStateExecutionMode.INFERENCE
            ),
            update_rule=spec.update_rule,
            collision_policy=spec.collision_policy,
            within_token_collision_policy=spec.within_token_collision_policy,
            state_policy=spec.state_policy,
            page_size=spec.page_size,
        )
        if not torch.cuda.is_available():
            return SparseStateSupportStatus.no(
                "unsupported_hardware", "CUDA is unavailable"
            )
        capability = torch.cuda.get_device_capability()
        return sparse_state_spec_status(
            state_spec, device_type="cuda", compute_capability=capability
        )

    def prepare(
        self,
        read_scores: object,
        *,
        write_scores: object | None = None,
        values: object | None = None,
        beta: object | None = None,
        log_decay: object | None = None,
    ) -> CertifiedSparseMemoryInputs:
        import torch

        read = CertifiedSparseRouteScores.certify(self.read_spec, read_scores)
        write = None
        supplied = (values, beta, log_decay)
        if self.spec.operation is SparseStateOperation.UPDATE:
            if write_scores is None or self.write_spec is None:
                raise ValueError("updating sparse memory requires write scores")
            write = CertifiedSparseRouteScores.certify(self.write_spec, write_scores)
            expected_value = (
                self.spec.parallel,
                self.spec.sequence,
                self.spec.value_dim,
            )
            expected_scalar = (self.spec.parallel, self.spec.sequence, 1)
            if values is None or tuple(values.shape) != expected_value:
                raise ValueError(f"values must have shape {expected_value}")
            if beta is None or tuple(beta.shape) != expected_scalar:
                raise ValueError(f"beta must have shape {expected_scalar}")
            if log_decay is None or tuple(log_decay.shape) != expected_scalar:
                raise ValueError(f"log_decay must have shape {expected_scalar}")
            expected_dtype = {
                DType.FLOAT32: torch.float32,
                DType.BFLOAT16: torch.bfloat16,
            }[self.spec.dtype]
            for tensor in supplied:
                if (
                    tensor.device != read_scores.device
                    or tensor.dtype != expected_dtype
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "update operands must be contiguous and match score device/dtype"
                    )
                if not bool(torch.isfinite(tensor).all().item()):
                    raise ValueError("update operands must be finite")
        elif write_scores is not None or any(x is not None for x in supplied):
            raise ValueError("read-only sparse memory accepts only read scores")
        tensors = tuple(x for x in supplied if x is not None)
        return CertifiedSparseMemoryInputs(
            self.spec,
            read,
            write,
            values,
            beta,
            log_decay,
            tuple(x._version for x in tensors),
            _PIPELINE_CERTIFICATE,
        )

    def execute(
        self, state: SparseState, prepared: CertifiedSparseMemoryInputs
    ) -> SparseMemoryResult:
        if prepared.spec != self.spec:
            raise ValueError("prepared inputs do not match sparse memory semantics")
        prepared.require_intact()
        read = self.read_backend.generate_certified(prepared.read_scores)
        write = (
            self.write_backend.generate_certified(prepared.write_scores)
            if self.write_backend is not None and prepared.write_scores is not None
            else None
        )
        routes = CertifiedSparseStateRoutes.from_native_generation(
            self.state_spec, read, write_output=write
        )
        state_inputs = self.state_backend._prepare_generated_routes(
            routes,
            values=prepared.values,
            beta=prepared.beta,
            log_decay=prepared.log_decay,
        )
        readings, state = self.state_backend.execute(state, state_inputs)
        return SparseMemoryResult(
            readings,
            state,
            read.addresses,
            read.weights,
            write.addresses if write is not None else None,
            write.weights if write is not None else None,
        )


__all__ = [
    "NATIVE_SPARSE_MEMORY_NAME",
    "CertifiedSparseMemoryInputs",
    "SparseMemoryResult",
    "TritonSparseMemoryBackend",
]
