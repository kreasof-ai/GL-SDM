"""Semantic layer: routed sequence models as typed programs over logical domains.

The semantic IR describes WHAT a routed mixer computes - logical domains,
scores and selection, route edges and weights, gather/exchange, transforms,
reductions, ordered recurrence, state reads/updates with merge policies, and
commit/version boundaries. It deliberately says nothing about devices, tiles,
threads, or schedules: routing operates over logical domains, never physical
tensor indices.

Programs are frozen dataclasses; construction goes through
:meth:`SemanticProgram.build`, which validates name binding and rejects
programs that would need an untyped escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from urm.compiler.diagnostics import DiagnosticCode, DiagnosticsCollector
from urm.compiler.effects import (
    ATOMIC_ACCUMULATE,
    COLLECTIVE,
    COMMIT,
    ORDERED_STATE,
    PURE,
    REDUCING,
    STATE_READ_EFFECT,
    EffectSignature,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class LogicalDomain(StrEnum):
    """Logical iteration/name spaces routes live in."""

    SEQUENCE = "sequence"
    HEAD = "head"
    EXPERT = "expert"
    PARAMETER_BLOCK = "parameter_block"
    RECURRENT_STATE = "recurrent_state"
    MEMORY_PAGE = "memory_page"


class SelectionKind(StrEnum):
    DENSE = "dense"
    BLOCK_SPARSE = "block_sparse"
    TOP_K = "top_k"
    THRESHOLD = "threshold"


class ScoreNormalization(StrEnum):
    NONE = "none"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    L1 = "l1"


class MergePolicy(StrEnum):
    """Collision policy for multi-writer state updates."""

    NOT_APPLICABLE = "not_applicable"
    ORDERED = "ordered"  # requires an ordered-scan lowering
    SUM = "sum"
    MEAN = "mean"
    LAST_WRITE = "last_write"
    REJECT = "reject"


class CapacityPolicy(StrEnum):
    DROPLESS = "dropless"
    FIXED_CAPACITY = "fixed_capacity"
    EXPERT_QUOTA = "expert_quota"


class TransformKind(StrEnum):
    """Closed transform vocabulary; linear members declare linearity."""

    IDENTITY = "identity"
    ROW_SCALE = "row_scale"  # one scalar per leading (query) row
    AFFINE = "affine"  # per-channel affine; linear when the weight is fixed
    RELU = "relu"
    GELU = "gelu"
    SIGMOID = "sigmoid"

    @property
    def is_linear(self) -> bool:
        return self in {TransformKind.IDENTITY, TransformKind.ROW_SCALE}


class CollectiveKind(StrEnum):
    """Collective semantic intent (placement decides the concrete algorithm)."""

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"


class DType(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"


Dim = int | str  # symbolic or concrete extent


@dataclass(frozen=True, slots=True)
class TensorHandle:
    """A named, typed, symbolically shaped logical tensor."""

    name: str
    dtype: DType
    shape: tuple[Dim, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Semantic choices of one route edge set."""

    query_domain: LogicalDomain
    source_domain: LogicalDomain
    selection: SelectionKind
    normalization: ScoreNormalization
    top_k: int | None = None
    threshold: float | None = None
    page_size: int | None = None
    capacity_policy: CapacityPolicy = CapacityPolicy.DROPLESS
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.selection is SelectionKind.TOP_K and not (
            self.top_k and self.top_k > 0
        ):
            raise ValueError("top_k selection requires top_k > 0")
        if self.selection is SelectionKind.THRESHOLD and self.threshold is None:
            raise ValueError("threshold selection requires a threshold")
        if (
            self.selection in {SelectionKind.DENSE, SelectionKind.BLOCK_SPARSE}
            and self.top_k is not None
        ):
            raise ValueError("dense/block-sparse routes must not declare top_k")
        if self.source_domain is LogicalDomain.MEMORY_PAGE and not self.page_size:
            raise ValueError("memory-page sources require page_size")

    @property
    def width_is_dynamic(self) -> bool:
        return self.selection is SelectionKind.THRESHOLD


class SDMExecutionMode(StrEnum):
    """Autograd intent of the restricted sparse-memory mixer skeleton."""

    INFERENCE = "inference"
    TRAINING = "training"


class SparseAddressingKind(StrEnum):
    PRODUCT_KEY_TOP_K = "product_key_top_k"


class SparseScoreComposition(StrEnum):
    """Closed score-composition vocabulary for sparse route production."""

    PAIRWISE_ADDITIVE_FACTORS = "pairwise_additive_factors"


class SparseAddressCanonicalization(StrEnum):
    ASCENDING = "ascending"


class SparseRouteTiePolicy(StrEnum):
    """Deterministic selected-set policy for equal route scores."""

    HIGHEST_ADDRESS = "highest_address"


class SparseUpdateRule(StrEnum):
    DECAYED_DELTA = "decayed_delta"


class SparseReadTiming(StrEnum):
    CURRENT_STATE = "current_state"
    BEFORE_UPDATE = "before_update"
    AFTER_UPDATE = "after_update"


class SparseStatePolicy(StrEnum):
    PERSISTENT_IN_PLACE = "persistent_in_place"


class SparseStateOperation(StrEnum):
    """Closed execution vocabulary for the route-to-state mixer."""

    READ_ONLY = "read_only"
    UPDATE = "update"


class SparseStateLayout(StrEnum):
    """Logical state layout; physical tiling remains a lowering choice."""

    PARTITION_SLOT_VALUE = "partition_slot_value"


class SparseStateExecutionMode(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"


@dataclass(frozen=True, slots=True)
class SparseRouteSelectionSpec:
    """Restricted score-to-partition-local-route semantic operation.

    Pairwise additive factor scores are the product-key specialization used by
    the SDM comparison, but neither an external API nor a physical schedule is
    part of this contract.
    """

    parallel: int
    sequence: int
    source_extent: int
    route_width: int
    dtype: DType
    composition: SparseScoreComposition = (
        SparseScoreComposition.PAIRWISE_ADDITIVE_FACTORS
    )
    selection: SelectionKind = SelectionKind.TOP_K
    normalization: ScoreNormalization = ScoreNormalization.SOFTMAX
    canonicalization: SparseAddressCanonicalization = (
        SparseAddressCanonicalization.ASCENDING
    )
    tie_policy: SparseRouteTiePolicy = SparseRouteTiePolicy.HIGHEST_ADDRESS
    output_index_dtype: DType = DType.INT32
    partition_local: bool = True

    def __post_init__(self) -> None:
        if self.parallel <= 0 or self.sequence <= 0 or self.source_extent <= 0:
            raise ValueError("sparse route dimensions must be positive")
        if not 0 < self.route_width <= self.source_extent:
            raise ValueError("route width must be in [1, source_extent]")
        if self.dtype not in {DType.FLOAT32, DType.BFLOAT16}:
            raise ValueError("sparse route selection supports float32 and bfloat16")
        factor_extent = round(self.source_extent**0.5)
        if factor_extent * factor_extent != self.source_extent:
            raise ValueError(
                "pairwise factor composition requires square source extent"
            )
        if self.route_width > factor_extent:
            raise ValueError("route width must not exceed factor extent")
        exact = (
            self.composition is SparseScoreComposition.PAIRWISE_ADDITIVE_FACTORS
            and self.selection is SelectionKind.TOP_K
            and self.normalization is ScoreNormalization.SOFTMAX
            and self.canonicalization is SparseAddressCanonicalization.ASCENDING
            and self.tie_policy is SparseRouteTiePolicy.HIGHEST_ADDRESS
            and self.output_index_dtype in {DType.INT32, DType.INT64}
            and self.partition_local
        )
        if not exact:
            raise ValueError("sparse route selection is outside the frozen vocabulary")

    @property
    def factor_extent(self) -> int:
        return round(self.source_extent**0.5)

    @property
    def score_width(self) -> int:
        return 2 * self.factor_extent


@dataclass(frozen=True, slots=True)
class SparseMemoryMixerSpec:
    """URM-owned restricted algebra for ordered sparse-memory access.

    This describes mathematical choices only. It contains no upstream module,
    callable, tensor-layout convention, or kernel implementation identity.
    """

    parallel: int
    sequence: int
    slots_per_partition: int
    value_dim: int
    writes: int
    reads: int
    dtype: DType = DType.BFLOAT16
    mode: SDMExecutionMode = SDMExecutionMode.INFERENCE
    operation: SparseStateOperation = SparseStateOperation.UPDATE
    addressing: SparseAddressingKind = SparseAddressingKind.PRODUCT_KEY_TOP_K
    normalization: ScoreNormalization = ScoreNormalization.SOFTMAX
    update_rule: SparseUpdateRule = SparseUpdateRule.DECAYED_DELTA
    collision_policy: MergePolicy = MergePolicy.ORDERED
    within_token_collision_policy: MergePolicy = MergePolicy.REJECT
    read_timing: SparseReadTiming = SparseReadTiming.AFTER_UPDATE
    state_policy: SparseStatePolicy = SparseStatePolicy.PERSISTENT_IN_PLACE
    page_size: int = 1

    def __post_init__(self) -> None:
        dimensions = (
            "parallel",
            "sequence",
            "slots_per_partition",
            "value_dim",
            "reads",
        )
        for name in dimensions:
            if getattr(self, name) <= 0:
                raise ValueError(f"SDM {name} must be positive")
        if self.writes < 0:
            raise ValueError("sparse memory writes must be non-negative")
        if self.operation is SparseStateOperation.READ_ONLY:
            if (
                self.writes != 0
                or self.read_timing is not SparseReadTiming.CURRENT_STATE
            ):
                raise ValueError(
                    "read-only sparse memory requires writes=0 and current-state reads"
                )
            if self.mode is SDMExecutionMode.TRAINING:
                raise ValueError("read-only sparse memory does not advertise training")
        elif self.writes <= 0:
            raise ValueError("updating sparse memory requires writes > 0")
        root = round(self.slots_per_partition**0.5)
        if root * root != self.slots_per_partition:
            raise ValueError("SDM slots_per_partition must be a perfect square")
        if self.slots_per_partition % 8:
            raise ValueError("SDM slots_per_partition must be divisible by 8")
        max_width = min(128, self.slots_per_partition)
        if self.writes > max_width or self.reads > max_width:
            raise ValueError("SDM read/write widths exceed the frozen upstream subset")
        if self.reads > root or self.writes > root:
            raise ValueError(
                "product-key read/write widths must not exceed factor extent"
            )
        if self.dtype not in (DType.FLOAT32, DType.BFLOAT16):
            raise ValueError("SDM adapter supports float32 and bfloat16")
        if self.mode is SDMExecutionMode.TRAINING and self.sequence < 16:
            raise ValueError(
                "SDM upstream training kernel on the frozen runtime requires "
                "sequence >= 16"
            )


# Compatibility name for the external baseline slice. Native lowering work
# consumes SparseMemoryMixerSpec and must not depend on the upstream adapter.
SparseDeltaMemorySpec = SparseMemoryMixerSpec


@dataclass(frozen=True, slots=True)
class SparseStateMixerSpec:
    """Restricted URM-owned route-to-state algebra.

    Routes are already certified logical addresses and normalized weights.
    The spec does not describe how they were produced and contains no external
    library API, physical kernel layout, or unchecked tensor expression.
    """

    parallel: int
    sequence: int
    slots_per_partition: int
    value_dim: int
    writes: int
    reads: int
    dtype: DType
    operation: SparseStateOperation
    read_timing: SparseReadTiming
    mode: SparseStateExecutionMode = SparseStateExecutionMode.INFERENCE
    update_rule: SparseUpdateRule = SparseUpdateRule.DECAYED_DELTA
    collision_policy: MergePolicy = MergePolicy.ORDERED
    within_token_collision_policy: MergePolicy = MergePolicy.REJECT
    state_policy: SparseStatePolicy = SparseStatePolicy.PERSISTENT_IN_PLACE
    accumulation_dtype: DType = DType.FLOAT32
    state_layout: SparseStateLayout = SparseStateLayout.PARTITION_SLOT_VALUE
    page_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "parallel",
            "sequence",
            "slots_per_partition",
            "value_dim",
            "reads",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"SparseStateMixer {name} must be positive")
        if self.writes < 0:
            raise ValueError("SparseStateMixer writes must be non-negative")
        if self.reads > self.slots_per_partition:
            raise ValueError("read width must not exceed slots per partition")
        if self.operation is SparseStateOperation.READ_ONLY:
            if self.writes != 0:
                raise ValueError("read-only SparseStateMixer requires writes=0")
            if self.read_timing is not SparseReadTiming.CURRENT_STATE:
                raise ValueError("read-only SparseStateMixer reads current state")
        else:
            if self.writes <= 0:
                raise ValueError("updating SparseStateMixer requires writes > 0")
            if self.writes > self.slots_per_partition:
                raise ValueError("write width must not exceed slots per partition")
            if self.read_timing not in {
                SparseReadTiming.BEFORE_UPDATE,
                SparseReadTiming.AFTER_UPDATE,
            }:
                raise ValueError("updates require pre-update or post-update reads")
        if self.dtype not in {DType.FLOAT32, DType.BFLOAT16}:
            raise ValueError("SparseStateMixer v0 supports float32 and bfloat16")
        if self.accumulation_dtype is not DType.FLOAT32:
            raise ValueError("SparseStateMixer v0 requires float32 accumulation")
        if self.collision_policy is not MergePolicy.ORDERED:
            raise ValueError("SparseStateMixer requires ordered cross-token collisions")
        if self.within_token_collision_policy is not MergePolicy.REJECT:
            raise ValueError("SparseStateMixer rejects within-token collisions")
        if self.state_policy is not SparseStatePolicy.PERSISTENT_IN_PLACE:
            raise ValueError("SparseStateMixer v0 requires persistent state")
        if self.state_layout is not SparseStateLayout.PARTITION_SLOT_VALUE:
            raise ValueError("SparseStateMixer v0 requires partition-slot-value state")
        if self.page_size != 1:
            raise ValueError("SparseStateMixer v0 uses one logical slot per page")


@dataclass(frozen=True, slots=True)
class EpilogueSpec:
    """Typed, constrained epilogue attached to a reduction anchor.

    Epilogues are declarative descriptors interpreted by anchors; they are not
    Python callables. A delayed per-row scale is the first supported member.
    """

    kind: TransformKind  # ROW_SCALE only for now
    scale: str  # name of the logical scalar tensor

    def __post_init__(self) -> None:
        if self.kind is not TransformKind.ROW_SCALE:
            raise ValueError("only ROW_SCALE epilogues are defined")


@dataclass(frozen=True, slots=True)
class SemanticOp:
    """Base fields shared by every semantic operation."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("op name must not be empty")

    @property
    def effect(self) -> EffectSignature:
        return PURE


@dataclass(frozen=True, slots=True)
class Score(SemanticOp):
    """Per-(query, source) routing scores over logical domains."""

    spec: RouteSpec


@dataclass(frozen=True, slots=True)
class Select(SemanticOp):
    """Selection of route edges from scores (top-k/threshold/dense)."""

    spec: RouteSpec


@dataclass(frozen=True, slots=True)
class Gather(SemanticOp):
    """Bring payload rows to queries along precomputed logical edges."""

    spec: RouteSpec
    shape_hint: tuple[int, int, int, int] | None = None  # (Q, S, K, D)

    @property
    def effect(self) -> EffectSignature:
        return REDUCING


@dataclass(frozen=True, slots=True)
class WeightedReduce(SemanticOp):
    """``out[q, d] = sum_k w[q, k] * gathered[q, k, d]`` over the route width."""

    spec: RouteSpec
    epilogue: EpilogueSpec | None = None
    # Concrete dims for analytical costing (architecture params, not schedule):
    shape_hint: tuple[int, int, int, int] | None = None  # (Q, S, K, D)

    @property
    def effect(self) -> EffectSignature:
        return REDUCING


@dataclass(frozen=True, slots=True)
class Matmul(SemanticOp):
    """Trusted GEMM anchor semantics: ``out = lhs @ rhs``."""

    transpose_rhs: bool = False
    shape_hint: tuple[int, int, int] | None = None  # (M, K, N)


@dataclass(frozen=True, slots=True)
class Transform(SemanticOp):
    """Pointwise transform from the closed vocabulary."""

    kind: TransformKind = TransformKind.IDENTITY


@dataclass(frozen=True, slots=True)
class OrderedRecurrence(SemanticOp):
    """Ordered scan shell; the exact equation stays backend-owned.

    Ordered recurrence can never be reordered by a rewrite; it is a movement
    barrier by construction.
    """

    algorithm: str = "gated_delta_rule"

    @property
    def effect(self) -> EffectSignature:
        return ORDERED_STATE


@dataclass(frozen=True, slots=True)
class StateRead(SemanticOp):
    """Read a versioned state tensor at the current version."""

    state: str = ""
    versioned: bool = True

    @property
    def effect(self) -> EffectSignature:
        return STATE_READ_EFFECT


@dataclass(frozen=True, slots=True)
class StateUpdate(SemanticOp):
    """Merge deltas into state under an explicit collision policy.

    ``commit_boundary`` marks a transactional version boundary: readers after
    the commit observe exactly one merged write per address.
    """

    state: str = ""
    policy: MergePolicy = MergePolicy.NOT_APPLICABLE
    commit_boundary: bool = False

    def __post_init__(self) -> None:
        # NOTE: explicit base-class call; dataclass slots=True breaks zero-arg
        # super() inside regenerated classes.
        SemanticOp.__post_init__(self)
        if self.policy is MergePolicy.NOT_APPLICABLE:
            raise ValueError("state updates require an explicit merge policy")
        if self.commit_boundary and self.policy is MergePolicy.ORDERED:
            raise ValueError("transactional commits cannot use ordered (scan) merges")

    @property
    def effect(self) -> EffectSignature:  # type: ignore[override]
        if self.policy is MergePolicy.ORDERED:
            return ORDERED_STATE
        if self.commit_boundary:
            return COMMIT
        return ATOMIC_ACCUMULATE


@dataclass(frozen=True, slots=True)
class SparseMemoryAccess(SemanticOp):
    """Restricted sparse-memory skeleton; external SDM is one temporary anchor."""

    spec: SparseMemoryMixerSpec

    @property
    def effect(self) -> EffectSignature:
        return ORDERED_STATE


@dataclass(frozen=True, slots=True)
class SparseRouteGeneration(SemanticOp):
    """Pure score-to-route operation with explicit constrained semantics."""

    spec: SparseRouteSelectionSpec

    @property
    def effect(self) -> EffectSignature:
        return PURE


@dataclass(frozen=True, slots=True)
class SparseStateMixerAccess(SemanticOp):
    """Stateful route-to-memory skeleton lowered by URM-native kernels."""

    spec: SparseStateMixerSpec

    @property
    def effect(self) -> EffectSignature:
        if self.spec.operation is SparseStateOperation.READ_ONLY:
            return STATE_READ_EFFECT
        return ORDERED_STATE


# Compatibility name for users of the external-baseline constructor. The
# semantic node itself is the URM-owned sparse-memory skeleton above.
SparseDeltaMemoryAccess = SparseMemoryAccess


@dataclass(frozen=True, slots=True)
class CollectiveExchange(SemanticOp):
    """Collective semantic intent over a named mesh axis."""

    kind: CollectiveKind = CollectiveKind.ALL_REDUCE
    mesh_axis: str = "data"

    @property
    def effect(self) -> EffectSignature:
        return COLLECTIVE


SemanticNode = (
    Score
    | Select
    | Gather
    | WeightedReduce
    | Matmul
    | Transform
    | OrderedRecurrence
    | StateRead
    | StateUpdate
    | SparseMemoryAccess
    | SparseRouteGeneration
    | SparseStateMixerAccess
    | CollectiveExchange
)


@dataclass(frozen=True, slots=True)
class SemanticProgram:
    """A validated, ordered semantic program."""

    name: str
    inputs: tuple[TensorHandle, ...]
    ops: tuple[SemanticNode, ...]
    outputs: tuple[str, ...]

    @classmethod
    def build(
        cls,
        name: str,
        inputs: tuple[TensorHandle, ...],
        ops: tuple[SemanticNode, ...],
        outputs: tuple[str, ...],
    ) -> SemanticProgram:
        program = cls(name=name, inputs=inputs, ops=ops, outputs=outputs)
        program.validate()
        return program

    # -- accessors ---------------------------------------------------------

    @property
    def op_names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self.ops)

    def find(self, op_name: str) -> SemanticNode | None:
        for op in self.ops:
            if op.name == op_name:
                return op
        return None

    def producer_of(self, tensor: str) -> SemanticNode | None:
        if any(handle.name == tensor for handle in self.inputs):
            return None
        for op in self.ops:
            if tensor in op.outputs:
                return op
        return None

    def consumers_of(self, tensor: str) -> tuple[SemanticNode, ...]:
        return tuple(op for op in self.ops if tensor in op.inputs)

    def iter_ops(self) -> Iterator[SemanticNode]:
        return iter(self.ops)

    def replaced(self, ops: tuple[SemanticNode, ...]) -> SemanticProgram:
        """A new program with ``ops`` substituted; validate before compiling."""
        return SemanticProgram(
            name=self.name, inputs=self.inputs, ops=ops, outputs=self.outputs
        )

    # -- validation --------------------------------------------------------

    def validate(self) -> tuple[object, ...]:
        """Check name binding and structural rules; raise on errors."""
        collector = DiagnosticsCollector()
        defined: set[str] = {handle.name for handle in self.inputs}

        for op in self.ops:
            for tensor in op.inputs:
                if tensor not in defined:
                    collector.error(
                        DiagnosticCode.UNKNOWN_TENSOR,
                        f"{op.name}: input {tensor!r} is never defined before use",
                        subject=op.name,
                    )
            for tensor in op.outputs:
                if tensor in defined:
                    collector.error(
                        DiagnosticCode.DUPLICATE_DEFINITION,
                        f"{op.name}: output {tensor!r} is already defined",
                        subject=op.name,
                    )
                defined.add(tensor)

        for tensor in self.outputs:
            if tensor not in defined:
                collector.error(
                    DiagnosticCode.UNBOUND_INPUT,
                    f"program output {tensor!r} is never produced",
                )

        seen_commits: set[str] = set()
        for op in self.ops:
            if isinstance(op, StateUpdate) and op.commit_boundary:
                if op.state in seen_commits:
                    collector.error(
                        DiagnosticCode.MULTIPLE_COMMITS,
                        f"state {op.state!r} has more than one commit boundary",
                        subject=op.name,
                    )
                seen_commits.add(op.state)

        collector.raise_if_errors()
        return tuple(collector)


def routed_reduction_program(
    *,
    name: str = "routed_reduction",
    queries: Dim = "q",
    route_width: Dim = "k",
    sources: Dim = "s",
    value_dim: Dim = "d",
    value_dtype: DType = DType.FLOAT32,
    source_domain: LogicalDomain = LogicalDomain.SEQUENCE,
    selection: SelectionKind = SelectionKind.TOP_K,
    normalization: ScoreNormalization = ScoreNormalization.SOFTMAX,
    top_k: int | None = None,
) -> SemanticProgram:
    """The canonical routed-reduction program (pre-epilogue).

        base[q, d] = sum_k weights[q, k] * values[indices[q, k], d]

    Scores/selection are represented upstream of this program; indices and
    weights are inputs here, matching the frozen routed-reduction v1 contract.
    """

    if selection is SelectionKind.TOP_K and top_k is None:
        # Precomputed routes do not re-run selection; width is symbolic.
        selection = SelectionKind.DENSE
    spec = RouteSpec(
        query_domain=LogicalDomain.SEQUENCE,
        source_domain=source_domain,
        selection=selection,
        normalization=normalization,
        top_k=top_k,
        page_size=64 if source_domain is LogicalDomain.MEMORY_PAGE else None,
    )
    hint: tuple[int, int, int, int] | None = None
    if all(isinstance(v, int) for v in (queries, route_width, sources, value_dim)):
        hint = (int(queries), int(sources), int(route_width), int(value_dim))
    return SemanticProgram.build(
        name=name,
        inputs=(
            TensorHandle("indices", DType.INT64, ("queries", route_width)),
            TensorHandle("weights", value_dtype, ("queries", route_width)),
            TensorHandle("values", value_dtype, (sources, value_dim)),
        ),
        ops=(
            Gather(
                name="gather",
                inputs=("indices", "values"),
                outputs=("gathered",),
                spec=spec,
                shape_hint=hint,
            ),
            WeightedReduce(
                name="reduce",
                inputs=("gathered", "weights"),
                outputs=("base",),
                spec=spec,
                shape_hint=hint,
            ),
        ),
        outputs=("base",),
    )


def sparse_delta_memory_program(
    *,
    name: str = "sparse_delta_memory",
    parallel: int = 1,
    sequence: int = 128,
    slots_per_partition: int = 4096,
    value_dim: int = 256,
    writes: int = 64,
    reads: int = 64,
    dtype: DType = DType.BFLOAT16,
    mode: SDMExecutionMode = SDMExecutionMode.INFERENCE,
    operation: SparseStateOperation = SparseStateOperation.UPDATE,
    read_timing: SparseReadTiming = SparseReadTiming.AFTER_UPDATE,
) -> SemanticProgram:
    """Compatibility builder for the typed sparse-memory mixer skeleton."""
    spec = SparseMemoryMixerSpec(
        parallel=parallel,
        sequence=sequence,
        slots_per_partition=slots_per_partition,
        value_dim=value_dim,
        writes=writes,
        reads=reads,
        dtype=dtype,
        mode=mode,
        operation=operation,
        read_timing=read_timing,
    )
    root = round(slots_per_partition**0.5)
    inputs = [
        TensorHandle("read_scores", dtype, (parallel, sequence, 2 * root)),
        TensorHandle(
            "memory",
            dtype,
            (parallel, slots_per_partition, value_dim),
        ),
    ]
    op_inputs = ["read_scores", "memory"]
    outputs = ["readings", "updated_memory", "read_addresses", "read_weights"]
    if operation is SparseStateOperation.UPDATE:
        inputs = [
            TensorHandle("write_scores", dtype, (parallel, sequence, 2 * root)),
            TensorHandle("values", dtype, (parallel, sequence, value_dim)),
            TensorHandle("beta", dtype, (parallel, sequence, 1)),
            TensorHandle("log_decay", dtype, (parallel, sequence, 1)),
            *inputs,
        ]
        op_inputs = [
            "write_scores",
            "read_scores",
            "values",
            "beta",
            "log_decay",
            "memory",
        ]
        outputs = [
            "readings",
            "updated_memory",
            "write_addresses",
            "write_weights",
            "read_addresses",
            "read_weights",
        ]
    return SemanticProgram.build(
        name=name,
        inputs=tuple(inputs),
        ops=(
            SparseMemoryAccess(
                name="sdm_access",
                inputs=tuple(op_inputs),
                outputs=tuple(outputs),
                spec=spec,
            ),
        ),
        outputs=tuple(outputs),
    )


def sparse_route_selection_program(
    *,
    name: str = "sparse_route_selection",
    parallel: int = 1,
    sequence: int = 128,
    source_extent: int = 4096,
    route_width: int = 64,
    dtype: DType = DType.BFLOAT16,
) -> SemanticProgram:
    """Build the independently compilable score-to-route operation."""
    spec = SparseRouteSelectionSpec(
        parallel=parallel,
        sequence=sequence,
        source_extent=source_extent,
        route_width=route_width,
        dtype=dtype,
    )
    return SemanticProgram.build(
        name=name,
        inputs=(TensorHandle("scores", dtype, (parallel, sequence, spec.score_width)),),
        ops=(
            SparseRouteGeneration(
                name="sparse_route_generation",
                inputs=("scores",),
                outputs=("addresses", "weights"),
                spec=spec,
            ),
        ),
        outputs=("addresses", "weights"),
    )


def sparse_state_mixer_program(
    *,
    name: str = "sparse_state_mixer",
    parallel: int = 1,
    sequence: int = 128,
    slots_per_partition: int = 4096,
    value_dim: int = 256,
    writes: int = 4,
    reads: int = 4,
    dtype: DType = DType.BFLOAT16,
    operation: SparseStateOperation = SparseStateOperation.UPDATE,
    read_timing: SparseReadTiming = SparseReadTiming.AFTER_UPDATE,
    mode: SparseStateExecutionMode = SparseStateExecutionMode.INFERENCE,
) -> SemanticProgram:
    """Build the native kernel-only operation over certified logical routes."""
    spec = SparseStateMixerSpec(
        parallel=parallel,
        sequence=sequence,
        slots_per_partition=slots_per_partition,
        value_dim=value_dim,
        writes=writes,
        reads=reads,
        dtype=dtype,
        operation=operation,
        read_timing=read_timing,
        mode=mode,
    )
    inputs = [
        TensorHandle("read_addresses", DType.INT64, (parallel, sequence, reads)),
        TensorHandle("read_weights", dtype, (parallel, sequence, reads)),
        TensorHandle("memory", dtype, (parallel, slots_per_partition, value_dim)),
    ]
    op_inputs = ["read_addresses", "read_weights", "memory"]
    if operation is SparseStateOperation.UPDATE:
        inputs = [
            TensorHandle("write_addresses", DType.INT64, (parallel, sequence, writes)),
            TensorHandle("write_weights", dtype, (parallel, sequence, writes)),
            TensorHandle("values", dtype, (parallel, sequence, value_dim)),
            TensorHandle("beta", dtype, (parallel, sequence, 1)),
            TensorHandle("log_decay", dtype, (parallel, sequence, 1)),
            *inputs,
        ]
        op_inputs = [
            "write_addresses",
            "write_weights",
            "values",
            "beta",
            "log_decay",
            *op_inputs,
        ]
    return SemanticProgram.build(
        name=name,
        inputs=tuple(inputs),
        ops=(
            SparseStateMixerAccess(
                name="sparse_state_mixer",
                inputs=tuple(op_inputs),
                outputs=("readings", "updated_memory"),
                spec=spec,
            ),
        ),
        outputs=("readings", "updated_memory"),
    )


def row_scaled_routed_reduction_program(
    *,
    name: str = "row_scaled_routed_reduction",
    queries: Dim = "q",
    route_width: Dim = "k",
    sources: Dim = "s",
    value_dim: Dim = "d",
    value_dtype: DType = DType.FLOAT32,
    source_domain: LogicalDomain = LogicalDomain.SEQUENCE,
    selection: SelectionKind = SelectionKind.TOP_K,
    normalization: ScoreNormalization = ScoreNormalization.SOFTMAX,
    top_k: int | None = None,
) -> SemanticProgram:
    """Routed reduction followed by an explicit per-row scale transform.

        output[q, d] = row_scale[q] * base[q, d]

    This is the materialized reference form for the CODA-style epilogue
    reparameterization (docs/coda-retrospective.md).
    """

    base_program = routed_reduction_program(
        name=name,
        queries=queries,
        route_width=route_width,
        sources=sources,
        value_dim=value_dim,
        value_dtype=value_dtype,
        source_domain=source_domain,
        selection=selection,
        normalization=normalization,
        top_k=top_k,
    )
    ops = (*base_program.ops[:1], *base_program.ops[1:])
    scale_op = Transform(
        name="apply_row_scale",
        inputs=("base", "row_scale"),
        outputs=("output",),
        kind=TransformKind.ROW_SCALE,
    )
    return SemanticProgram.build(
        name=name,
        inputs=(
            *base_program.inputs,
            TensorHandle("row_scale", value_dtype, ("queries",)),
        ),
        ops=(*ops, scale_op),
        outputs=("output",),
    )
