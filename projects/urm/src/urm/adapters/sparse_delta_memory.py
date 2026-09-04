"""Typed adapter for the original Facebook Sparse Delta Memory repository.

This is an external baseline integration, not a native URM lowering.  It calls
the pinned checkout's product-key, sparse-read, and gated write/read methods
without copying or modifying any upstream kernel source.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

EXPECTED_SDM_COMMIT = "183e7df809131b80ad4393741029d0f20fc3640b"
SDM_REPOSITORY = "https://github.com/facebookresearch/sparse-delta-memory"
SDM_LICENSE = "CC-BY-NC-4.0"
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
_TRACE_CERTIFICATE = object()


class SDMAdapterMode(StrEnum):
    """Execution paths exposed by the external adapter boundary."""

    READ_ONLY = "read_only"
    INFERENCE = "inference"
    TRAINING = "training"


MODE_READ_ONLY = SDMAdapterMode.READ_ONLY
MODE_INFERENCE = SDMAdapterMode.INFERENCE
MODE_TRAINING = SDMAdapterMode.TRAINING


class SDMTraceOrigin(StrEnum):
    """How a certified trace was obtained without admitting arbitrary routes."""

    PRODUCT_KEY_GENERATED = "product_key_generated"
    PRODUCT_KEY_DERIVED = "product_key_derived"


@dataclass(frozen=True, slots=True)
class SDMSupportStatus:
    supported: bool
    code: str
    reason: str | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "code": self.code,
            "reason": self.reason,
            "details": dict(self.details),
        }


def _git_output(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _checkout_root(module_file: str) -> Path | None:
    path = Path(module_file).resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent
    return None


def sdm_upstream_identity() -> dict[str, object]:
    """Resolve the installed checkout and verify its exact Git revision."""
    try:
        from lingua.sparse_delta_memory import SparseDeltaMemory

        source = Path(inspect.getfile(SparseDeltaMemory)).resolve()
        root = _checkout_root(str(source))
        revision = _git_output(root, "rev-parse", "HEAD") if root else None
        dirty = bool(_git_output(root, "status", "--porcelain")) if root else None
        versions = {}
        for package in ("torch", "triton", "einops", "ninja"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {
            "repository": SDM_REPOSITORY,
            "expected_commit": EXPECTED_SDM_COMMIT,
            "installed_commit": revision,
            "revision_compatible": revision == EXPECTED_SDM_COMMIT,
            "checkout_root": str(root) if root else None,
            "module_file": str(source),
            "checkout_dirty": dirty,
            "license": SDM_LICENSE,
            "installation": "git checkout plus PYTHONPATH (upstream has no package manifest)",
            "runtime_versions": versions,
            "runtime_requirements": {
                "upstream_declared_python": "3.11 (setup/create_env.sh)",
                "upstream_declared_torch": ">=2.8",
                "upstream_declared_triton": ">=3.4",
                "adapter_validated_python": sys.version.split()[0],
                "adapter_frozen_torch": "2.8.0",
                "adapter_frozen_triton": "3.4.0",
                "cuda": "runtime toolkit with nvcc, CUDART/driver/CRT/NVVM development files, gcc/g++, SM80+",
            },
            "entry_points": {
                "address": "lingua.sparse_delta_memory.layer.SparseDeltaMemory._get_product_keys_for_head",
                "read": "lingua.sparse_delta_memory.layer.SparseDeltaMemory.read",
                "update": "lingua.sparse_delta_memory.layer.SparseDeltaMemory.gated_write_read",
                "cache": "lingua.sparse_delta_memory.cache.SDMLayerState",
            },
            "source_usage": "external checkout; no upstream source vendored or modified",
        }
    except Exception as error:  # noqa: BLE001 - structured optional-dependency probe
        return {
            "status": "not_applicable",
            "reason": repr(error),
            "repository": SDM_REPOSITORY,
            "expected_commit": EXPECTED_SDM_COMMIT,
        }


def probe_sdm_support(*, require_cuda: bool = True) -> SDMSupportStatus:
    identity = sdm_upstream_identity()
    if identity.get("status") == "not_applicable":
        return SDMSupportStatus(
            False,
            "missing_dependency",
            f"original SDM checkout is unavailable: {identity['reason']}",
            identity,
        )
    if not identity.get("revision_compatible"):
        return SDMSupportStatus(
            False,
            "incompatible_revision",
            f"installed revision {identity.get('installed_commit')!r} does not equal pinned {EXPECTED_SDM_COMMIT}",
            identity,
        )
    if identity.get("checkout_dirty"):
        return SDMSupportStatus(
            False,
            "modified_upstream_checkout",
            "upstream checkout contains local modifications",
            identity,
        )
    versions = identity["runtime_versions"]
    if versions.get("torch") != "2.8.0" or versions.get("triton") != "3.4.0":
        return SDMSupportStatus(
            False,
            "incompatible_runtime",
            "frozen baseline requires torch==2.8.0 and triton==3.4.0",
            identity,
        )
    missing_helpers = [
        package for package in ("einops", "ninja") if versions.get(package) is None
    ]
    if missing_helpers:
        return SDMSupportStatus(
            False,
            "missing_dependency",
            f"missing upstream runtime helpers: {missing_helpers}",
            identity,
        )
    if require_cuda:
        if not torch.cuda.is_available():
            return SDMSupportStatus(
                False, "unsupported_hardware", "CUDA is unavailable", identity
            )
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            return SDMSupportStatus(
                False,
                "unsupported_hardware",
                f"SM80+ is required; found sm{capability[0]}{capability[1]}",
                {**identity, "compute_capability": list(capability)},
            )
        tools = {name: shutil.which(name) for name in ("nvcc", "g++", "ninja")}
        missing_tools = [name for name, path in tools.items() if path is None]
        if missing_tools:
            return SDMSupportStatus(
                False,
                "missing_dependency",
                f"missing upstream CUDA build tools: {missing_tools}",
                {**identity, "build_tools": tools},
            )
        library_dirs = [
            Path(item)
            for item in os.environ.get("LIBRARY_PATH", "").split(os.pathsep)
            if item
        ]
        if not any((path / "libcudart.so").exists() for path in library_dirs):
            return SDMSupportStatus(
                False,
                "missing_dependency",
                "LIBRARY_PATH must include a directory containing libcudart.so "
                "for the upstream runtime extension build",
                {**identity, "build_tools": tools},
            )
        identity = {
            **identity,
            "compute_capability": list(capability),
            "build_tools": tools,
            "library_path": [str(path) for path in library_dirs],
        }
    return SDMSupportStatus(True, "supported", None, identity)


@dataclass(frozen=True, slots=True)
class SDMAddressTrace:
    """Certified route trace with global addresses in ``[P,T,K]`` layout."""

    write_indices: torch.Tensor
    write_weights: torch.Tensor
    read_indices: torch.Tensor
    read_weights: torch.Tensor
    slots_per_partition: int
    origin: SDMTraceOrigin
    _tensor_versions: tuple[int, int, int, int] = field(repr=False, compare=False)
    _certificate: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _TRACE_CERTIFICATE:
            raise ValueError("SDMAddressTrace must be created by a certification path")

    @classmethod
    def _certify(
        cls,
        write_indices: torch.Tensor,
        write_weights: torch.Tensor,
        read_indices: torch.Tensor,
        read_weights: torch.Tensor,
        *,
        slots_per_partition: int,
        origin: SDMTraceOrigin,
    ) -> SDMAddressTrace:
        if write_indices.ndim != 3 or read_indices.ndim != 3:
            raise ValueError("SDM traces require rank-3 [P,T,K] indices")
        if write_weights.shape != write_indices.shape:
            raise ValueError("write weights must match write indices")
        if read_weights.shape != read_indices.shape:
            raise ValueError("read weights must match read indices")
        if write_indices.shape[:2] != read_indices.shape[:2]:
            raise ValueError("read and write traces must share [P,T]")
        if write_indices.dtype != torch.int64 or read_indices.dtype != torch.int64:
            raise ValueError("SDM addresses must be int64")
        tensors = (write_indices, write_weights, read_indices, read_weights)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("trace tensors must share a device")
        if write_weights.dtype != read_weights.dtype:
            raise ValueError("read and write weight dtypes must match")
        if write_weights.dtype not in SUPPORTED_DTYPES:
            raise ValueError("trace weights must use a supported floating dtype")
        if not all(tensor.is_contiguous() for tensor in tensors):
            raise ValueError("trace tensors must be contiguous")
        p, sequence, writes = write_indices.shape
        reads = read_indices.shape[-1]
        if p < 1 or sequence < 1 or writes < 1 or reads < 1:
            raise ValueError(
                "empty SDM routing is unsupported; P,T,W,R must be positive"
            )
        if slots_per_partition < 8 or slots_per_partition % 8:
            raise ValueError("slots_per_partition must be >=8 and divisible by 8")
        for name, indices in (
            ("write", write_indices),
            ("read", read_indices),
        ):
            local = (
                indices
                - torch.arange(p, device=indices.device, dtype=torch.int64).view(
                    p, 1, 1
                )
                * slots_per_partition
            )
            if bool(((local < 0) | (local >= slots_per_partition)).any().item()):
                raise ValueError(f"{name} addresses cross partition bounds")
            if indices.shape[-1] > 1 and bool(
                (indices[..., 1:] <= indices[..., :-1]).any().item()
            ):
                raise ValueError(
                    f"{name} addresses must be strictly increasing; "
                    "within-token duplicates/ties are unsupported"
                )
        normalization_atol = 2e-5 if write_weights.dtype == torch.float32 else 2e-3
        for name, weights in (
            ("write", write_weights),
            ("read", read_weights),
        ):
            if not bool(torch.isfinite(weights).all().item()):
                raise ValueError(f"{name} weights must be finite")
            if bool((weights < 0).any().item()):
                raise ValueError(f"{name} weights must be nonnegative")
            sums = weights.float().sum(dim=-1)
            if not torch.allclose(
                sums,
                torch.ones_like(sums),
                atol=normalization_atol,
                rtol=0,
            ):
                raise ValueError(
                    f"{name} weights must satisfy the frozen Softmax normalization"
                )
        tensor_versions = tuple(tensor._version for tensor in tensors)
        return cls(
            write_indices,
            write_weights,
            read_indices,
            read_weights,
            slots_per_partition,
            origin,
            tensor_versions,
            _TRACE_CERTIFICATE,
        )

    @classmethod
    def _from_product_key(
        cls,
        write_indices: torch.Tensor,
        write_weights: torch.Tensor,
        read_indices: torch.Tensor,
        read_weights: torch.Tensor,
        *,
        slots_per_partition: int,
    ) -> SDMAddressTrace:
        return cls._certify(
            write_indices,
            write_weights,
            read_indices,
            read_weights,
            slots_per_partition=slots_per_partition,
            origin=SDMTraceOrigin.PRODUCT_KEY_GENERATED,
        )

    def token_slice(self, start: int, stop: int) -> SDMAddressTrace:
        """Create a certified contiguous token view derived from this trace."""
        if not 0 <= start < stop <= self.sequence:
            raise ValueError("token slice must satisfy 0 <= start < stop <= sequence")
        self._require_intact()
        return self._certify(
            self.write_indices[:, start:stop].contiguous(),
            self.write_weights[:, start:stop].contiguous(),
            self.read_indices[:, start:stop].contiguous(),
            self.read_weights[:, start:stop].contiguous(),
            slots_per_partition=self.slots_per_partition,
            origin=SDMTraceOrigin.PRODUCT_KEY_DERIVED,
        )

    def with_differentiable_weights(
        self, write_weights: torch.Tensor, read_weights: torch.Tensor
    ) -> SDMAddressTrace:
        """Attach cloned autograd leaves without changing certified route values."""
        self._require_intact()
        if not torch.equal(write_weights.detach(), self.write_weights.detach()):
            raise ValueError("replacement write weights must equal the certified trace")
        if not torch.equal(read_weights.detach(), self.read_weights.detach()):
            raise ValueError("replacement read weights must equal the certified trace")
        return self._certify(
            self.write_indices,
            write_weights,
            self.read_indices,
            read_weights,
            slots_per_partition=self.slots_per_partition,
            origin=SDMTraceOrigin.PRODUCT_KEY_DERIVED,
        )

    def _require_intact(self) -> None:
        tensors = (
            self.write_indices,
            self.write_weights,
            self.read_indices,
            self.read_weights,
        )
        if (
            self._certificate is not _TRACE_CERTIFICATE
            or tuple(tensor._version for tensor in tensors) != self._tensor_versions
        ):
            raise ValueError(
                "certified SDM trace tensors were mutated after construction"
            )

    @property
    def parallel(self) -> int:
        return self.write_indices.shape[0]

    @property
    def sequence(self) -> int:
        return self.write_indices.shape[1]


@dataclass(slots=True)
class SDMState:
    """Persistent mutable state; ``memory`` is changed by upstream in place."""

    memory: torch.Tensor
    sequence_length: int = 0


@dataclass(frozen=True, slots=True)
class SDMAdapterConfig:
    """Static runtime configuration against which every trace is checked."""

    slots_per_partition: int
    value_dim: int
    num_writes: int
    num_reads: int
    chunk_size: int
    mode: SDMAdapterMode
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class SDMOperationSpec:
    parallel: int
    sequence: int
    writes: int
    reads: int
    slots_per_partition: int
    value_dim: int
    dtype: torch.dtype
    mode: SDMAdapterMode
    mutation_order: str = "decay_retrieve_delta_scatter_then_read"
    collision_semantics: str = "ordered_across_tokens_unique_within_token"

    @classmethod
    def from_call(
        cls,
        trace: SDMAddressTrace,
        state: SDMState,
        values: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        log_decay: torch.Tensor | None = None,
        grad_final_memory: torch.Tensor | None = None,
        *,
        mode: SDMAdapterMode,
        config: SDMAdapterConfig,
    ) -> SDMOperationSpec:
        if mode not in (MODE_READ_ONLY, MODE_INFERENCE, MODE_TRAINING):
            raise ValueError(f"unknown SDM mode {mode!r}")
        p, sequence, writes = trace.write_indices.shape
        reads = trace.read_indices.shape[-1]
        memory = state.memory
        trace._require_intact()
        if mode != config.mode:
            raise ValueError(
                f"operation mode {mode!r} does not match adapter mode {config.mode!r}"
            )
        if trace.slots_per_partition != config.slots_per_partition:
            raise ValueError("trace slots_per_partition does not match adapter config")
        if writes != config.num_writes or reads != config.num_reads:
            raise ValueError("trace read/write widths do not match adapter config")
        if not isinstance(state.sequence_length, int) or state.sequence_length < 0:
            raise ValueError("state sequence_length must be a non-negative integer")
        expected_memory = (p * config.slots_per_partition, config.value_dim)
        if memory.ndim != 2 or memory.shape != expected_memory:
            raise ValueError(
                f"memory must match configured [P*slots_per_partition,D]={expected_memory}"
            )
        if memory.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"SDM baseline supports {SUPPORTED_DTYPES}")
        if not memory.is_cuda:
            raise ValueError("original SDM kernels require CUDA memory")
        if memory.device != config.device or memory.dtype != config.dtype:
            raise ValueError("state device/dtype does not match adapter config")
        if not memory.is_contiguous():
            raise ValueError("memory must be contiguous")
        for tensor in (
            trace.write_indices,
            trace.write_weights,
            trace.read_indices,
            trace.read_weights,
        ):
            if tensor.device != memory.device:
                raise ValueError("state and trace tensors must share a device")
        if trace.write_weights.dtype != memory.dtype:
            raise ValueError("trace weights and memory dtype must match")
        if mode != MODE_READ_ONLY:
            expected = (p, sequence, memory.shape[1])
            if values is None or values.shape != expected:
                raise ValueError(f"values must have shape {expected}")
            if beta is None or beta.shape != (p, sequence, 1):
                raise ValueError("beta must have shape [P,T,1]")
            if log_decay is None or log_decay.shape != (p, sequence, 1):
                raise ValueError("log_decay must have shape [P,T,1]")
            for tensor in (values, beta, log_decay):
                if (
                    tensor.device != memory.device
                    or tensor.dtype != memory.dtype
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "values/beta/log_decay must be contiguous and match state device/dtype"
                    )
            if mode == MODE_TRAINING and min(sequence, config.chunk_size) < 16:
                raise ValueError(
                    "upstream training kernel on the frozen runtime requires "
                    "effective sequence/chunk >= 16"
                )
            if grad_final_memory is not None:
                if mode != MODE_TRAINING:
                    raise ValueError(
                        "grad_final_memory is supported only by training mode"
                    )
                if (
                    grad_final_memory.shape != memory.shape
                    or grad_final_memory.device != memory.device
                    or grad_final_memory.dtype != memory.dtype
                    or not grad_final_memory.is_contiguous()
                ):
                    raise ValueError(
                        "grad_final_memory must be contiguous and match state shape/device/dtype"
                    )
        elif grad_final_memory is not None:
            raise ValueError("grad_final_memory is supported only by training mode")
        return cls(
            p,
            sequence,
            writes,
            reads,
            trace.slots_per_partition,
            memory.shape[1],
            memory.dtype,
            mode,
        )


class UrmSparseDeltaMemoryAdapter:
    """URM baseline dispatch around the pinned original SDM call sites."""

    name = "facebook_sparse_delta_memory_external_adapter"

    def __init__(
        self,
        *,
        slots_per_partition: int,
        value_dim: int,
        num_writes: int,
        num_reads: int,
        chunk_size: int = 64,
        mode: SDMAdapterMode = MODE_INFERENCE,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if mode not in (MODE_READ_ONLY, MODE_INFERENCE, MODE_TRAINING):
            raise ValueError(f"unknown SDM mode {mode!r}")
        if slots_per_partition < 8 or slots_per_partition % 8:
            raise ValueError("slots_per_partition must be >=8 and divisible by 8")
        root = round(slots_per_partition**0.5)
        if root * root != slots_per_partition:
            raise ValueError("upstream product-key slots must be a perfect square")
        if not 0 < num_writes <= min(128, slots_per_partition):
            raise ValueError("num_writes must be in [1,min(128,slots)]")
        if not 0 < num_reads <= min(128, slots_per_partition):
            raise ValueError("num_reads must be in [1,min(128,slots)]")
        if value_dim < 1:
            raise ValueError("value_dim must be positive")
        if chunk_size < 16:
            raise ValueError(
                "chunk_size must be >= 16 for upstream tensor-core kernels"
            )
        support = probe_sdm_support()
        if not support.supported:
            raise RuntimeError(
                f"{self.name} unavailable [{support.code}]: {support.reason}"
            )
        configured_device = torch.device(device)
        if configured_device.type != "cuda":
            raise ValueError("original SDM kernels require a CUDA adapter device")
        if configured_device.index is None:
            configured_device = torch.device("cuda", torch.cuda.current_device())
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"SDM adapter dtype must be one of {SUPPORTED_DTYPES}")
        self.config = SDMAdapterConfig(
            slots_per_partition=slots_per_partition,
            value_dim=value_dim,
            num_writes=num_writes,
            num_reads=num_reads,
            chunk_size=chunk_size,
            mode=mode,
            device=configured_device,
            dtype=dtype,
        )
        from lingua.sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

        args = SparseDeltaMemoryArgs(
            dim=value_dim,
            num_heads=1,
            slots_per_head=slots_per_partition,
            num_writes=num_writes,
            num_reads=num_reads,
            memory_block_size=chunk_size,
            read_act="Softmax",
            write_act="Softmax",
            backprop_on_memory=False,
            key_weighted_decay=False,
            snapshot_quant="none",
        )
        self.layer = SparseDeltaMemory(args, layer_id=0)
        self.mode = mode
        self.support = support
        self._address_fn = self.layer._get_product_keys_for_head
        self._read_fn = self.layer.read
        self._update_fn = self.layer.gated_write_read
        self.layer.train(mode == MODE_TRAINING)

    @property
    def direct_calls(self) -> dict[str, Any]:
        """Exact bound upstream methods used below URM dispatch."""
        return {
            "address": self._address_fn,
            "read": self._read_fn,
            "update": self._update_fn,
        }

    def generate_trace(
        self, write_scores: torch.Tensor, read_scores: torch.Tensor
    ) -> SDMAddressTrace:
        """Generate, normalize, offset, and certify the exact upstream trace."""
        if write_scores.ndim != 3 or read_scores.ndim != 3:
            raise ValueError("address scores require [P,T,2*sqrt(slots)]")
        if write_scores.shape[:2] != read_scores.shape[:2]:
            raise ValueError("write/read scores must share [P,T]")
        if (
            not write_scores.is_cuda
            or write_scores.device != self.config.device
            or write_scores.dtype != self.config.dtype
            or read_scores.device != write_scores.device
            or not write_scores.is_contiguous()
            or not read_scores.is_contiguous()
        ):
            raise ValueError("address scores must match configured CUDA device/dtype")
        expected_key_dim = 2 * round(self.layer.slots_per_head**0.5)
        if (
            write_scores.shape[-1] != expected_key_dim
            or read_scores.shape[-1] != expected_key_dim
        ):
            raise ValueError(f"score width must be {expected_key_dim}")
        if read_scores.dtype != write_scores.dtype:
            raise ValueError("address scores must share the configured dtype")
        write_values, write_indices = self._address_fn(
            write_scores, self.layer.args.num_writes, expected_key_dim // 2
        )
        read_values, read_indices = self._address_fn(
            read_scores, self.layer.args.num_reads, expected_key_dim // 2
        )
        write_weights = self.layer.write_act(write_values)
        read_weights = self.layer.read_act(read_values)
        parallel = write_scores.shape[0]
        offsets = (
            torch.arange(parallel, device=write_scores.device, dtype=torch.int64).view(
                parallel, 1, 1
            )
            * self.layer.slots_per_head
        )
        return SDMAddressTrace._from_product_key(
            (write_indices + offsets).contiguous(),
            write_weights.contiguous(),
            (read_indices + offsets).contiguous(),
            read_weights.contiguous(),
            slots_per_partition=self.layer.slots_per_head,
        )

    def read(self, state: SDMState, trace: SDMAddressTrace, *, return_info=False):
        spec = SDMOperationSpec.from_call(
            trace, state, mode=MODE_READ_ONLY, config=self.config
        )
        output = self._read_fn(state.memory, trace.read_weights, trace.read_indices)
        if return_info:
            return output, self._info(spec)
        return output

    def execute(
        self,
        state: SDMState,
        trace: SDMAddressTrace,
        values: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        *,
        grad_final_memory: torch.Tensor | None = None,
        return_info: bool = False,
    ):
        spec = SDMOperationSpec.from_call(
            trace,
            state,
            values,
            beta,
            log_decay,
            grad_final_memory,
            mode=self.mode,
            config=self.config,
        )
        if self.mode == MODE_READ_ONLY:
            raise ValueError("read-only adapter uses read(), not execute()")
        if grad_final_memory is None:
            readings, memory = self._update_fn(
                state.memory,
                trace.write_indices,
                trace.write_weights,
                values,
                beta,
                log_decay,
                trace.read_indices,
                trace.read_weights,
            )
        else:
            readings, memory = self._update_fn(
                state.memory,
                trace.write_indices,
                trace.write_weights,
                values,
                beta,
                log_decay,
                trace.read_indices,
                trace.read_weights,
                grad_final_memory=grad_final_memory,
            )
        # Upstream mutates memory during the call.  The sequence count becomes
        # visible only after that call returns, matching SDMLayerState.update_.
        state.memory = memory
        state.sequence_length += spec.sequence
        if return_info:
            return readings, state, self._info(spec)
        return readings, state

    def _info(self, spec: SDMOperationSpec) -> dict[str, object]:
        return {
            "adapter_backend": self.name,
            "native_urm_lowering": False,
            "upstream_lowering": f"facebookresearch/sparse-delta-memory@{EXPECTED_SDM_COMMIT}",
            "mode": spec.mode,
            "spec": {
                "parallel": spec.parallel,
                "sequence": spec.sequence,
                "writes": spec.writes,
                "reads": spec.reads,
                "slots_per_partition": spec.slots_per_partition,
                "value_dim": spec.value_dim,
                "dtype": str(spec.dtype),
                "mutation_order": spec.mutation_order,
                "collision_semantics": spec.collision_semantics,
            },
            "upstream": self.support.details,
        }


__all__ = [
    "EXPECTED_SDM_COMMIT",
    "MODE_INFERENCE",
    "MODE_READ_ONLY",
    "MODE_TRAINING",
    "SDMAdapterConfig",
    "SDMAdapterMode",
    "SDMAddressTrace",
    "SDMOperationSpec",
    "SDMState",
    "SDMSupportStatus",
    "SDMTraceOrigin",
    "UrmSparseDeltaMemoryAdapter",
    "probe_sdm_support",
    "sdm_upstream_identity",
]
