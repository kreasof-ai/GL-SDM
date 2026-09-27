"""Compile snapshot reads against the frozen, installed URM package."""
from functools import lru_cache
from importlib.metadata import distribution
from contextlib import contextmanager
from dataclasses import replace
from threading import RLock
from unittest.mock import patch
import json

URM_REVISION = "604bfdf5d2c827266a32ef142ca996cc712d70f0"
URM_REPOSITORY = "https://github.com/kreasof-ai/urm.git"
_compile_lock = RLock()


@contextmanager
def large_route_override(enabled):
    """Experimental, temporary support declaration for F=512, K=8, FP32.

    The normal compiler and native providers still execute the original shape.
    Only this declared shape is extended; dependency/hardware checks are rerun
    through the original probe. No installed source or dependency pin changes.
    """
    with _compile_lock:
        if not enabled:
            yield
            return
        verify_dependency()
        from urm.backends.triton.k3.route_generation import TritonSparseRouteBackend
        from urm.ir.program import DType
        original = TritonSparseRouteBackend.support_status
        def support(spec):
            if (spec.factor_extent == 512 and spec.route_width == 8
                    and spec.dtype is DType.FLOAT32 and spec.output_index_dtype is DType.INT32):
                return original(replace(spec, source_extent=256 ** 2))
            return original(spec)
        with patch.object(TritonSparseRouteBackend, "support_status", staticmethod(support)):
            yield


@lru_cache(None)
def verify_dependency():
    package = distribution("urm-kernel-lab")
    origin = json.loads(package.read_text("direct_url.json") or "{}")
    if origin.get("url") != URM_REPOSITORY or origin.get("vcs_info", {}).get("commit_id") != URM_REVISION:
        raise RuntimeError("GL-SDM requires the frozen URM installation from shared/requirements-urm.txt")
    return {"repository": URM_REPOSITORY, "revision": URM_REVISION}


@lru_cache(maxsize=128)
def read_plan(slots, queries, dim, width):
    verify_dependency()
    from urm.compiler.pipeline import CompilationIntent, compile_graph
    from urm.ir.program import (DType, SparseStateOperation, SparseReadTiming,
                                SparseStateExecutionMode, sparse_state_mixer_program)
    program = sparse_state_mixer_program(
        name="gl_sdm_snapshot_read", parallel=1, sequence=queries,
        slots_per_partition=slots, value_dim=dim, writes=0, reads=width,
        dtype=DType.FLOAT32, operation=SparseStateOperation.READ_ONLY,
        read_timing=SparseReadTiming.CURRENT_STATE, mode=SparseStateExecutionMode.TRAINING,
    )
    return compile_graph(program, target="native", intent=CompilationIntent.TRAINING)


def snapshot_read(values, addresses, weights):
    # Request/head offsets put each query in its own disjoint address region.
    # This avoids copying an entire bank when ACT removes requests.
    dim = values.shape[-1]
    flat = values.view(1, -1, dim)
    width = weights.shape[-1]
    queries = weights.numel() // width
    plan = read_plan(flat.shape[1], queries, dim, width)
    import torch
    # Frozen URM certifies operand version counters. Tensors allocated by
    # torch.inference_mode have no counters; create ordinary route tensors at
    # this package boundary without changing the caller's serving mode.
    with torch.inference_mode(False), torch.no_grad():
        if addresses.is_inference():
            addresses = addresses.clone()
        if weights.is_inference():
            weights = weights.clone()
        return plan.execute(memory=flat, read_addresses=addresses.view(1, queries, width),
                            read_weights=weights.view(1, queries, width))["readings"].view(*weights.shape[:-1], dim)


@lru_cache(maxsize=128)
def routed_read_plan(partitions, sequence, slots, dim, width, allow_large_route=False):
    """Compose the public route and read operations, including URM backward."""
    verify_dependency()
    from urm.compiler.pipeline import CompilationIntent, compile_graph
    from urm.ir.program import (DType, TensorHandle, SemanticProgram,
        SparseRouteGeneration, SparseRouteSelectionSpec, SparseStateMixerAccess,
        SparseStateMixerSpec, SparseStateOperation, SparseReadTiming, SparseStateExecutionMode)
    route = SparseRouteSelectionSpec(partitions, sequence, slots, width, DType.FLOAT32)
    read = SparseStateMixerSpec(partitions, sequence, slots, dim, 0, width,
        DType.FLOAT32, SparseStateOperation.READ_ONLY, SparseReadTiming.CURRENT_STATE,
        SparseStateExecutionMode.TRAINING)
    program = SemanticProgram.build(name="gl_sdm_routed_snapshot_read", inputs=(
        TensorHandle("scores", DType.FLOAT32, (partitions, sequence, route.score_width)),
        TensorHandle("memory", DType.FLOAT32, (partitions, slots, dim))), ops=(
        SparseRouteGeneration(name="router", inputs=("scores",),
            outputs=("read_addresses", "read_weights"), spec=route),
        SparseStateMixerAccess(name="snapshot", inputs=("read_addresses", "read_weights", "memory"),
            outputs=("readings", "updated_memory"), spec=read)),
        outputs=("readings", "read_addresses", "read_weights"))
    with large_route_override(allow_large_route):
        return compile_graph(program, target="native", intent=CompilationIntent.TRAINING)


def routed_snapshot_read(values, scores, width, allow_large_route=False):
    import torch
    B, H, S, D = values.shape
    T = scores.shape[1]
    plan = routed_read_plan(B * H, T, S, D, width, allow_large_route)
    grad_enabled = torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
        memory = values.reshape(B * H, S, D)
        scores = scores.permute(0, 2, 1, 3).contiguous().reshape(B * H, T, -1)
        # Equal signed zeros must have the same logical tie rank.
        scores = scores + 0.0
        result = plan.execute(memory=memory, scores=scores)
        return tuple(result[name].reshape(B, H, T, -1).transpose(1, 2)
                     for name in ("readings", "read_addresses", "read_weights"))
