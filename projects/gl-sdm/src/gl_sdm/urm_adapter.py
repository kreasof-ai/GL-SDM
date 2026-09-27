"""Compile snapshot reads against the frozen, installed URM package."""
from functools import lru_cache
from importlib.metadata import distribution
import json

URM_REVISION = "604bfdf5d2c827266a32ef142ca996cc712d70f0"
URM_REPOSITORY = "https://github.com/kreasof-ai/urm.git"


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
