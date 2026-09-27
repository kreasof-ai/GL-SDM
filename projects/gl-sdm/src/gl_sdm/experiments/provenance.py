"""Runtime, source and kernel provenance for experiment artifacts."""
import hashlib
import os
from pathlib import Path
from gl_sdm.baselines.upstream import PINS, source, sdm_layer


def metadata(arch, cfg=None):
    import torch
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "reference_fallback": False}
    info["residual_dtype"] = (cfg or {}).get("residual_dtype", (cfg or {}).get("dtype"))
    info["cuda_allocator_config"] = os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"))
    if arch == "gl_sdm":
        backend = (cfg or {}).get("gl_memory_backend", "torch")
        info.update(implementation="GL-SDM transaction kernels + frozen URM" if backend == "urm" else "project-owned PyTorch GL-SDM",
                    memory_backend=backend, transaction="one token", state_dtype="float32", optimized_kernel=backend == "urm")
        chunk = (cfg or {}).get("gl_chunk_size", 1)
        info["dense_compilation"] = (cfg or {}).get("gl_compile", False)
        info["training_cuda_graph"] = (cfg or {}).get("gl_cuda_graph", False)
        if chunk > 1:
            info.update(transaction=f"one commit per {chunk} tokens", local_context="causal SDPA within chunk",
                        write_mass="per-token reasoning mass divided by chunk size", routing="highest-address ties")
        if cfg and "gl_layer_pattern" in cfg:
            info.update(layer_pattern=cfg["gl_layer_pattern"], physical_layers=cfg["num_hidden_layers"],
                        weight_loops=0, shared_memory_banks=1,
                        local_context=f"rolling causal SDPA, window {cfg.get('gl_local_window', 128)} including current token",
                        write_mass="sum token/layer deltas; no averaging",
                        route_projection="model-dtype GEMM, FP32 scores for URM top-k",
                        inference_read_value_dim=cfg["head_dim"],
                        urm_large_route_override=cfg.get("gl_urm_large_route_override", False))
        digest = hashlib.sha256()
        source_root = Path(__file__).resolve().parents[1]
        for path in sorted(source_root.rglob("*.py")):
            digest.update(path.relative_to(source_root).as_posix().encode())
            digest.update(path.read_bytes())
        info["source_sha256"] = digest.hexdigest()
        if backend == "urm":
            from gl_sdm.memory.backends.urm import verify_dependency, read_plan, routed_read_plan
            info["urm"] = dict(verify_dependency())
            if cfg and "hidden_size" in cfg:
                heads = cfg["hidden_size"] // cfg["head_dim"]
                queries = cfg["mbs"] * heads
                if chunk > 1:
                    physical_dim = 128 if cfg["head_dim"] == 64 else cfg["head_dim"]
                    info["logical_value_dim"] = cfg["head_dim"]
                    info["physical_read_value_dim"] = physical_dim
                    info["initial_batch_read_plan"] = routed_read_plan(queries, min(chunk, cfg["seq_len"]),
                        cfg.get("gl_slots", 1024), physical_dim, cfg.get("gl_reads", 8),
                        cfg.get("gl_urm_large_route_override", False)).serialized_plan()
                    if cfg.get("gl_urm_large_route_override", False):
                        info["urm"]["experimental_support_override"] = {"factor_extent": 512, "route_width": 8,
                            "score_dtype": "float32", "index_dtype": "int32", "scope": "compiler support declaration only"}
                else:
                    info["initial_batch_read_plan"] = read_plan(queries * cfg.get("gl_slots", 1024), queries,
                        cfg["head_dim"], cfg.get("gl_reads", 8)).serialized_plan()
            info["routing"] = "URM product-key top-k; highest address wins ties" if chunk > 1 else "GL-SDM stable product-key top-k; smaller address wins ties"
            info["commit"] = "stable ordered sum per address; no forward atomics"
            info["backward_storage"] = "URM saves one aliased snapshot per chunk; selected write rows" if chunk > 1 else "selected rows; full bank versions are not saved"
    if arch in {"sdm", "gdn2"}:
        name = "sdm" if arch == "sdm" else "fla"
        info.update(repository=PINS[name][0], revision=PINS[name][1], source=str(source(name)))
    if arch == "sdm":
        info["cuda_extensions"] = sdm_layer()[1]
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
    return info
