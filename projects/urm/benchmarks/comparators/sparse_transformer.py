"""Load unchanged dense attention functions from OpenAI sparse_attention source."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from types import ModuleType


SOURCE_REVISION = "c53f3bdbf6225be0582f0357072e82b13c69be7d"
SOURCE_FILE = "attention.py"


def load_dense_attention(repository: Path | None = None):
    """AST-load upstream ``attention_impl`` without unavailable TF1/blocksparse imports.

    Function bodies are the pinned source text. The namespace supplies TensorFlow
    2's compatible v1 API and the two utility functions used by the source module.
    The custom ``BlocksparseTransformer`` path remains unavailable and is not loaded.
    """
    if repository is None:
        repository = Path("/tmp/urm-comparator-pins/sparse_transformer")
    repository = repository.resolve()
    path = repository / SOURCE_FILE
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != SOURCE_REVISION or dirty or not path.is_file():
        raise RuntimeError("the pinned OpenAI sparse_attention source checkout is unavailable or changed")

    import numpy as np
    import tensorflow as tensorflow

    np.__dict__.setdefault("bool", bool)

    source = path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(path))
    names = {
        "get_attn_mask",
        "split_states",
        "merge_states",
        "split_heads",
        "merge_heads",
        "attention_impl",
        "get_blocksparse_obj",
        "get_callback",
    }
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("the pinned attention.py dense callable set changed")

    def shape_list(tensor):
        shape = tensor.shape
        if shape.is_fully_defined():
            return shape.as_list()
        return tensorflow.shape(tensor).numpy().tolist()

    def recomputable(_name):
        return lambda function: function

    class _PinnedBlocksparsePattern:
        def __init__(self, layout, *, block_size, mask_callback, heads):
            self.layout = layout
            self.block_size = block_size
            self.mask_callback = mask_callback
            self.heads = heads

    namespace = {
        "np": np,
        "tf": tensorflow.compat.v1,
        "shape_list": shape_list,
        "recomputable": recomputable,
        "BlocksparseTransformer": _PinnedBlocksparsePattern,
    }
    module = ModuleType("urm_sparse_transformer_pinned_dense_attention")
    module.__file__ = str(path)
    module.__dict__.update(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    identity = {
        "repository": str(repository),
        "revision": revision,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "loaded_functions": sorted(names),
        "runtime_bridge": "TensorFlow 2.19 eager with tf.compat.v1 namespace; source function bodies unchanged; recomputable decorator is a no-op",
    }
    return module, identity


def fixed_mode_mask(module, *, n_ctx: int, heads: int, block_size: int, local_attn_ctx: int, num_verts: int, vertsize: int):
    """Expand the pinned source fixed-mode block layout and callback to a dense mask."""
    import numpy as np

    pattern = module.get_blocksparse_obj(
        n_ctx,
        heads,
        "fixed",
        blocksize=block_size,
        local_attn_ctx=local_attn_ctx,
        num_verts=num_verts,
        vertsize=vertsize,
    )
    layout = np.asarray(pattern.layout)
    if layout.ndim == 2:
        layout = np.broadcast_to(layout, (heads, *layout.shape))
    mask = np.zeros((heads, n_ctx, n_ctx), dtype=np.bool_)
    n_blocks = n_ctx // block_size
    block_index = 0
    for head in range(heads):
        for query_block in range(n_blocks):
            for key_block in range(n_blocks):
                if not layout[head, query_block, key_block]:
                    continue
                block_mask = pattern.mask_callback(
                    (block_size, block_size),
                    head,
                    query_block,
                    key_block,
                    block_index,
                )
                q_start, k_start = query_block * block_size, key_block * block_size
                mask[
                    head,
                    q_start : q_start + block_size,
                    k_start : k_start + block_size,
                ] = block_mask
                block_index += 1
    return mask
