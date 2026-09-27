"""Shared, explicit memory and execution policy for both benchmark arms."""
from functools import lru_cache
import hashlib
from pathlib import Path
MEASUREMENT_VERSION = 2
CHECKPOINT_ROWS = frozenset({
    "conformer_attention", "forgetting_attention", "gated_delta_product",
    "gdn2", "log_linear_attention", "log_linear_mamba2", "path_attention", "tucker_attention",
})


def use_checkpointing(row):
    return row in CHECKPOINT_ROWS


@lru_cache(maxsize=1)
def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("architectures", "train", "src/urm", "extra/comparators"):
        for path in sorted((root / folder).rglob("*.py")):
            if path.name == "report.py":
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def valid_cached_result(record, expected_config):
    return (record.get("measurement_version") == MEASUREMENT_VERSION
            and "error" not in record
            and record.get("source_fingerprint") == source_fingerprint()
            and record.get("config") == expected_config)
