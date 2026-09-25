"""URM model-agnostic training harness (ATMA pattern).

Replaces the SDM-tied ``train/loop.py`` runner with a generic trainer that serves any
registered public-path mixer (K1 dense/K2 linear-state/K3 sparse-state) through one
AdamW+Muon loop with MFU accounting, checkpointing, and the three correctness gates
(checkpoint alignment, gradient alignment, KL divergence vs upstream).

Modules:
- ``data`` — finewebedu shard loader + a self-contained synthetic stream.
- ``model`` — the pluggable-mixer ``URMDecoderLM``.
- ``registry`` — the architecture→upstream map with honest capability flags.
- ``optimizer`` — the AdamW+Muon split with a coverage assertion.
- ``harness`` — the model-agnostic runner and the correctness gates.
- ``run`` — the CLI driver (``python -m train.run --mixer gla --steps 10``).

The SDM-specific ``loop.py`` is retained for the SDM comparison evidence; new training
runs go through this package.
"""
