"""Verified algebraic rewrites.

Owns the typed rewrite rules, the rewrite engine, and the rewrite proof
obligations (forward/backward correctness, numerical tolerances, effect
safety). See :mod:`urm.compiler.rewrite.rules`,
:mod:`urm.compiler.rewrite.engine`, and :mod:`urm.compiler.rewrite.proof`.
"""

from urm.compiler.rewrite.engine import (
    RewriteEngine,
    RewriteResult,
    RewriteTrace,
    RuleAttempt,
)
from urm.compiler.rewrite.proof import (
    BackwardContract,
    BackwardStrategy,
    EquivalenceClass,
    ForwardOnlyRestriction,
    Obligation,
    SavedStatePolicy,
)
from urm.compiler.rewrite.rules import (
    BARRIER_FREE,
    CheckOutcome,
    DEFAULT_RULES,
    DELAY_ROW_SCALE_THROUGH_GEMM,
    FOLD_ROW_SCALE_EPILOGUE,
    Precondition,
    RewriteMatch,
    RewriteRule,
    SCALE_IS_ROWWISE_LINEAR,
    SINGLE_CONSUMER,
)

__all__ = [
    "BARRIER_FREE",
    "BackwardContract",
    "BackwardStrategy",
    "CheckOutcome",
    "DEFAULT_RULES",
    "DELAY_ROW_SCALE_THROUGH_GEMM",
    "EquivalenceClass",
    "FOLD_ROW_SCALE_EPILOGUE",
    "ForwardOnlyRestriction",
    "Obligation",
    "Precondition",
    "RewriteEngine",
    "RewriteMatch",
    "RewriteResult",
    "RewriteRule",
    "RewriteTrace",
    "RuleAttempt",
    "SCALE_IS_ROWWISE_LINEAR",
    "SINGLE_CONSUMER",
    "SavedStatePolicy",
]
