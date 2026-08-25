# Project II: Consolidated SDM

## Research question

Can sparse memory overlays support rapid deployment-time learning while offline
consolidation preserves useful knowledge, limits interference, and provides
rollback and provenance?

## Dependency

CSDM depends on the address-space and snapshot/commit semantics established by
GL-SDM. It should consume that contract rather than redefine the core operator.

## This project owns

- Aligned pretrained, slow, and fast memory tiers.
- Wake-time sparse writes and their governance metadata.
- Sleep-time promotion, merge, rejection, quarantine, and capacity reclamation.
- Stability controls such as novelty filters, preconditioning, write budgets,
  collision monitoring, and replay mixtures.
- Versioning, provenance, validation, deletion, and rollback behavior.
- Long-horizon continual-learning and deployment simulations.

## First milestone

On top of a stable GL-SDM reference implementation, define fast/slow storage
budgets and promotion thresholds, then run controlled streams containing concept
recurrence, collisions, contradictions, and distribution drift.

## Acceptance gate

Proceed when aligned overlays plus validated consolidation improve the
retention-plasticity frontier under fixed storage and sleep-compute budgets.
Treat behavior equivalent to a growing context cache, replay costs comparable to
ordinary retraining, unreachable old memories, or incomplete rollback as
negative results.

## Expected output

A continual-learning study and a governed memory lifecycle supporting reversible
wake/sleep operation.
