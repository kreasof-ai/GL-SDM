# Shared Program Assets

This directory is for stable contracts and evaluation assets used by two or more
of the three research projects.

## Intended contents

- `interfaces/`: memory snapshot, read, write proposal, merge, commit,
  consolidate, validate, and rollback contracts.
- `benchmarks/`: controlled generators and shared language-model task adapters.
- `evaluation/`: resource matching, metrics, result schemas, and Pareto analysis.
- `testing/`: cross-project conformance and deterministic transaction tests.
- `claims/`: the living novelty matrix, claim ledger, and evidence references.

Create these subdirectories when their first concrete artifact is added. Until a
contract is shared and stable, keep its implementation inside the owning project.

## Shared experimental controls

- Match training FLOPs, active parameters, optimizer budget, data, and memory
  bandwidth whenever the comparison permits.
- Report total memory capacity, active retrieved parameters, and mutable-state
  bytes separately.
- Report mean and tail recurrent depth and latency.
- Separate transient context recall from durable cross-session learning.
- Track routing stability, collisions, memory utilization, and multiple seeds.
- Preserve negative scaling results and resource-normalized Pareto curves.
