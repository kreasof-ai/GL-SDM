# GL-SDM and CSDM workspace

This repository owns the GL-SDM architecture and transactional memory operator,
the CSDM lifecycle, and shared experimental contracts. URM is an external frozen
dependency maintained at https://github.com/kreasof-ai/urm.

## URM boundary

- Use the exact commit in `shared/requirements-urm.txt` for both projects.
- Do not vendor URM source or edit a sibling URM checkout during GL-SDM/CSDM work.
- Implement project-owned snapshot/commit behavior, overlays and experimental
  kernels in the owning project. Keep the GL-SDM memory contract reusable by CSDM.
- URM changes and dependency-pin updates require an explicitly scoped URM task.
  A dependency update must identify its reason and verify the dependent workloads.

Read each project's README before changing its models or experiments.
