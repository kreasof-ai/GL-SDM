# Research archive

Historical material moved here during the compiler restructuring. Reports retain
their original contents, including claims subsequently found unsupported. They
are evidence of experiments, not normative specifications or current support.

- `docs/`: previous runtime/coverage proposals and experiment reports.
- `benchmarks/`: unvalidated dual-form performance and ten-step demonstrations.
- `results/`: early tuning sweeps and the historical dual-form audit.
- `project-status-2026-09-19.md`: previous project README.
- `manifest.json`: original paths, archive paths and SHA256 digests, plus source revision.

Files are retained byte-for-byte. Internal paths and commands in snapshots refer
to their original checkout; reproduce them at the manifest's source revision.
Do not run archived benchmarks as current acceptance gates. No accepted result is
made stronger by moving it here, and negative results remain part of the record.

Maintained acceptance harnesses, schemas, and evidence used by current regression
tests remain in `benchmarks/` and `results/`. Executable prototypes live under
`urm.experimental` and `urm.experiments`, with compatibility imports where needed.
Future artifacts should declare source revision, semantic contract, numerical
policy, backend/version, shape, gradient coverage, measurement boundary and status.

## Superseded construction references

The nested adapter, validation, kernel and runtime snapshots preserve earlier
integration freezes and milestone protocols. They were archived when active docs
were narrowed to full-version construction. Paths and hashes are in the manifest.
