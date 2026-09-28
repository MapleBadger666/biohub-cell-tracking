# Historical source archive

`submission_recovery_20260923/` preserves the earlier submission recovery audit, environment bootstrap, integrity checks, regression script, historical build/inference cells, and source-only draft notebooks. Its status statements refer to September 23, before the reviewed September 28 execution. Use the current [submission archive](../notebooks/submission/README.md) for the reviewed version and verified locks.

`legacy/` preserves seven distinct earlier scripts: Kaggle training, training before sparse-label/domain-aware/division-positive changes, residual census before G3H3E1, bridge association before B12 coverage, and iterative reciprocal recovery before 12.09. These differ from the current implementations and are retained for provenance. They are not promoted into the frozen submission pipeline. Backups identical to an existing published file are mapped to that file in the source archive manifest instead of uploaded twice.

No archive script is automatically imported by the package or selected by pytest. Historical local paths and dependencies remain visible so that the original research decisions can be inspected without claiming that every experiment is immediately reproducible.
