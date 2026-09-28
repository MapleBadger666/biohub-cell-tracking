# Historical source archive

`submission_recovery_20260923/` preserves the earlier submission recovery audit, environment bootstrap, integrity checks, regression script, historical build/inference cells, and source-only draft notebooks. Its status statements refer to September 23, before the reviewed September 28 execution. Use the current [submission archive](../notebooks/submission/README.md) for the reviewed version and verified locks.

`legacy/` preserves the earlier Kaggle training script and the residual census script before G3H3E1. These differ from the current implementations and are retained for provenance. They are not promoted into the frozen submission pipeline.

No archive script is automatically imported by the package or selected by pytest. Historical local paths and dependencies remain visible so that the original research decisions can be inspected without claiming that every experiment is immediately reproducible.
