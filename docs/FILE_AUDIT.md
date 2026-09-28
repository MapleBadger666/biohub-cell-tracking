# Local and GitHub file audit — September 28, 2026

Before packaging, all 70 files in the local Git index matched GitHub `main` at commit `fb51e46126c8461a44ec89c17adfd30550409d9c`, including each path and Git blob SHA. Local working file bytes also matched the index. There were no missing main source modules, scripts, tests, or split configurations.

The audit found additional source excluded by broad notebook and scratch-directory ignore rules. This update preserves:

- All 14 numbered exploratory notebooks, with identical cell sources and public execution outputs removed.
- The reviewed submission notebook and saved offline wheelhouse builder.
- Three earlier recovery draft notebooks, eight recovery Python files, and their dated audit records.
- Two distinct historical scripts, the Warm2 model architecture, source bundle manifest, verified wheelhouse hashes, Kaggle configuration, submission record, and offline environment report.

[The source archive manifest](evidence/source_archive_manifest_20260928.json) records original and published SHA256 values and confirms notebook cell-source equality. Notebook execution outputs, original cell metadata, counts, and attachments were preserved locally in `reports/notebook_execution_archive_20260928/` before stripping public copies. Python source copies are byte-identical.

Competition data, weights, predictions, submission CSVs, original validation results, and bulk reports remain local, as requested. They are excluded from the public repository. Core source files and the source-locked official checkout are unchanged. The two previously documented full-suite provenance failures have not been rewritten or hidden.

Local cleanup is limited to recognized caches, abandoned Git temporary packs, empty placeholders, and copies whose content has an independently verified retained destination. Git commit history and Codex checkpoint references are retained. An exact deletion record is stored locally under `reports/local_cleanup_20260928/`.
