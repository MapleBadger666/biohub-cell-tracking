# Reviewed Kaggle execution archive

- `frozen_v9_submission.ipynb`: cell sources used for the reviewed Version 2 offline inference.
- `build_offline_wheelhouse.ipynb`: saved wheelhouse builder source for Python 3.12/Linux x86_64.
- `verified_wheelhouse_lock.json`: verified wheel ZIP and manifest SHA256 values.
- `source_bundle_manifest.json`: exact 29-file source/input bundle manifest. It references a checkpoint and sample CSV; those files are not included here.
- `kernel-metadata.json`: historical execution configuration and Kaggle input references. It is an archive, not an instruction to publish or submit another version.
- `review_status.json`: dated operational review, local official baseline reproduction, and accepted competition submission record.

Notebook outputs are stripped; code and markdown cell sources are unchanged. Some original cell comments say `STATIC REVIEW` or `NOT RUN` because they were written before the successful saved run. Those comments are historical. The dated review record and [validation notes](../../docs/VALIDATION.md) describe the subsequent completed execution. No leaderboard score is asserted.

This archive requires the separately obtained source bundle, checkpoint, competition data, and offline wheelhouse. It does not distribute weights, prediction CSVs, or binary wheels. Kaggle input references may require account access. Merely opening these notebooks locally does not recreate the Kaggle environment.
