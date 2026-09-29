# Validation evidence — September 28–29, 2026

## Official baseline reproduction

`scripts/verify_frozen_v9.py` ran on 40 Fold0 datasets against the source-locked official evaluator. The recorded final score is 0.770921082, adjusted edge Jaccard 0.767074928, raw edge Jaccard 0.774393229, division Jaccard 0.038461538, and node recall 0.956612481.

The gate also checked exact candidate and intervention counts. It recovered 3,945 single edges and added 2,111 two-edge bridges (4,222 associations), rejecting 35 degree conflicts and 143 existing-edge conflicts. The two-edge policy requested 2,289 bridges. All locked reproduction comparisons passed.

The [transcript](evidence/frozen_v9_reproduction.txt) is retained as evidence. This reproduction confirms an existing baseline; it is not evidence for a new improvement. TP/FP/FN and node-count ratio are not included in this transcript and are not inferred from rounded aggregate scores.

## Tests

- Core Frozen V9 suite: **40 passed**; synthetic in-memory graphs, no competition data or weights.
- Full project suite before portfolio packaging: **412 passed, 2 failed** in Python 3.11.
- `test_frozen_training_and_predictor_scripts_unchanged`: training source SHA differs from its historical expected hash (`c6cd…cbc1` expected, `3e9d…06ac` observed).
- `test_portable_post_repair_sha`: portable Stage12 source SHA differs from its historical expected hash (`a1fe…36ed` expected, `b0b5…06e0` observed).

The failures indicate unresolved provenance contracts. Expected hashes have not been rewritten and the tests have not been skipped or marked as passing. The one portfolio portability edit uses the test file's location instead of a developer-specific absolute root in `test_stage12_v2_invented_sort_repair.py`; it does not alter a research algorithm or expected source hash.

The cleanup audit recovered the earlier [training source before division-positive sampling](../research/legacy/train_unet_transformer_mps_before_divpos.py), whose SHA256 is exactly the historical training expectation `c6cd…cbc1`. It is retained as an archive, without replacing the current script or changing the failing test's contract.

## Reviewed Kaggle execution

The reviewed Frozen V9 Notebook completed offline GPU execution, both detection thresholds, frozen graph reconstruction, structural CSV validation, and official CSV/GEFF roundtrip checks. Its saved output had 282,579 rows. The prediction file is deliberately excluded from GitHub.

Kaggle identity verification cleared the submission restriction, and the authenticated competition page accepted Version 2 on September 28, 2026. On September 29, the [submission page](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/submissions) showed **Succeeded** and a **public score of 0.823**, displayed to three decimal places. The [dated result record](evidence/kaggle_public_result_20260929.json) links the exact notebook version. A final private score, rank, medal, or prize is not claimed. The public score is on competition evaluation data, whereas 0.770921082 is the separately reproduced local Fold0 baseline; they are not a before/after experiment. The saved execution and local checks establish submission validity, not a new measured improvement over another Kaggle submission. The [September 28 review snapshot](../notebooks/submission/review_status.json) preserves the earlier rejected API attempt and accepted web submission before scoring finished.

## Reproducibility scope

The repository includes source and the `uv.lock` dependency resolution. The official evaluator is independently fetched at a fixed commit. Kaggle used a separately reviewed offline wheel overlay; its builder, lock, and environment report are now archived, but the generic local setup is not represented as bit-for-bit identical to that runtime. Archived notebook outputs, raw annotations, models, and prediction GEFFs are not distributed.
