# Research design

## Scientific problem

Microscopy tracking must identify cells, link detections between frames, preserve division events, and avoid biologically implausible merges. Sparse annotations make raw detection counts and naive edge accuracy insufficient summaries of quality.

## Evaluation and geometry

The principal optimization target is the official adjusted edge Jaccard. Node-count calibration, temporal association, lineage consistency, and division Jaccard are tracked alongside it. Ground truth is used for training and evaluation, not as a runtime input to the prediction-only submission pipeline.

The five-fold manifest is locked at dataset level; frames from a dataset are not randomly split across folds. Split generation balances prefix, density, and division information. Physical z/y/x scale is explicit because microscopy voxels are anisotropic.

## Frozen V9 graph reconstruction

1. Filter predicted divisions using edge confidence and daughter separation, angle, parent distance, and distance balance.
2. Prune low-confidence nodes only within small components using the second detection threshold.
3. Recover single missing links using temporal and physical geometry, mutual candidate selection, and ambiguity checks.
4. Recover strict two-edge bridges through isolated middle detections, with endpoint and degree-conflict rejection.
5. Apply frozen component cleanup, then evaluate with the locked official metric.

The order and thresholds are part of the experiment identity. Candidate tables are regenerated for the current prediction node IDs; historical prediction-specific node IDs are not reused. Frozen V9 intentionally excludes ILP, motion correction, synthetic node insertion, and prefix-specific graph policies.

## Engineering contribution and boundaries

Project-specific work includes sparse-label training adaptations, device support, physical-scale detection pooling, frozen post-processing, graph audits, experiment plumbing, and reproducibility checks. The temporal UNet, transformer baseline, evaluator, and much of the I/O are upstream components.

Stage11 and Stage12 explore residual association, divisions, and portable source-derived providers. Their presence in the repository does not establish a reproducible gain over Frozen V9. Some retain local artifact dependencies and historical source hashes; unresolved contracts are disclosed in VALIDATION.md.

## Current limitations

- Division Jaccard is low, despite substantially higher edge metrics.
- Local validation does not establish hidden-test generalization or leaderboard standing.
- Full model reproduction needs separate data and weights.
- Some historical experiments contain fixed local paths and require additional migration before portable execution.
- Two full-suite source-hash checks currently fail; no metric or scientific source was changed to suppress those failures.
