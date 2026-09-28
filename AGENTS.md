# AGENTS.md

## Project

Biohub Cell Tracking During Development Kaggle research project.

## Primary Objective

Optimize the official Kaggle metric.

Priority:

1. adjusted edge Jaccard
2. node-count calibration
3. temporal association accuracy
4. globally consistent lineage reconstruction
5. division Jaccard

## Source Lock

The official competition repository is stored under `external/official/`.

Never silently modify official metric code.

Local metric implementations must be validated against the source-locked official implementation.

## Data

Do not commit competition data, model weights, caches, large predictions, or submissions.

Do not assume the full dataset fits in memory.

Use lazy Zarr access where practical.

## Geometry

Never assume z, y, and x have isotropic physical spacing.

Tracking and matching calculations must respect physical voxel scale.

## Architecture

Prefer modular implementations for:

- data loading
- detection
- centroid extraction
- node-count calibration
- candidate generation
- association
- motion estimation
- global optimization
- gap recovery
- division detection
- metric evaluation
- submission serialization

## Validation

Before claiming an improvement:

1. evaluate with the official metric
2. compare against the current baseline
3. record the exact configuration
4. report component metrics
5. separate reproducible gains from leaderboard noise

Track edge TP, FP, FN, raw edge Jaccard, adjusted edge Jaccard, node count ratio, division TP, FP, FN, division Jaccard, and final score.

## Engineering

Use Python 3.11.

Reusable code belongs under `src/biohub_cell_tracking/`.

Notebooks are for exploration, not production logic.

Run tests before committing.

## Competition Integrity

Do not intentionally exploit implementation bugs in the competition metric.

## Agent Delegation

Codex may handle bounded engineering tasks such as tests, refactors, adapters, metric utilities, experiment plumbing, and edge-case checks.

Research decisions involving metric interpretation, model strategy, or biological assumptions require explicit evidence before promotion into the main pipeline.
