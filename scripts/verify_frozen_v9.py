#!/usr/bin/env python
"""Read-only Warm2 Fold0 Frozen V9 reproduction gate.

Runs the migrated :mod:`biohub_cell_tracking.frozen_v9` pipeline against the
existing local Warm2 predictions and GT, and checks the locked Stage10 result.

This script does not train, does not run inference, does not write prediction
GEFFs, and does not cache candidate tables. It exits nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from biohub_cell_tracking.frozen_v9 import (  # noqa: E402 - path bootstrap must precede import
    FROZEN_V9_CONFIG,
    evaluate_frozen_v9,
    summarize_by_prefix,
)

WARM2_995 = "predictions/heqiuyan/official_baseline_fold0_warm2_lr1e5_det0995/split_0"
WARM2_9975 = "predictions/heqiuyan/official_baseline_fold0_warm2_lr1e5_det09975/split_0"
GT_DIR = "data/raw/competition/biohub-cell-tracking-during-development/train"

# Locked Stage10 Warm2 Fold0 Frozen V9 result.
EXPECTED_CANDIDATES = {
    "single_opportunities": 253757,
    "single_policy": 3945,
    "isolated_nodes": 70018,
    "isolated_with_local_pair": 22085,
    "strict_two_policy": 2289,
}

EXPECTED_INTERVENTIONS = {
    "single_added": 3945,
    "two_requested": 2289,
    "two_added": 2111,
    "associations_added": 4222,
    "two_missing_endpoint": 0,
    "two_degree_conflict": 35,
    "two_existing_edge_conflict": 143,
    "cleanup_removed": 80922,
}

EXPECTED_METRICS = {
    "edge_jaccard": 0.774393,
    "adj_edge_jaccard": 0.767075,
    "division_jaccard": 0.038462,
    "node_recall": 0.956612,
    "score": 0.770921,
}

EXPECTED_PREFIX = {
    "44b6": {
        "n": 15,
        "edge_jaccard": 0.774347,
        "adj_edge_jaccard": 0.767283,
        "division_jaccard": 0.000000,
        "node_recall": 0.962260,
        "score": 0.767283,
    },
    "6bba": {
        "n": 25,
        "edge_jaccard": 0.774404,
        "adj_edge_jaccard": 0.767026,
        "division_jaccard": 0.046512,
        "node_recall": 0.953224,
        "score": 0.771677,
    },
}

# The recorded values are rounded to 6 decimals and the previous exact
# reproduction differed by only +8.2e-8, so 5e-7 is strict but achievable.
METRIC_TOLERANCE = 5e-7


def check_int(label: str, actual: int, expected: int, failures: list[str]) -> str:
    ok = int(actual) == int(expected)
    if not ok:
        failures.append(f"{label}: expected {expected}, got {actual}")
    return f"  {label:<28} {actual:>10}  expected {expected:>10}  {'OK' if ok else 'FAIL'}"


def check_float(label: str, actual: float, expected: float, failures: list[str]) -> str:
    delta = abs(float(actual) - float(expected))
    ok = delta <= METRIC_TOLERANCE
    if not ok:
        failures.append(f"{label}: expected {expected:.6f}, got {actual:.9f} (delta {delta:.3g})")
    return f"  {label:<28} {actual:>12.9f}  expected {expected:>10.6f}  delta {delta:>9.2e}  {'OK' if ok else 'FAIL'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--progress", action="store_true", help="print per-dataset progress")
    args = parser.parse_args()

    project = args.project.resolve()

    pred_995 = project / WARM2_995
    pred_9975 = project / WARM2_9975
    gt_dir = project / GT_DIR
    manifest = project / "configs/cv_manifest.csv"

    for path in (pred_995, pred_9975, gt_dir, manifest):
        if not path.exists():
            print(f"FAIL: missing required input: {path}")
            return 2

    cv_meta = pd.read_csv(manifest)
    fold0 = cv_meta[cv_meta["fold"] == 0].copy()

    if len(fold0) != 40:
        print(f"FAIL: expected 40 Fold0 datasets, found {len(fold0)}")
        return 2

    dataset_names = sorted(fold0["dataset"])
    prefix_map = dict(zip(fold0["dataset"], fold0["prefix"], strict=True))
    estimated_nodes = dict(zip(fold0["dataset"], fold0["estimated_nodes"], strict=True))

    print("Warm2 Fold0 Frozen V9 reproduction")
    print(f"  datasets    {len(dataset_names)}")
    print(f"  det=.995    {pred_995.relative_to(project)}")
    print(f"  det=.9975   {pred_9975.relative_to(project)}")
    print(
        f"  config      K={FROZEN_V9_CONFIG.confidence_component_limit} "
        f"p2>={FROZEN_V9_CONFIG.division_p2_min} "
        f"residual<={FROZEN_V9_CONFIG.recovery_residual_max_um}um"
    )
    print()

    started = time.time()

    evaluation = evaluate_frozen_v9(
        dataset_names,
        pred_995,
        pred_9975,
        gt_dir,
        estimated_nodes,
        prefix_map=prefix_map,
        progress=args.progress,
    )

    elapsed = time.time() - started
    failures: list[str] = []

    print(f"Candidate counts  ({elapsed / 60:.1f} min total)")
    counts = evaluation.policies.counts()
    for key, expected in EXPECTED_CANDIDATES.items():
        print(check_int(key, counts[key], expected, failures))

    print()
    print("Intervention counts")
    for key, expected in EXPECTED_INTERVENTIONS.items():
        print(check_int(key, evaluation.interventions[key], expected, failures))

    print()
    print("Overall metrics")
    for key, expected in EXPECTED_METRICS.items():
        print(check_float(key, evaluation.summary[key], expected, failures))

    print()
    print("Prefix metrics")
    prefix_df = summarize_by_prefix(evaluation, prefix_map=prefix_map)

    for prefix, expected in EXPECTED_PREFIX.items():
        rows = prefix_df[prefix_df["prefix"] == prefix]
        if rows.empty:
            failures.append(f"{prefix}: missing from prefix summary")
            print(f"  {prefix}: MISSING")
            continue

        row = rows.iloc[0]
        print(f"  {prefix} (n={int(row['n'])})")
        check_int(f"{prefix} n", int(row["n"]), expected["n"], failures)

        for key in ("edge_jaccard", "adj_edge_jaccard", "division_jaccard", "node_recall", "score"):
            print(check_float(f"{prefix} {key}", row[key], expected[key], failures))

    print()

    if failures:
        print(f"FAIL — {len(failures)} mismatch(es):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PASS — Warm2 Fold0 Frozen V9 reproduction matches the locked result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
