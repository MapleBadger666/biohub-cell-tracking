#!/usr/bin/env python
"""Stage 11.01 — read-only Fold0 division error decomposition audit.

Reconstructs the current Stage10 best system (dual-checkpoint domain routing:
44b6 -> Sparse9995, 6bba -> Baseline256), applies the frozen V9 pipeline, checks
the recorded Fold0 score, and only then decomposes every GT division and every
predicted division.

This script does not train, does not run inference, does not write a prediction
GEFF, and does not touch GT. It exits nonzero if the reproduction gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from tracking_cellmot.metrics import summarise  # noqa: E402 - path bootstrap must precede import

from biohub_cell_tracking.division_audit import (  # noqa: E402 - path bootstrap must precede import
    DIVISION_TP,
    NODES_PRESENT_EDGES_MISSING,
    DivisionAudit,
    assert_candidates_match_g2,
    audit_dataset,
    fn_bottleneck_summary,
    geometry_distributions,
    staged_frozen_v9,
    summarize_classifications,
    summarize_fp_structure,
    summarize_gt_by_prefix,
    summarize_pred_fork_scope,
)
from biohub_cell_tracking.frozen_v9 import (  # noqa: E402 - path bootstrap must precede import
    FROZEN_V9_CONFIG,
    build_frozen_v9_policies,
    evaluate_graph,
    load_graph,
)

GT_DIR = "data/raw/competition/biohub-cell-tracking-during-development/train"

# Prediction sets, discovered from notebooks/10_sparse_label_training.ipynb
# (cell 10.16 VARIANTS_1016 and cell 10.18/10.19 PRED_ROOT_1019).
VARIANTS = {
    "BASELINE_256": {
        "pred_995": "predictions/heqiuyan/stage10_infer_baseline_256_det995/split_0",
        "pred_9975": "predictions/heqiuyan/stage10_infer_baseline_256_det9975/split_0",
    },
    "SPARSE_9995_256": {
        "pred_995": "predictions/heqiuyan/stage10_infer_sparse9995_256_det995/split_0",
        "pred_9975": "predictions/heqiuyan/stage10_infer_sparse9995_256_det9975/split_0",
    },
}

# Dual-checkpoint domain routing (notebook cell 10.20 DOMAIN_HYBRID_44b6_SPARSE).
ROUTING = {
    "44b6": "SPARSE_9995_256",
    "6bba": "BASELINE_256",
}

# Recorded Fold0 result of the routed hybrid (notebook 10.20 output row
# DOMAIN_HYBRID_44b6_SPARSE; 6 decimals is the full recorded precision).
EXPECTED_HYBRID = {
    "edge_jaccard": 0.781989,
    "adj_edge_jaccard": 0.774278,
    "division_jaccard": 0.031746,
    "node_recall": 0.959822,
    "score": 0.777453,
}

METRIC_TOLERANCE = 5e-7


def check_float(label: str, actual: float, expected: float, failures: list[str]) -> str:
    delta = abs(float(actual) - float(expected))
    ok = delta <= METRIC_TOLERANCE
    if not ok:
        failures.append(f"{label}: expected {expected:.6f}, got {actual:.9f} (delta {delta:.3g})")
    return f"  {label:<20} {actual:>12.9f}  expected {expected:>10.6f}  delta {delta:>9.2e}  {'OK' if ok else 'FAIL'}"


def run(project: Path, progress: bool) -> tuple[int, DivisionAudit | None]:
    gt_dir = project / GT_DIR
    manifest = project / "configs/cv_manifest.csv"

    for path in (gt_dir, manifest):
        if not path.exists():
            print(f"FAIL: missing required input: {path}")
            return 2, None

    variant_dirs = {label: {key: project / value for key, value in cfg.items()} for label, cfg in VARIANTS.items()}

    for label, cfg in variant_dirs.items():
        for key, path in cfg.items():
            if not path.exists():
                print(f"FAIL: {label} {key}: missing prediction directory {path}")
                return 2, None
            n_geff = len(sorted(path.glob("*.geff")))
            if n_geff != 40:
                print(f"FAIL: {label} {key}: {n_geff} GEFFs, expected 40")
                return 2, None

    cv_meta = pd.read_csv(manifest)
    fold0 = cv_meta[cv_meta["fold"] == 0].copy()

    if len(fold0) != 40:
        print(f"FAIL: expected 40 Fold0 datasets, found {len(fold0)}")
        return 2, None

    dataset_names = sorted(fold0["dataset"])
    prefix_map = dict(zip(fold0["dataset"], fold0["prefix"], strict=True))
    estimated_nodes = dict(zip(fold0["dataset"], fold0["estimated_nodes"], strict=True))

    unknown = sorted({prefix_map[name] for name in dataset_names} - set(ROUTING))
    if unknown:
        print(f"FAIL: unrouted prefix(es): {unknown}")
        return 2, None

    print("Stage 11.01 — Fold0 division error decomposition audit")
    print(f"  datasets   {len(dataset_names)}")
    for prefix, variant in ROUTING.items():
        n = sum(1 for name in dataset_names if prefix_map[name] == prefix)
        print(f"  {prefix} (n={n:2d}) -> {variant}")
        print(f"      det=.995  {VARIANTS[variant]['pred_995']}")
        print(f"      det=.9975 {VARIANTS[variant]['pred_9975']}")
    print(
        f"  frozen G2  p2>={FROZEN_V9_CONFIG.division_p2_min} "
        f"sep>={FROZEN_V9_CONFIG.division_daughter_sep_min_um}um "
        f"angle>={FROZEN_V9_CONFIG.division_angle_min_deg}deg "
        f"maxd<={FROZEN_V9_CONFIG.division_max_parent_daughter_um}um "
        f"balance<={FROZEN_V9_CONFIG.division_distance_balance_max}"
    )
    print()

    started = time.time()

    # Policies are regenerated per prediction set over all 40 Fold0 datasets,
    # exactly as notebook cells 10.16 / 10.19 did before 10.20 routed the
    # already-computed per-dataset records.
    policies = {}
    for label, cfg in variant_dirs.items():
        print(f"Generating candidates: {label}", flush=True)
        policies[label] = build_frozen_v9_policies(
            dataset_names,
            cfg["pred_995"],
            cfg["pred_9975"],
            gt_dir,
            prefix_map=prefix_map,
            progress=progress,
        )
        print(f"  {label} policy counts: {policies[label].counts()}", flush=True)

    print()
    print("Applying frozen V9 under domain routing", flush=True)

    records = []
    per_dataset_rows = []
    gt_rows: list[dict] = []
    pred_rows: list[dict] = []

    for i, name in enumerate(dataset_names, start=1):
        prefix = prefix_map[name]
        variant = ROUTING[prefix]
        cfg = variant_dirs[variant]

        staged = staged_frozen_v9(
            name,
            cfg["pred_995"],
            cfg["pred_9975"],
            policies[variant],
            gt_dir,
        )

        assert_candidates_match_g2(staged)

        gt_graph = load_graph(gt_dir / f"{name}.geff")

        metrics = evaluate_graph(staged.graph, gt_graph, staged.scale, estimated_nodes[name])
        records.append(metrics)

        audit = audit_dataset(staged, gt_graph, prefix=prefix)
        gt_rows.extend(audit.gt_rows)
        pred_rows.extend(audit.pred_rows)

        if (
            audit.division_tp != metrics["division_tp"]
            or audit.division_fp != metrics["division_fp"]
            or audit.division_fn != metrics["division_fn"]
        ):
            print(f"FAIL: {name}: audit division counts disagree with the locked metric")
            return 1, None

        per_dataset_rows.append(
            {
                "dataset": name,
                "prefix": prefix,
                "variant": variant,
                "edge_jaccard": float(metrics["edge_jaccard"]),
                "adj_edge_jaccard": float(metrics["adj_edge_jaccard"]),
                "node_recall": float(metrics["node_recall"]),
                "division_tp": int(metrics["division_tp"]),
                "division_fp": int(metrics["division_fp"]),
                "division_fn": int(metrics["division_fn"]),
                "gt_divisions": len(audit.gt_rows),
                "pred_forks": len(audit.pred_rows),
                **staged.application.to_row(),
            }
        )

        if progress and (i == 1 or i % 10 == 0 or i == len(dataset_names)):
            print(f"  evaluated {i:02d}/{len(dataset_names)}", flush=True)

    elapsed = time.time() - started
    summary = summarise(records)

    print()
    print(f"Reproduction gate  ({elapsed / 60:.1f} min)")
    failures: list[str] = []
    for key, expected in EXPECTED_HYBRID.items():
        print(check_float(key, summary[key], expected, failures))

    if failures:
        print()
        print(f"STOP — hybrid reproduction FAILED ({len(failures)} mismatch(es)):")
        for failure in failures:
            print(f"  - {failure}")
        print("Division diagnostics from a non-equivalent pipeline are not interpretable.")
        return 1, None

    print()
    print("PASS — dual-checkpoint domain-routed hybrid reproduces the recorded Fold0 result.")

    audit = DivisionAudit(
        gt_divisions=pd.DataFrame(gt_rows),
        pred_divisions=pd.DataFrame(pred_rows),
        per_dataset=pd.DataFrame(per_dataset_rows),
        metric_summary={k: float(v) for k, v in summary.items()},
    )

    return 0, audit


def report(audit: DivisionAudit, out_dir: Path) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    gt = audit.gt_divisions
    pred = audit.pred_divisions

    print()
    print("=" * 72)
    print("A. GT divisions")
    print("=" * 72)
    print(f"  total GT divisions in Fold0 : {len(gt)}")
    print(f"  division TP / FP / FN       : {audit.division_tp} / {audit.division_fp} / {audit.division_fn}")
    print(f"  division Jaccard            : {audit.metric_summary['division_jaccard']:.6f}")

    print()
    print("C. Failure decomposition (all Fold0)")
    overall = summarize_classifications(gt)
    print(overall.to_string(index=False))

    print()
    print("D. Failure decomposition by prefix")
    by_prefix = summarize_gt_by_prefix(gt)
    for prefix in sorted(gt["prefix"].unique()):
        block = by_prefix[by_prefix["prefix"] == prefix]
        n = int((gt["prefix"] == prefix).sum())
        print(f"  {prefix} (GT divisions = {n})")
        print(block.drop(columns=["prefix"]).to_string(index=False))
        print()

    print("Association gaps inside NODES_PRESENT_EDGES_MISSING")
    gap = gt[gt["classification"] == NODES_PRESENT_EDGES_MISSING]
    if len(gap):
        gap_counts = (
            gap.groupby("association_gap_kind")
            .agg(n=("gt_parent", "size"), raw_parent_out_degree_max=("raw_parent_out_degree", "max"))
            .reset_index()
        )
        print(gap_counts.to_string(index=False))
    else:
        print("  (none)")

    print()
    bottleneck = fn_bottleneck_summary(gt)
    print("FN bottleneck summary")
    for key in (
        "n_fn",
        "parent_missing",
        "daughter_missing",
        "all_nodes_present",
        "no_candidate_generated",
        "rejected_confidence_only",
        "rejected_geometry_only",
        "rejected_confidence_and_geometry",
        "accepted_then_lost",
        "detection_limited",
        "linking_limited",
    ):
        pct_key = f"pct_{key}" if key != "n_fn" else None
        pct = f"  ({bottleneck[pct_key]:5.1f}% of FN)" if pct_key and pct_key in bottleneck else ""
        print(f"  {key:<26} {bottleneck[key]:>4}{pct}")
    print(f"  largest_group              {bottleneck['largest_group']}")

    print()
    print("Geometry distributions by classification (candidate rows only)")
    dist = geometry_distributions(gt)
    if len(dist):
        print(dist.to_string(index=False))
    else:
        print("  (no candidate rows)")

    print()
    print("Geometry distributions, TP vs FN (candidate rows only)")
    gt_tp_fn = gt.assign(tp_or_fn=gt["classification"].map(lambda c: "TP" if c == DIVISION_TP else "FN"))
    dist_tp_fn = geometry_distributions(gt_tp_fn, group_column="tp_or_fn")
    if len(dist_tp_fn):
        print(dist_tp_fn.to_string(index=False))
    else:
        print("  (no candidate rows)")

    print()
    print("=" * 72)
    print("B. Predicted final divisions")
    print("=" * 72)
    print(f"  predicted forks (out-degree >= 2) : {len(pred)}")
    print(f"  counted as division TP            : {int(pred['counted_as_tp'].sum())}")
    print(f"  counted as division FP            : {int(pred['counted_as_fp'].sum())}")
    print(f"  not evaluated by the metric       : {int((~pred['evaluated_by_metric']).sum())}")

    print()
    print("Predicted fork scope under the locked metric")
    fork_scope = summarize_pred_fork_scope(pred)
    print(fork_scope.to_string(index=False))

    print()
    print("FP structural summary")
    fp_structure = summarize_fp_structure(pred)
    print(fp_structure.to_string(index=False))

    fp = pred[pred["counted_as_fp"]]
    if len(fp):
        print()
        print("FP GT-context breakdown")
        print(f"  parent matches a GT node            : {int(fp['parent_matches_gt'].sum())}")
        print(f"  parent matches a GT division        : {int(fp['gt_parent_is_division'].sum())}")
        print(f"  parent matches a non-division GT parent : {int(fp['gt_parent_is_non_division_parent'].sum())}")
        print(f"  parent matches where GT annotation ends : {int(fp['gt_parent_annotation_ends'].sum())}")
        print(f"  plausibly unlabeled (sparse GT)     : {int(fp['sparse_gt_plausible'].sum())}")
        print(f"  GT contradicts the division         : {int(fp['gt_contradicts_division'].sum())}")
        print(f"  forks with an untracked daughter    : {int((fp['unmatched_daughter_branches'] > 0).sum())}")
        print(f"  origin: accepted by frozen G2       : {int(fp['origin_g2_accepted'].sum())}")
        print(f"  origin: includes a recovered edge   : {int(fp['origin_includes_recovered_edge'].sum())}")

    print()
    print("Per-prefix predicted-division counts")
    prefix_pred = (
        pred.groupby("prefix")
        .agg(
            forks=("parent_node", "size"),
            tp=("counted_as_tp", "sum"),
            fp=("counted_as_fp", "sum"),
            not_evaluated=("evaluated_by_metric", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    print(prefix_pred.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)

    gt.to_csv(out_dir / "gt_divisions.csv", index=False)
    pred.to_csv(out_dir / "pred_divisions.csv", index=False)
    audit.per_dataset.to_csv(out_dir / "per_dataset.csv", index=False)
    overall.to_csv(out_dir / "classification_summary.csv", index=False)
    by_prefix.to_csv(out_dir / "prefix_summary.csv", index=False)
    fp_structure.to_csv(out_dir / "fp_structure.csv", index=False)
    fork_scope.to_csv(out_dir / "pred_fork_scope.csv", index=False)
    dist.to_csv(out_dir / "geometry_by_classification.csv", index=False)
    dist_tp_fn.to_csv(out_dir / "geometry_tp_vs_fn.csv", index=False)

    payload = {
        "metric_summary": audit.metric_summary,
        "division_tp": audit.division_tp,
        "division_fp": audit.division_fp,
        "division_fn": audit.division_fn,
        "gt_divisions": len(gt),
        "fn_bottleneck": bottleneck,
        "classification_counts": overall.set_index("classification")["n"].to_dict(),
        "fp_structure_counts": fp_structure.set_index("fp_structure_class")["n"].to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float))

    print()
    print(f"Artifacts written to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true", help="print per-dataset progress")
    args = parser.parse_args()

    project = args.project.resolve()
    out_dir = args.out_dir or (project / "reports/stage11_division_audit")

    status, audit = run(project, args.progress)

    if status != 0 or audit is None:
        return status

    report(audit, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
