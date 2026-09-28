#!/usr/bin/env python
"""Stage 11.02 — read-only association candidate-generation audit.

Reads the Stage 11.01 division-audit artifacts, isolates the
``NODES_PRESENT_EDGES_MISSING`` GT divisions, and inspects the *saved prediction
GEFFs* to determine which stage of the frozen predictor dropped each missing
parent -> daughter association.

This script loads no checkpoint, runs no model, writes no prediction GEFF, and
does not modify any Stage 11.01 artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from tracking_cellmot.io import open_dataset  # noqa: E402 - path bootstrap must precede import

from biohub_cell_tracking.association_audit import (  # noqa: E402 - path bootstrap must precede import
    ASSOCIATION_DATAFLOW,
    EDGE_STATUS_MEANING,
    EDGE_STATUSES,
    FROZEN_PREDICT_CONFIG,
    build_case_table,
    graph_view,
    summarize_cases,
)
from biohub_cell_tracking.frozen_v9 import load_graph  # noqa: E402 - path bootstrap must precede import

GT_DIR = "data/raw/competition/biohub-cell-tracking-during-development/train"

# Same dual-checkpoint domain routing Stage 11.01 reproduced (score 0.777452947).
ROUTING = {
    "44b6": "predictions/heqiuyan/stage10_infer_sparse9995_256_det995/split_0",
    "6bba": "predictions/heqiuyan/stage10_infer_baseline_256_det995/split_0",
}

TARGET_CLASSIFICATION = "NODES_PRESENT_EDGES_MISSING"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--stage1101-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    stage1101 = args.stage1101_dir or (project / "reports/stage11_division_audit")
    out_dir = args.out_dir or (project / "reports/stage11_association_audit")
    gt_dir = project / GT_DIR

    gt_path = stage1101 / "gt_divisions.csv"
    if not gt_path.exists():
        print(f"FAIL: missing Stage 11.01 artifact: {gt_path}")
        return 2

    gt = pd.read_csv(gt_path)
    cases = gt[gt["classification"] == TARGET_CLASSIFICATION].copy().reset_index(drop=True)

    print("Stage 11.02 — association candidate-generation audit")
    print(f"  Stage 11.01 GT divisions : {len(gt)}")
    print(f"  {TARGET_CLASSIFICATION} : {len(cases)}")

    if cases.empty:
        print("FAIL: no target cases found")
        return 2

    missing_ids = cases[["pred_parent", "pred_daughter_1", "pred_daughter_2"]].isna().any(axis=1)
    if missing_ids.any():
        print(f"FAIL: {int(missing_ids.sum())} case(s) lack predicted node ids")
        return 2

    print()
    print("Frozen predictor association policy")
    for key, value in FROZEN_PREDICT_CONFIG.items():
        print(f"  {key:<24} {value}")

    print()
    print(ASSOCIATION_DATAFLOW["diagram"])

    datasets = sorted(cases["dataset"].unique())
    print()
    print(f"Loading {len(datasets)} prediction GEFF(s) (read-only)")

    views = {}
    gt_views = {}
    for name in datasets:
        prefix = str(cases.loc[cases["dataset"] == name, "prefix"].iloc[0])
        pred_dir = project / ROUTING[prefix]
        geff = pred_dir / f"{name}.geff"
        if not geff.exists():
            print(f"FAIL: missing prediction GEFF: {geff}")
            return 2
        scale = np.asarray(open_dataset(gt_dir / name, load_image=False).scale, dtype=float)
        views[name] = graph_view(load_graph(geff), name, tuple(scale))
        # GT node coordinates only — used to test whether the division is
        # compatible with the frozen G2 geometry policy at all. No metric is run.
        gt_views[name] = graph_view(load_graph(gt_dir / f"{name}.geff"), name, tuple(scale))

    case_table = build_case_table(cases, views, gt_views=gt_views)
    summary = summarize_cases(case_table)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    print()
    print("=" * 72)
    print("Per-case structure")
    print("=" * 72)
    print(
        case_table[
            [
                "dataset",
                "prefix",
                "parent",
                "parent_t",
                "daughter_1_frame_gap",
                "daughter_2_frame_gap",
                "raw_parent_out_degree",
                "daughter_1_in_degree",
                "daughter_2_in_degree",
                "edge_1_status",
                "edge_2_status",
                "case_structure",
            ]
        ].to_string(index=False)
    )

    print()
    print("Edge-status counts (2 required edges per case)")
    for status in EDGE_STATUSES:
        n = summary["edge_status_counts"][status]
        meaning = EDGE_STATUS_MEANING[status]
        flag = "disk-determinable" if meaning["determinable_from_geff"] else "NEEDS RAW SCORES"
        print(f"  {status:<38} {n:>3}   {flag}")

    print()
    print("Case structure counts")
    for structure, n in summary["case_structure_counts"].items():
        print(f"  {structure:<24} {n:>3}")

    print()
    print("Root cause")
    for cause, n in summary["root_cause_counts"].items():
        print(f"  {cause:<28} {n:>3}")
    print(f"  cases determinable from GEFF : {summary['cases_determinable_from_geff']}")
    print(f"  cases requiring raw scores   : {summary['cases_requiring_raw_scores']}")

    print()
    print("Probability context of the surviving edge, per case")
    present = case_table[case_table["case_structure"] == "ONE_EDGE_PRESENT"]
    if len(present):
        kept = present.apply(
            lambda r: r["edge_1_prob"] if r["edge_1_status"] == "EDGE_PRESENT" else r["edge_2_prob"],
            axis=1,
        )
        print(f"  n={len(kept)}  min={kept.min():.4f}  median={kept.median():.4f}  max={kept.max():.4f}")

    competitor = case_table[case_table["matching_suppression_indicated"]]
    if len(competitor):
        probs = pd.to_numeric(
            competitor[["competing_in_1_probs", "competing_in_2_probs"]]
            .apply(lambda s: s.str.split(",").str[0])
            .stack(),
            errors="coerce",
        ).dropna()
        print()
        print("Competitor edges that claimed a required daughter column")
        print(f"  n={len(probs)}  min={probs.min():.4f}  median={probs.median():.4f}  max={probs.max():.4f}")

    print()
    print("Would a restored fork pass Frozen G2 geometry?")
    print(f"  passes all four criteria : {summary['restored_fork_passes_g2_geometry']}")
    print(f"  fails at least one       : {summary['restored_fork_fails_g2_geometry']}")
    for flag, n in summary["restored_geometry_failure_counts"].items():
        print(f"    fails {flag:<12} {n:>3}")
    print(f"  recoverable (geometry OK and stage known) : {summary['recoverable_cases']}")

    print()
    print("Missing-edge parent->daughter distance vs the frozen 8 um G2 cap")
    md = case_table["missing_daughter_max_distance_um"].dropna()
    print(f"  n={len(md)}  min={md.min():.2f}  median={md.median():.2f}  max={md.max():.2f}")
    print(f"  above 8 um : {int((md > 8.0).sum())}")

    comp = pd.concat([case_table["competitor_1_to_daughter_um"], case_table["competitor_2_to_daughter_um"]]).dropna()
    if len(comp):
        print()
        print("Competitor -> claimed daughter distance (um)")
        print(f"  n={len(comp)}  min={comp.min():.2f}  median={comp.median():.2f}  max={comp.max():.2f}")

    print()
    print("Is the GT division itself compatible with the frozen G2 geometry policy?")
    print(f"  GT fork passes all four criteria : {summary['gt_fork_passes_g2_geometry']} / {len(case_table)}")
    for flag, n in summary["gt_geometry_failure_counts"].items():
        print(f"    GT fails {flag:<12} {n:>3}")

    print()
    print("GT vs predicted fork geometry (um / deg)")
    print(
        case_table[
            [
                "dataset",
                "parent",
                "gt_max_parent_daughter_um",
                "restored_max_parent_daughter_um",
                "gt_division_angle_deg",
                "restored_division_angle_deg",
                "gt_passes_g2_geometry",
                "restored_passes_g2_geometry",
            ]
        ].to_string(index=False)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    case_table.to_csv(out_dir / "missing_second_edge_cases.csv", index=False)
    (out_dir / "source_dataflow.json").write_text(
        json.dumps(
            {
                "frozen_predict_config": FROZEN_PREDICT_CONFIG,
                "dataflow": ASSOCIATION_DATAFLOW,
                "edge_status_meaning": EDGE_STATUS_MEANING,
            },
            indent=2,
        )
    )
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"Artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
