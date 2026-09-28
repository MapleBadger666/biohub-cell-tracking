#!/usr/bin/env python
"""Stage 11.03 — read-only Frozen G2 division geometry calibration audit.

Measures the geometry of every annotated division in the training set, sweeps a
predeclared counterfactual geometry grid over it, and evaluates a small
shortlist through the *real* Frozen V9 pipeline and the locked official metric
using ``FrozenV9Config`` overrides.

Diagnostic only. Nothing here promotes a policy: ``frozen_v9.py`` is imported,
never modified, and every candidate gate is reported as a candidate for later
validation. No training, no inference, no GEFF is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from tracking_cellmot.io import open_dataset  # noqa: E402 - path bootstrap must precede import
from tracking_cellmot.metrics import summarise  # noqa: E402 - path bootstrap must precede import

from biohub_cell_tracking.association_audit import (  # noqa: E402 - path bootstrap must precede import
    graph_view,
)
from biohub_cell_tracking.division_audit import (  # noqa: E402 - path bootstrap must precede import
    build_division_candidates,
    fork_geometry,
)
from biohub_cell_tracking.division_geometry import (  # noqa: E402 - path bootstrap must precede import
    GEOMETRY_VARIABLES,
    GeometryGate,
    census_gt_divisions,
    enumerate_geometry_grid,
    evaluate_geometry_gate,
    geometry_pass_rates,
    grid_gt_recall,
    quantile_summary,
)
from biohub_cell_tracking.frozen_v9 import (  # noqa: E402 - path bootstrap must precede import
    build_frozen_v9_policies,
    evaluate_frozen_v9,
    load_graph,
)

GT_DIR = "data/raw/competition/biohub-cell-tracking-during-development/train"

# Stage 11.01 dual-checkpoint domain routing, reproduced at score 0.777452947.
ROUTING = {
    "44b6": (
        "predictions/heqiuyan/stage10_infer_sparse9995_256_det995/split_0",
        "predictions/heqiuyan/stage10_infer_sparse9995_256_det9975/split_0",
    ),
    "6bba": (
        "predictions/heqiuyan/stage10_infer_baseline_256_det995/split_0",
        "predictions/heqiuyan/stage10_infer_baseline_256_det9975/split_0",
    ),
}

# Predeclared Part C grid: 4 x 4 x 4 x 3 = 192 configurations, p2 fixed.
GRID_SEPARATIONS = [0.0, 2.0, 4.0, 6.0]
GRID_ANGLES = [0.0, 60.0, 90.0, 120.0]
GRID_DISTANCES = [8.0, 10.0, 12.0, 14.0]
GRID_BALANCES = [0.50, 0.75, 1.00]

# Part E secondary sweep.
P2_LEVELS = [0.75, 0.80, 0.85, 0.90]


def log(lines: list[str], text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--faithful-shortlist",
        type=int,
        default=3,
        help="How many relaxed candidates to re-run through the real Frozen V9 pipeline.",
    )
    args = parser.parse_args()

    project = args.project.resolve()
    out_dir = args.out_dir or (project / "reports/stage11_geometry_audit")
    gt_dir = project / GT_DIR
    manifest = project / "configs/cv_manifest.csv"
    stage1101 = project / "reports/stage11_division_audit"

    for path in (gt_dir, manifest, stage1101 / "gt_divisions.csv"):
        if not path.exists():
            print(f"FAIL: missing required input: {path}")
            return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 80)

    meta = pd.read_csv(manifest)
    dataset_names = sorted(meta["dataset"])
    fold_map = dict(zip(meta["dataset"], meta["fold"], strict=True))
    prefix_map = dict(zip(meta["dataset"], meta["prefix"], strict=True))
    estimated_nodes = dict(zip(meta["dataset"], meta["estimated_nodes"], strict=True))

    frozen_gate = GeometryGate.frozen()

    log(lines, "Stage 11.03 — Frozen G2 division geometry calibration audit")
    log(lines, f"  training datasets : {len(dataset_names)}")
    log(lines, f"  frozen gate       : {frozen_gate.label}")
    log(lines)

    # ------------------------------------------------------------------
    # Part A — GT geometry census over the whole labelled training set
    # ------------------------------------------------------------------
    log(lines, "PART A — ground-truth division geometry census")
    started = time.time()
    census = census_gt_divisions(dataset_names, gt_dir, fold_map=fold_map, prefix_map=prefix_map, progress=True)
    log(lines, f"  GT divisions found : {len(census)}  ({time.time() - started:.1f}s)")
    log(lines, f"  manifest total     : {int(meta['n_divisions'].sum())}")

    if len(census) != int(meta["n_divisions"].sum()):
        log(lines, "  NOTE: census count differs from the manifest; census is authoritative.")

    log(lines, f"  divisions with >2 daughters : {int(census['extra_daughters'].sum())}")
    log(lines, f"  degenerate zero-length step : {int(census['degenerate_zero_step'].sum())}")
    log(lines, f"  non-finite angle            : {int(census['degenerate_angle_nan'].sum())}")
    log(lines, f"  daughters not at t+1        : {int((~census['daughter_frame_gap_ok']).sum())}")
    log(lines, f"  by fold   : {census['fold'].value_counts().sort_index().to_dict()}")
    log(lines, f"  by prefix : {census['prefix'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Part B — distributions and current-gate pass rates
    # ------------------------------------------------------------------
    log(lines)
    log(lines, "PART B — geometry distributions")
    global_quantiles = quantile_summary(census)
    log(lines, global_quantiles.to_string(index=False))

    by_fold_q = quantile_summary(census, group_cols=["fold"])
    by_prefix_q = quantile_summary(census, group_cols=["prefix"])

    log(lines)
    log(lines, "Median by fold (all geometry variables)")
    fold_medians = by_fold_q.pivot(index="fold", columns="variable", values="q50")
    log(lines, fold_medians.to_string())

    global_rates = geometry_pass_rates(census, frozen_gate)
    fold_rates = geometry_pass_rates(census, frozen_gate, group_cols=["fold"])
    prefix_rates = geometry_pass_rates(census, frozen_gate, group_cols=["prefix"])

    log(lines)
    log(lines, "Current frozen gate — GT pass rates (global)")
    for criterion in ("separation", "angle", "distance", "balance"):
        col = f"passes_{criterion}"
        log(
            lines,
            f"  {criterion:<12} {int(global_rates.loc[0, col + '_n']):>4} / {len(census)}"
            f"   ({100 * global_rates.loc[0, col]:5.1f}%)",
        )
    log(
        lines,
        f"  FULL GATE    {int(global_rates.loc[0, 'passes_geometry_n']):>4} / {len(census)}"
        f"   ({100 * global_rates.loc[0, 'passes_geometry']:5.1f}%)",
    )

    log(lines)
    log(lines, "Current frozen gate — GT pass rate by fold")
    log(
        lines,
        fold_rates[["fold", "n", "passes_geometry_n", "passes_geometry"]].to_string(index=False),
    )
    log(lines)
    log(lines, "Current frozen gate — GT pass rate by prefix")
    log(
        lines,
        prefix_rates[
            ["prefix", "n", "passes_separation", "passes_angle", "passes_distance", "passes_balance", "passes_geometry"]
        ].to_string(index=False),
    )

    fold0_recall = float(fold_rates.loc[fold_rates["fold"] == 0, "passes_geometry"].iloc[0])
    other_recalls = fold_rates.loc[fold_rates["fold"] != 0, "passes_geometry"]
    fold0_outlier = bool(fold0_recall < other_recalls.min() - 0.10 or fold0_recall > other_recalls.max() + 0.10)
    log(lines)
    log(
        lines,
        f"  Fold0 full-gate recall {fold0_recall:.3f} vs other folds "
        f"[{other_recalls.min():.3f}, {other_recalls.max():.3f}] -> "
        f"{'OUTLIER' if fold0_outlier else 'representative'}",
    )

    # ------------------------------------------------------------------
    # Part C — geometry grid, GT recall only
    # ------------------------------------------------------------------
    log(lines)
    log(lines, "PART C — geometry threshold grid (GT recall)")
    grid = enumerate_geometry_grid(GRID_SEPARATIONS, GRID_ANGLES, GRID_DISTANCES, GRID_BALANCES)
    grid_recall = grid_gt_recall(census, grid)
    log(lines, f"  configurations : {len(grid)}")
    log(
        lines,
        f"  GT recall range: {grid_recall['gt_recall'].min():.3f} .. {grid_recall['gt_recall'].max():.3f}",
    )

    # ------------------------------------------------------------------
    # Part D — Fold0 prediction counterfactual (observed graph + oracle)
    # ------------------------------------------------------------------
    log(lines)
    log(lines, "PART D — Fold0 prediction counterfactual")

    fold0 = meta[meta["fold"] == 0]
    fold0_names = sorted(fold0["dataset"])

    log(lines, "  loading raw det=.995 prediction graphs (read-only)")
    candidate_rows: list[dict] = []
    views = {}
    for name in fold0_names:
        prefix = prefix_map[name]
        pred_995 = project / ROUTING[prefix][0]
        scale = tuple(float(s) for s in open_dataset(gt_dir / name, load_image=False).scale)
        graph = load_graph(pred_995 / f"{name}.geff")
        views[name] = graph_view(graph, name, scale)
        for candidate in build_division_candidates(graph, np.asarray(scale, dtype=float)).values():
            candidate_rows.append(
                {
                    "dataset": name,
                    "prefix": prefix,
                    "source": candidate.source_id,
                    "p1": candidate.p1,
                    "p2": candidate.p2,
                    "daughter_separation_um": candidate.geometry.daughter_separation_um,
                    "division_angle_deg": candidate.geometry.division_angle_deg,
                    "max_parent_daughter_um": candidate.geometry.max_parent_daughter_um,
                    "distance_balance": candidate.geometry.distance_balance,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    log(lines, f"  raw out-degree-2 fork candidates in Fold0 : {len(candidates)}")

    # Oracle-association upper bound: GT divisions whose three nodes are already
    # matched, measured on PREDICTED coordinates.
    gt_divisions = pd.read_csv(stage1101 / "gt_divisions.csv")
    nodes_present = gt_divisions[
        gt_divisions["parent_matched"] & gt_divisions["daughter1_matched"] & gt_divisions["daughter2_matched"]
    ].copy()

    oracle_rows = []
    for row in nodes_present.itertuples(index=False):
        view = views[row.dataset]
        p = view.raw_position(int(row.pred_parent))
        c1 = view.raw_position(int(row.pred_daughter_1))
        c2 = view.raw_position(int(row.pred_daughter_2))
        if p is None or c1 is None or c2 is None:
            continue
        geometry = fork_geometry(p, c1, c2, view.scale)
        oracle_rows.append(
            {
                "dataset": row.dataset,
                "prefix": row.prefix,
                "classification": row.classification,
                "daughter_separation_um": geometry.daughter_separation_um,
                "division_angle_deg": geometry.division_angle_deg,
                "max_parent_daughter_um": geometry.max_parent_daughter_um,
                "distance_balance": geometry.distance_balance,
            }
        )

    oracle = pd.DataFrame(oracle_rows)
    oracle_fn = oracle[oracle["classification"] != "DIVISION_TP"]
    log(lines, f"  GT divisions with all three nodes matched : {len(oracle)}")
    log(lines, f"  of which currently false negatives        : {len(oracle_fn)}")

    counterfactual_rows = []
    for gate in grid:
        accepted = evaluate_geometry_gate(candidates, gate)["passes_all"]
        oracle_pass = (
            evaluate_geometry_gate(oracle_fn, gate)["passes_geometry"] if len(oracle_fn) else pd.Series(dtype=bool)
        )
        recall_row = grid_recall[grid_recall["gate"] == gate.label].iloc[0]

        counterfactual_rows.append(
            {
                **gate.as_dict(),
                "is_frozen_geometry": gate.is_frozen_geometry,
                # OBSERVED GRAPH COUNTERFACTUAL — candidates actually in the saved predictions.
                "observed_accepted_forks": int(accepted.sum()),
                "observed_accepted_ratio_vs_frozen": float("nan"),
                # ORACLE-ASSOCIATION UPPER BOUND — separate quantity, do not mix.
                "oracle_recoverable_fn": int(oracle_pass.sum()) if len(oracle_fn) else 0,
                "oracle_fn_total": len(oracle_fn),
                "gt_recall": float(recall_row["gt_recall"]),
                "gt_recall_fold0": float(recall_row.get("gt_recall_fold0", float("nan"))),
                "gt_recall_fold_min": float(recall_row.get("gt_recall_fold_min", float("nan"))),
                "gt_recall_fold_max": float(recall_row.get("gt_recall_fold_max", float("nan"))),
            }
        )

    counterfactual = pd.DataFrame(counterfactual_rows)
    frozen_forks = int(counterfactual.loc[counterfactual["is_frozen_geometry"], "observed_accepted_forks"].iloc[0])
    counterfactual["observed_accepted_ratio_vs_frozen"] = counterfactual["observed_accepted_forks"] / max(
        frozen_forks, 1
    )

    log(lines, f"  frozen gate accepts {frozen_forks} of {len(candidates)} raw fork candidates")
    log(
        lines,
        f"  observed accepted forks across the grid: "
        f"{counterfactual['observed_accepted_forks'].min()} .. "
        f"{counterfactual['observed_accepted_forks'].max()}",
    )

    # ------------------------------------------------------------------
    # Shortlist: material GT gain, fold-stable, bounded fork inflation
    # ------------------------------------------------------------------
    frozen_recall = float(counterfactual.loc[counterfactual["is_frozen_geometry"], "gt_recall"].iloc[0])

    shortlist = counterfactual[
        (~counterfactual["is_frozen_geometry"])
        & (counterfactual["gt_recall"] >= frozen_recall + 0.15)
        & (counterfactual["gt_recall_fold_min"] >= frozen_recall)
        & (counterfactual["observed_accepted_ratio_vs_frozen"] <= 3.0)
    ].copy()

    shortlist["fork_cost"] = shortlist["observed_accepted_ratio_vs_frozen"]
    shortlist = shortlist.sort_values(["gt_recall", "fork_cost"], ascending=[False, True]).drop_duplicates(
        subset=["gt_recall"], keep="first"
    )

    top = shortlist.head(args.faithful_shortlist)

    log(lines)
    log(lines, "PART F — shortlist (material GT gain, fold-stable, bounded fork inflation)")
    log(lines, f"  frozen GT recall {frozen_recall:.3f}; shortlist candidates {len(shortlist)}")
    if len(top):
        log(
            lines,
            top[
                [
                    "gate",
                    "gt_recall",
                    "gt_recall_fold0",
                    "gt_recall_fold_min",
                    "gt_recall_fold_max",
                    "observed_accepted_forks",
                    "observed_accepted_ratio_vs_frozen",
                    "oracle_recoverable_fn",
                ]
            ].to_string(index=False),
        )

    # ------------------------------------------------------------------
    # Faithful Fold0 re-evaluation through the real pipeline + locked metric
    # ------------------------------------------------------------------
    log(lines)
    log(lines, "PART D (faithful) — real Frozen V9 + locked official metric on Fold0")
    log(lines, "  frozen_v9.py is unmodified; only FrozenV9Config thresholds differ per run.")

    faithful_gates = [frozen_gate, *[GeometryGate(**_gate_kwargs(r)) for _, r in top.iterrows()]]

    faithful_rows = []
    for gate in faithful_gates:
        config = gate.to_config()
        started = time.time()
        records = []
        for prefix, (p995, p9975) in ROUTING.items():
            names = [n for n in fold0_names if prefix_map[n] == prefix]
            policies = build_frozen_v9_policies(
                names, project / p995, project / p9975, gt_dir, prefix_map=prefix_map, config=config
            )
            evaluation = evaluate_frozen_v9(
                names,
                project / p995,
                project / p9975,
                gt_dir,
                estimated_nodes,
                prefix_map=prefix_map,
                policies=policies,
                config=config,
            )
            records.extend(evaluation.records)

        summary = summarise(records)
        faithful_rows.append(
            {
                **gate.as_dict(),
                "is_frozen_geometry": gate.is_frozen_geometry,
                "division_tp": int(summary["division_tp"]),
                "division_fp": int(summary["division_fp"]),
                "division_fn": int(summary["division_fn"]),
                "division_jaccard": float(summary["division_jaccard"]),
                "edge_jaccard": float(summary["edge_jaccard"]),
                "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
                "node_recall": float(summary["node_recall"]),
                "score": float(summary["score"]),
                "runtime_s": round(time.time() - started, 1),
            }
        )
        log(
            lines,
            f"  {gate.label:<48} TP={summary['division_tp']:>3} FP={summary['division_fp']:>4} "
            f"FN={summary['division_fn']:>3}  divJ={summary['division_jaccard']:.6f}  "
            f"score={summary['score']:.6f}",
        )

    faithful = pd.DataFrame(faithful_rows)

    # ------------------------------------------------------------------
    # Part E — small p2 secondary sensitivity
    # ------------------------------------------------------------------
    log(lines)
    log(lines, "PART E — p2 secondary sensitivity (geometry candidates x p2)")
    p2_gates = enumerate_geometry_grid(
        [frozen_gate.daughter_sep_min_um],
        [frozen_gate.angle_min_deg],
        [frozen_gate.max_parent_daughter_um],
        [frozen_gate.distance_balance_max],
        P2_LEVELS,
    )
    if len(top):
        mild = top.iloc[0]
        p2_gates += enumerate_geometry_grid(
            [mild["sep_min_um"]],
            [mild["angle_min_deg"]],
            [mild["max_parent_daughter_um"]],
            [mild["balance_max"]],
            P2_LEVELS,
        )
    p2_gates += enumerate_geometry_grid([0.0], [0.0], [14.0], [1.0], P2_LEVELS)

    p2_rows = []
    for gate in p2_gates:
        accepted = evaluate_geometry_gate(candidates, gate)["passes_all"]
        p2_rows.append(
            {
                **gate.as_dict(),
                "observed_accepted_forks": int(accepted.sum()),
                "gt_recall_geometry_only": float(evaluate_geometry_gate(census, gate)["passes_geometry"].mean()),
            }
        )
    p2_sensitivity = pd.DataFrame(p2_rows)
    log(
        lines,
        p2_sensitivity[["gate", "p2_min", "observed_accepted_forks", "gt_recall_geometry_only"]].to_string(index=False),
    )

    # ------------------------------------------------------------------
    # Part G — TP/FP geometry overlap
    # ------------------------------------------------------------------
    log(lines)
    log(lines, "PART G — TP/FP geometry overlap among currently accepted forks")
    pred_divisions = pd.read_csv(stage1101 / "pred_divisions.csv")
    judgeable = pred_divisions[pred_divisions["evaluated_by_metric"]]
    overlap_rows = []
    for label, block in (
        ("division_TP", judgeable[judgeable["counted_as_tp"]]),
        ("division_FP", judgeable[judgeable["counted_as_fp"]]),
    ):
        row = {"group": label, "n": len(block)}
        for variable in ("p2", *GEOMETRY_VARIABLES[:2], "max_parent_daughter_um", "distance_balance"):
            if variable not in block.columns:
                continue
            values = pd.to_numeric(block[variable], errors="coerce").dropna()
            row[f"{variable}_min"] = float(values.min()) if len(values) else float("nan")
            row[f"{variable}_median"] = float(values.median()) if len(values) else float("nan")
            row[f"{variable}_max"] = float(values.max()) if len(values) else float("nan")
        overlap_rows.append(row)
    overlap = pd.DataFrame(overlap_rows)
    log(lines, overlap.to_string(index=False))

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    census.to_csv(out_dir / "gt_division_geometry.csv", index=False)
    pd.concat([by_fold_q, fold_rates.assign(variable="PASS_RATE")], ignore_index=True).to_csv(
        out_dir / "gt_geometry_summary_by_fold.csv", index=False
    )
    pd.concat([by_prefix_q, prefix_rates.assign(variable="PASS_RATE")], ignore_index=True).to_csv(
        out_dir / "gt_geometry_summary_by_prefix.csv", index=False
    )
    grid_recall.to_csv(out_dir / "geometry_grid.csv", index=False)
    counterfactual.to_csv(out_dir / "fold0_counterfactual_grid.csv", index=False)
    faithful.to_csv(out_dir / "fold0_faithful_reevaluation.csv", index=False)
    p2_sensitivity.to_csv(out_dir / "p2_secondary_sensitivity.csv", index=False)
    global_quantiles.to_csv(out_dir / "gt_geometry_quantiles_global.csv", index=False)

    payload = {
        "status": "DIAGNOSTIC ONLY — no policy promoted; every gate is a candidate for later validation",
        "n_gt_divisions": len(census),
        "gt_divisions_by_fold": {int(k): int(v) for k, v in census["fold"].value_counts().items()},
        "gt_divisions_by_prefix": census["prefix"].value_counts().to_dict(),
        "frozen_gate": frozen_gate.as_dict(),
        "frozen_gate_gt_recall_global": frozen_recall,
        "frozen_gate_gt_recall_by_fold": {
            int(r["fold"]): float(r["passes_geometry"]) for _, r in fold_rates.iterrows()
        },
        "frozen_gate_gt_recall_by_prefix": {
            str(r["prefix"]): float(r["passes_geometry"]) for _, r in prefix_rates.iterrows()
        },
        "frozen_criterion_pass_rates_global": {
            criterion: float(global_rates.loc[0, f"passes_{criterion}"])
            for criterion in ("separation", "angle", "distance", "balance")
        },
        "fold0_is_outlier": fold0_outlier,
        "grid_size": len(grid),
        "frozen_observed_accepted_forks": frozen_forks,
        "raw_candidate_forks_fold0": len(candidates),
        "oracle_fn_total": len(oracle_fn),
        "shortlist": top[
            ["gate", "gt_recall", "gt_recall_fold_min", "observed_accepted_forks", "oracle_recoverable_fn"]
        ].to_dict("records")
        if len(top)
        else [],
        "faithful_reevaluation": faithful.to_dict("records"),
        "tp_fp_overlap": overlap.to_dict("records"),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(payload, indent=2, default=float))
    (out_dir / "audit_run.log").write_text("\n".join(lines) + "\n")

    log(lines, "")
    print(f"Artifacts written to {out_dir}")
    (out_dir / "audit_run.log").write_text("\n".join(lines) + "\n")
    return 0


def _gate_kwargs(row) -> dict:
    return {
        "daughter_sep_min_um": float(row["sep_min_um"]),
        "angle_min_deg": float(row["angle_min_deg"]),
        "max_parent_daughter_um": float(row["max_parent_daughter_um"]),
        "distance_balance_max": float(row["balance_max"]),
        "p2_min": float(row["p2_min"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
