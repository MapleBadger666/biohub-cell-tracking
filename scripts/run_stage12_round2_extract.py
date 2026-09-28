# ============================================================
# RUN — Stage12 Full Round-2 Reciprocal Universe Extractor
# Purpose:
#   Extract the complete GT-blind Round-2 reciprocal candidate
#   universe on folds1–4 after locked MUTUAL-CONT-V1 occupancy.
#
# Changes files: YES — Stage12 report artifacts
# Runs training: NO
# Runs inference: YES — parity-verified fast association replay
# Uses GT: NO
# Keep after running: YES
# ============================================================

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]

TRAIN_ROOT = (
    PROJECT
    / "data/raw/competition/"
      "biohub-cell-tracking-during-development/train"
)

P995 = (
    PROJECT
    / "predictions/heqiuyan/"
      "stage11_fullgraph_baseline256_folds1to4_det995/split_0"
)

P9975 = (
    PROJECT
    / "predictions/heqiuyan/"
      "stage11_fullgraph_baseline256_folds1to4_det9975/split_0"
)

CKPT = (
    PROJECT
    / "models/official_baseline/"
      "stage10_pilot_baseline_256/"
      "split_0/edge_predictor_best.pth"
)

CV_PATH = PROJECT / "configs/cv_manifest.csv"

S11 = PROJECT / "reports/stage11_division_probe"
S12 = PROJECT / "reports/stage12_residual_audit"

V1_UNIVERSE_PATH = (
    S12 / "stage12_folds14_mutual_best_universe.pkl"
)

OUT = (
    S12 / "stage12_folds14_round2_reciprocal_universe.pkl"
)

SUMMARY_OUT = (
    S12 / "stage12_folds14_round2_reciprocal_universe_summary.json"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]

from biohub_cell_tracking import frozen_v9 as fv9

import run_stage11_multiframe_extract as runner
import stage12_iterative_reciprocal as ir

from replay_stage11_association import DatasetReplay


# ------------------------------------------------------------
# Corpus
# ------------------------------------------------------------

cv = pd.read_csv(CV_PATH)

corp = (
    cv[
        cv["fold"].isin([1, 2, 3, 4])
        & cv["dataset"].str.startswith("6bba_")
    ]
    .sort_values(["fold", "dataset"])
    .reset_index(drop=True)
)

assert len(corp) == 103

names = corp["dataset"].tolist()

fold_map = dict(
    zip(corp["dataset"], corp["fold"])
)

prefix_map = dict(
    zip(corp["dataset"], corp["prefix"])
)


# ------------------------------------------------------------
# Locked Stage11 morphology + V1
# ------------------------------------------------------------

morph_all = pd.read_pickle(
    S11
    / "stage11_deployable_final_v9_policy_grid_actions.pkl"
)

morph = (
    morph_all[
        morph_all["config_id"].eq("K3_R3_P100")
    ]
    .copy()
)

v1_universe = pd.read_pickle(
    V1_UNIVERSE_PATH
)

v1 = (
    v1_universe[
        v1_universe["candidate_prob"].ge(0.25)
    ]
    .copy()
)

assert len(v1) == 46951


policies = fv9.build_frozen_v9_policies(
    names,
    P995,
    P9975,
    TRAIN_ROOT,
    prefix_map=prefix_map,
    config=fv9.FROZEN_V9_CONFIG,
    progress=False,
)


# ------------------------------------------------------------
# Exact Stage11 final topology before V1
# ------------------------------------------------------------

def stage11_topology(name):

    g, scale, _ = fv9.apply_frozen_v9(
        name,
        P995,
        P9975,
        policies,
        TRAIN_ROOT,
        config=fv9.FROZEN_V9_CONFIG,
    )

    aa = (
        morph[
            morph["dataset"].eq(name)
        ]
        .sort_values(
            [
                "morph_rank",
                "t",
                "source_pred_node_id",
                "target_pred_node_id",
            ]
        )
    )

    nodes = (
        g.node_attrs(
            attr_keys=[
                "node_id", "t", "z", "y", "x"
            ]
        )
        .to_pandas()
        .set_index("node_id")
    )

    edges = (
        g.edge_attrs(unpack=True)
        .to_pandas()
    )

    outdeg = (
        edges.groupby("source_id")
        .size()
        .astype(int)
        .to_dict()
    )

    indeg = (
        edges.groupby("target_id")
        .size()
        .astype(int)
        .to_dict()
    )

    for a in aa.itertuples(index=False):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert outdeg.get(s, 0) == 1
        assert indeg.get(q, 0) == 0

        outdeg[s] = 2
        indeg[q] = 1

    return nodes, outdeg, indeg, np.asarray(scale, dtype=float)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

device = runner.resolve_device("mps")

model, window_size, downsample = runner.load_model(
    CKPT,
    device,
)

assert window_size == 2
assert tuple(downsample) == (1, 4, 4)


# ------------------------------------------------------------
# Extraction
# ------------------------------------------------------------

records = []
started = time.time()


for di, name in enumerate(names, start=1):

    nodes, outdeg, indeg, scale = (
        stage11_topology(name)
    )

    final_ids = set(
        map(int, nodes.index)
    )

    post_out = dict(outdeg)
    post_in = dict(indeg)


    ds_v1 = (
        v1[
            v1["dataset"].eq(name)
        ]
    )

    for a in ds_v1.itertuples(index=False):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert post_out.get(s, 0) == 0
        assert post_in.get(q, 0) == 0

        post_out[s] = 1
        post_in[q] = 1


    raw_graph = runner.load_graph(
        P995 / f"{name}.geff"
    )

    raw_nodes = (
        raw_graph.node_attrs(
            attr_keys=[
                "node_id", "t", "z", "y", "x"
            ]
        )
        .to_pandas()
    )


    replay = DatasetReplay(
        name,
        TRAIN_ROOT,
        model,
        window_size,
        downsample,
        device,
    )


    for t in range(replay.T - 1):

        src_ids, tgt_ids, probs = (
            ir.fast_association_pair(
                replay,
                raw_nodes,
                t,
                model,
                device,
            )
        )

        if probs is None:
            continue


        src_idx = np.asarray(
            [
                i
                for i, s in enumerate(src_ids)
                if (
                    int(s) in final_ids
                    and post_out.get(int(s), 0) == 0
                )
            ],
            dtype=int,
        )

        tgt_idx = np.asarray(
            [
                j
                for j, q in enumerate(tgt_ids)
                if (
                    int(q) in final_ids
                    and post_in.get(int(q), 0) == 0
                )
            ],
            dtype=int,
        )

        if len(src_idx) == 0 or len(tgt_idx) == 0:
            continue


        sub = probs[
            np.ix_(
                src_idx,
                tgt_idx,
            )
        ]

        row_top = np.argmax(
            sub,
            axis=1,
        )

        col_top = np.argmax(
            sub,
            axis=0,
        )


        for a, b in enumerate(row_top):

            if int(col_top[b]) != a:
                continue


            i = int(src_idx[a])
            j = int(tgt_idx[b])

            s = int(src_ids[i])
            q = int(tgt_ids[j])

            p = float(probs[i, j])


            # Original full-matrix ranks.
            original_row_rank = int(
                1 + np.sum(
                    probs[i, :] > p
                )
            )

            original_col_rank = int(
                1 + np.sum(
                    probs[:, j] > p
                )
            )


            # Round-2 free-submatrix margins.
            row_vals = sub[a, :]

            if len(row_vals) >= 2:
                second_row = float(
                    np.partition(
                        row_vals,
                        -2,
                    )[-2]
                )
                row_margin = p - second_row
            else:
                row_margin = np.nan


            col_vals = sub[:, b]

            if len(col_vals) >= 2:
                second_col = float(
                    np.partition(
                        col_vals,
                        -2,
                    )[-2]
                )
                col_margin = p - second_col
            else:
                col_margin = np.nan


            src_xyz = (
                nodes.loc[
                    s,
                    ["z", "y", "x"]
                ]
                .to_numpy(dtype=float)
            )

            tgt_xyz = (
                nodes.loc[
                    q,
                    ["z", "y", "x"]
                ]
                .to_numpy(dtype=float)
            )

            distance_um = float(
                np.linalg.norm(
                    (tgt_xyz - src_xyz)
                    * scale
                )
            )


            records.append({
                "dataset":
                    name,

                "fold":
                    int(fold_map[name]),

                "t":
                    int(t),

                "source_pred_node_id":
                    s,

                "target_pred_node_id":
                    q,

                "candidate_prob":
                    p,

                "original_row_rank":
                    original_row_rank,

                "original_column_rank":
                    original_col_rank,

                "round2_row_margin":
                    float(row_margin),

                "round2_column_margin":
                    float(col_margin),

                "distance_um":
                    distance_um,

                "free_sources_in_frame":
                    int(len(src_idx)),

                "free_targets_in_frame":
                    int(len(tgt_idx)),
            })


    if (
        di == 1
        or di % 10 == 0
        or di == len(names)
    ):

        print(
            f"{di:3d}/103 | "
            f"candidates={len(records):,} | "
            f"{(time.time()-started)/60:.1f} min"
        )


    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ------------------------------------------------------------
# Contract
# ------------------------------------------------------------

universe = (
    pd.DataFrame(records)
    .sort_values(
        [
            "fold",
            "dataset",
            "t",
            "source_pred_node_id",
            "target_pred_node_id",
        ]
    )
    .reset_index(drop=True)
)


assert len(universe) > 0
assert universe["dataset"].nunique() == 103

assert (
    universe[
        [
            "dataset",
            "source_pred_node_id",
            "target_pred_node_id",
        ]
    ]
    .duplicated()
    .sum()
    == 0
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

elapsed = time.time() - started


print(
    "\n========== FULL ROUND-2 UNIVERSE =========="
)

print(
    "datasets:",
    universe["dataset"].nunique(),
    "/ 103"
)

print(
    "candidates:",
    len(universe)
)

print(
    "median / dataset:",
    f"{universe.groupby('dataset').size().median():.1f}"
)


print(
    "\n========== ROUND-2 PROBABILITY =========="
)

print(
    "q10/q25/q50/q75/q90/q95:",
    np.round(
        universe["candidate_prob"]
        .quantile(
            [.10, .25, .50, .75, .90, .95]
        )
        .to_numpy(),
        6,
    )
)


print(
    "\n========== ORIGINAL RANK DISTRIBUTION =========="
)

print(
    "row rank <=1/2/3/5:",
    [
        int(
            universe[
                "original_row_rank"
            ].le(k).sum()
        )
        for k in [1, 2, 3, 5]
    ]
)

print(
    "col rank <=1/2/3/5:",
    [
        int(
            universe[
                "original_column_rank"
            ].le(k).sum()
        )
        for k in [1, 2, 3, 5]
    ]
)


print(
    "\n========== ROUND-2 MARGINS =========="
)

for col in [
    "round2_row_margin",
    "round2_column_margin",
]:

    print(
        col,
        np.round(
            universe[col]
            .quantile(
                [.10, .25, .50, .75, .90]
            )
            .to_numpy(),
            6,
        )
    )


print(
    "\n========== DISTANCE UM =========="
)

print(
    "q10/q25/q50/q75/q90:",
    np.round(
        universe["distance_um"]
        .quantile(
            [.10, .25, .50, .75, .90]
        )
        .to_numpy(),
        4,
    )
)


print(
    "\n========== CANDIDATES BY FOLD =========="
)

print(
    universe.groupby("fold")
    .size()
    .to_string()
)


print(
    "\n========== RUNTIME =========="
)

print(
    "elapsed min:",
    f"{elapsed/60:.1f}"
)


# ------------------------------------------------------------
# Persist
# ------------------------------------------------------------

universe.to_pickle(
    OUT
)


payload = {
    "stage":
        "12.06B",

    "policy_predecessor":
        "MUTUAL-CONT-V1",

    "datasets":
        int(
            universe["dataset"].nunique()
        ),

    "candidates":
        int(len(universe)),

    "runtime_minutes":
        float(elapsed / 60),

    "gt_used":
        False,

    "features": [
        "candidate_prob",
        "original_row_rank",
        "original_column_rank",
        "round2_row_margin",
        "round2_column_margin",
        "distance_um",
    ],

    "candidate_counts_by_fold": {
        str(k): int(v)
        for k, v in (
            universe.groupby("fold")
            .size()
            .items()
        )
    },
}


SUMMARY_OUT.write_text(
    json.dumps(
        payload,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "\nwritten:",
    OUT
)

print(
    "written:",
    SUMMARY_OUT
)


print(
    "\n========== 12.06B DECISION =========="
)

print(
    "FULL_ROUND2_RECIPROCAL_UNIVERSE: PASS"
)