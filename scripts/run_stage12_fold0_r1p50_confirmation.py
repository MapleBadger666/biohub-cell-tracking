# ============================================================
# RUN — Stage 12.10G4 Fold0 R1P50 Confirmation
# Purpose:
#   Independently confirm the already-locked R1P50 suppressed-
#   node recovery policy on the untouched Fold0 6bba corpus.
#
#   Locked policy:
#       B12
#       in_col_rank == 1
#       out_col_rank == 1
#       bridge_min_prob >= 0.50
#       deterministic greedy one-to-one endpoints
#
#   No alternative threshold is evaluated.
#
# Changes files: YES — Fold0 confirmation artifacts
# Runs training/inference: NO
# Uses GT: YES — Fold0 confirmation
# Keep after running: YES
# ============================================================

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

TRAIN_ROOT = (
    PROJECT
    / "data/raw/competition/"
      "biohub-cell-tracking-during-development/train"
)

P995 = (
    PROJECT
    / "predictions/heqiuyan/"
      "stage10_infer_baseline_256_det995/split_0"
)

P9975 = (
    PROJECT
    / "predictions/heqiuyan/"
      "stage10_infer_baseline_256_det9975/split_0"
)

S11 = (
    PROJECT
    / "reports/stage11_division_probe"
)

S12 = (
    PROJECT
    / "reports/stage12_residual_audit"
)

EVIDENCE_PATH = (
    S12
    / "stage12_fold0_bridge_association_evidence.pkl"
)

REFERENCE_PATH = (
    S12
    / "stage12_fold0_mutual_cont_v2_confirmatory_datasets.pkl"
)

GRID_OUT = (
    S12
    / "stage12_fold0_r1p50_confirmation_grid.pkl"
)

DATASET_OUT = (
    S12
    / "stage12_fold0_r1p50_confirmation_datasets.pkl"
)

ACTION_OUT = (
    S12
    / "stage12_fold0_r1p50_confirmation_actions.pkl"
)

SUMMARY_OUT = (
    S12
    / "stage12_fold0_r1p50_confirmation_summary.json"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]

import run_stage11_multiframe_extract as runner

from biohub_cell_tracking import frozen_v9 as fv9


SCALE = np.asarray(
    [1.625, 0.40625, 0.40625],
    dtype=np.float64,
)

MIN_PROB = 0.50

EXPECTED_RANK1 = 358
EXPECTED_RAW_ELIGIBLE = 152

DEV_ACTION_SHA = (
    "2d80d9b1cda3c4f0593aced5e8010051"
    "cc49500ef64868541365737dcbe1592f"
)


# ============================================================
# 1. Locked Fold0 corpus
# ============================================================

cv = pd.read_csv(
    PROJECT
    / "configs/cv_manifest.csv"
)

corp = (
    cv[
        cv["fold"].eq(0)
        &
        cv["dataset"].str.startswith("6bba_")
    ]
    .sort_values("dataset")
    .reset_index(drop=True)
)

assert len(corp) == 25
assert corp["dataset"].nunique() == 25

names = corp["dataset"].tolist()

estimated_map = dict(
    zip(
        corp["dataset"],
        corp["estimated_nodes"],
    )
)

prefix_map = {
    name: "6bba"
    for name in names
}


# ============================================================
# 2. Exact previously locked Fold0 CURRENT_V2 actions
# ============================================================

morph = pd.read_pickle(
    S11
    / "stage11_fold0_morphdiv_v2_actions_locked.pkl"
)

v1 = pd.read_pickle(
    S12
    / "stage12_fold0_mutual_cont_v1_actions_locked.pkl"
)

v2 = pd.read_pickle(
    S12
    / "stage12_fold0_mutual_cont_v2_actions_locked.pkl"
)

assert len(morph) == 34
assert len(v1) == 10107
assert len(v2) == 4076


# ============================================================
# 3. Frozen R1P50 evidence
# ============================================================

evidence = pd.read_pickle(
    EVIDENCE_PATH
)

assert len(evidence) == 3174
assert evidence["dataset"].nunique() == 25

rank1 = (
    evidence["in_col_rank"].eq(1)
    &
    evidence["out_col_rank"].eq(1)
)

assert int(rank1.sum()) == EXPECTED_RANK1

eligible = (
    evidence[
        rank1
        &
        evidence["bridge_min_prob"].ge(
            MIN_PROB
        )
    ]
    .copy()
)

assert len(eligible) == EXPECTED_RAW_ELIGIBLE


# ============================================================
# 4. Frozen V9 policies
# ============================================================

policies = fv9.build_frozen_v9_policies(
    names,
    P995,
    P9975,
    TRAIN_ROOT,
    prefix_map=prefix_map,
    config=fv9.FROZEN_V9_CONFIG,
    progress=False,
)


# ============================================================
# 5. Graph helpers
# ============================================================

def graph_state(g):

    nodes = (
        g.node_attrs(
            attr_keys=[
                "node_id",
                "t",
                "z",
                "y",
                "x",
            ]
        )
        .to_pandas()
        .set_index("node_id")
    )

    edges = (
        g.edge_attrs(
            unpack=True
        )
        .to_pandas()
    )

    outdeg = (
        edges.groupby("source_id")
        .size()
        .to_dict()
    )

    indeg = (
        edges.groupby("target_id")
        .size()
        .to_dict()
    )

    return nodes, edges, outdeg, indeg


def add_locked_actions(
    g,
    nodes,
    table,
    name,
):

    aa = table[
        table["dataset"].eq(name)
    ]

    for r in aa.itertuples(
        index=False
    ):

        s = int(
            r.source_pred_node_id
        )

        q = int(
            r.target_pred_node_id
        )

        dist = fv9.recovered_edge_dist(
            nodes,
            s,
            q,
            fv9.FROZEN_V9_CONFIG,
        )

        g.add_edge(
            s,
            q,
            {
                "edge_dist":
                    float(dist),

                "edge_prob":
                    float(
                        r.candidate_prob
                    ),
            },
            validate_keys=True,
        )


def build_current_v2(name):

    g, _, _ = fv9.apply_frozen_v9(
        name,
        P995,
        P9975,
        policies,
        TRAIN_ROOT,
        config=fv9.FROZEN_V9_CONFIG,
    )

    nodes, _, _, _ = graph_state(g)

    add_locked_actions(
        g,
        nodes,
        morph,
        name,
    )

    add_locked_actions(
        g,
        nodes,
        v1,
        name,
    )

    add_locked_actions(
        g,
        nodes,
        v2,
        name,
    )

    return g


def physical_distance(
    a,
    b,
):

    aa = np.asarray(
        [
            float(a["z"]),
            float(a["y"]),
            float(a["x"]),
        ]
    )

    bb = np.asarray(
        [
            float(b["z"]),
            float(b["y"]),
            float(b["x"]),
        ]
    )

    return float(
        np.linalg.norm(
            (aa - bb) * SCALE
        )
    )


# ============================================================
# 6. Locked R1P50 application
# ============================================================

def apply_r1p50(
    g,
    name,
):

    aa = (
        eligible[
            eligible["dataset"].eq(name)
        ]
        .sort_values(
            [
                "bridge_min_prob",
                "detector_prob",
                "midpoint_residual_um",
                "t",
                "z_ds",
                "y_ds",
                "x_ds",
                "nearest_source_id",
                "nearest_target_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    nodes, _, outdeg, indeg = (
        graph_state(g)
    )

    used_sources = set()
    used_targets = set()

    selected = []


    for r in aa.itertuples(
        index=False
    ):

        s = int(
            r.nearest_source_id
        )

        q = int(
            r.nearest_target_id
        )

        # Evidence was constructed on CURRENT_V2.
        assert outdeg.get(s, 0) == 0
        assert indeg.get(q, 0) == 0

        if s in used_sources:
            continue

        if q in used_targets:
            continue


        src = nodes.loc[s]
        tgt = nodes.loc[q]

        assert int(src["t"]) == int(r.t) - 1
        assert int(tgt["t"]) == int(r.t) + 1


        attrs = {
            "t": int(r.t),
            "z": float(r.z),
            "y": float(r.y),
            "x": float(r.x),
        }


        in_dist = physical_distance(
            src,
            attrs,
        )

        out_dist = physical_distance(
            attrs,
            tgt,
        )


        assert np.isclose(
            in_dist,
            float(r.src_dist_um),
            atol=1e-6,
        )

        assert np.isclose(
            out_dist,
            float(r.tgt_dist_um),
            atol=1e-6,
        )


        new_id = g.add_node(
            attrs,
            validate_keys=True,
        )


        g.add_edge(
            s,
            new_id,
            {
                "edge_dist":
                    in_dist,

                "edge_prob":
                    float(r.in_prob),
            },
            validate_keys=True,
        )


        g.add_edge(
            new_id,
            q,
            {
                "edge_dist":
                    out_dist,

                "edge_prob":
                    float(r.out_prob),
            },
            validate_keys=True,
        )


        used_sources.add(s)
        used_targets.add(q)


        selected.append({
            "dataset":
                name,

            "source_id":
                s,

            "target_id":
                q,

            "new_node_id":
                int(new_id),

            "t":
                int(r.t),

            "z":
                float(r.z),

            "y":
                float(r.y),

            "x":
                float(r.x),

            "detector_prob":
                float(
                    r.detector_prob
                ),

            "bridge_min_prob":
                float(
                    r.bridge_min_prob
                ),

            "in_prob":
                float(r.in_prob),

            "out_prob":
                float(r.out_prob),

            "src_dist_um":
                float(
                    r.src_dist_um
                ),

            "tgt_dist_um":
                float(
                    r.tgt_dist_um
                ),

            "midpoint_residual_um":
                float(
                    r.midpoint_residual_um
                ),
        })


    return selected, len(aa)


# ============================================================
# 7. Official GT evaluation
# ============================================================

def evaluate_graph(
    g,
    name,
):

    # Fresh GT each time: official evaluator mutates schema.
    gt = runner.load_graph(
        TRAIN_ROOT
        / f"{name}.geff"
    )

    return fv9.evaluate_graph(
        g,
        gt,
        SCALE,
        float(
            estimated_map[name]
        ),
        config=fv9.FROZEN_V9_CONFIG,
    )


# ============================================================
# 8. Confirmation
# ============================================================

rows = []
action_rows = []

started = time.time()


for di, name in enumerate(
    names,
    start=1,
):

    # Baseline CURRENT_V2.
    g0 = build_current_v2(
        name
    )

    base = evaluate_graph(
        g0,
        name,
    )

    base.update({
        "dataset":
            name,

        "policy_id":
            "CURRENT_V2",

        "raw_eligible":
            0,

        "accepted_actions":
            0,
    })

    rows.append(base)


    # Locked R1P50 only.
    g1 = build_current_v2(
        name
    )

    selected, raw_n = apply_r1p50(
        g1,
        name,
    )

    new = evaluate_graph(
        g1,
        name,
    )

    new.update({
        "dataset":
            name,

        "policy_id":
            "R1P50",

        "raw_eligible":
            int(raw_n),

        "accepted_actions":
            int(len(selected)),
    })

    rows.append(new)

    action_rows.extend(
        selected
    )


    if (
        di == 1
        or di % 5 == 0
        or di == 25
    ):

        print(
            f"{di:2d}/25 | "
            f"{name} | "
            f"{(time.time()-started)/60:.1f} min"
        )


metrics = pd.DataFrame(
    rows
)

actions = pd.DataFrame(
    action_rows
)


assert len(metrics) == 50

assert (
    metrics[
        metrics["policy_id"].eq(
            "CURRENT_V2"
        )
    ]["dataset"].nunique()
    == 25
)

assert (
    metrics[
        metrics["policy_id"].eq(
            "R1P50"
        )
    ]["dataset"].nunique()
    == 25
)


# ============================================================
# 9. Hard CURRENT_V2 reproduction against frozen Fold0 V2
# ============================================================

reference = pd.read_pickle(
    REFERENCE_PATH
)

assert len(reference) == 25

reference = reference.set_index(
    "dataset"
)


base_df = (
    metrics[
        metrics["policy_id"].eq(
            "CURRENT_V2"
        )
    ]
    .set_index("dataset")
)


comparison = [
    ("edge_tp", "new_edge_tp"),
    ("edge_fp", "new_edge_fp"),
    ("edge_fn", "new_edge_fn"),
    ("division_tp", "new_division_tp"),
    ("division_fp", "new_division_fp"),
    ("division_fn", "new_division_fn"),
]


for actual_col, ref_col in comparison:

    assert np.array_equal(
        base_df.loc[
            reference.index,
            actual_col,
        ].to_numpy(),
        reference[
            ref_col
        ].to_numpy(),
    ), (
        actual_col,
        ref_col,
    )


float_comparison = [
    ("node_recall", "new_node_recall"),
    ("edge_jaccard", "new_edge_jaccard"),
    (
        "adj_edge_jaccard",
        "new_adj_edge_jaccard",
    ),
]


for actual_col, ref_col in float_comparison:

    assert np.allclose(
        base_df.loc[
            reference.index,
            actual_col,
        ].to_numpy(
            dtype=float
        ),
        reference[
            ref_col
        ].to_numpy(
            dtype=float
        ),
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    ), (
        actual_col,
        ref_col,
    )


# ============================================================
# 10. Aggregate
# ============================================================

base_summary = fv9.summarise(
    metrics[
        metrics["policy_id"].eq(
            "CURRENT_V2"
        )
    ].to_dict(
        "records"
    )
)

r1_summary = fv9.summarise(
    metrics[
        metrics["policy_id"].eq(
            "R1P50"
        )
    ].to_dict(
        "records"
    )
)


raw_total = int(
    metrics[
        metrics["policy_id"].eq(
            "R1P50"
        )
    ]["raw_eligible"].sum()
)

accepted_total = int(
    metrics[
        metrics["policy_id"].eq(
            "R1P50"
        )
    ]["accepted_actions"].sum()
)


assert raw_total == 152


grid = pd.DataFrame([
    {
        "policy_id":
            "CURRENT_V2",

        "raw_eligible":
            0,

        "accepted_actions":
            0,

        **base_summary,
    },
    {
        "policy_id":
            "R1P50",

        "raw_eligible":
            raw_total,

        "accepted_actions":
            accepted_total,

        **r1_summary,
    },
])


for col in [
    "edge_jaccard",
    "adj_edge_jaccard",
    "division_jaccard",
    "node_recall",
    "score",
]:

    grid[
        f"delta_{col}"
    ] = (
        grid[col]
        -
        float(
            grid.iloc[0][col]
        )
    )


# ============================================================
# 11. Per-dataset confirmation anatomy
# ============================================================

base_idx = (
    metrics[
        metrics["policy_id"].eq(
            "CURRENT_V2"
        )
    ]
    .set_index("dataset")
)

new_idx = (
    metrics[
        metrics["policy_id"].eq(
            "R1P50"
        )
    ]
    .set_index("dataset")
)


delta_adj = (
    new_idx[
        "adj_edge_jaccard"
    ]
    -
    base_idx[
        "adj_edge_jaccard"
    ]
)


improved = int(
    (delta_adj > 1e-15).sum()
)

worsened = int(
    (delta_adj < -1e-15).sum()
)

unchanged = int(
    (
        delta_adj.abs()
        <= 1e-15
    ).sum()
)


# ============================================================
# 12. Deterministic Fold0 action hash
# ============================================================

hash_cols = [
    "dataset",
    "source_id",
    "target_id",
    "t",
    "z",
    "y",
    "x",
    "bridge_min_prob",
]

payload_rows = (
    actions[
        hash_cols
    ]
    .sort_values(
        [
            "dataset",
            "t",
            "z",
            "y",
            "x",
            "source_id",
            "target_id",
        ]
    )
    .to_dict(
        "records"
    )
)


blob = json.dumps(
    payload_rows,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")


action_sha = hashlib.sha256(
    blob
).hexdigest()


# ============================================================
# 13. Report
# ============================================================

print(
    "\n========== FOLD0 CURRENT_V2 REPRODUCTION =========="
)

print(
    "datasets reproduced:",
    25,
    "/ 25",
)

print(
    "edge_jaccard:",
    f"{base_summary['edge_jaccard']:.9f}",
)

print(
    "adj_edge_jaccard:",
    f"{base_summary['adj_edge_jaccard']:.9f}",
)

print(
    "division_jaccard:",
    f"{base_summary['division_jaccard']:.9f}",
)

print(
    "node_recall:",
    f"{base_summary['node_recall']:.9f}",
)

print(
    "score:",
    f"{base_summary['score']:.9f}",
)


print(
    "\n========== LOCKED R1P50 CONFIRMATION =========="
)

print(
    grid[
        [
            "policy_id",
            "raw_eligible",
            "accepted_actions",
            "edge_jaccard",
            "adj_edge_jaccard",
            "division_jaccard",
            "node_recall",
            "score",
            "delta_score",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


print(
    "\n========== CONFIRMATION ANATOMY =========="
)

print(
    "raw eligible:",
    raw_total,
)

print(
    "accepted actions:",
    accepted_total,
)

print(
    "conflict drops:",
    raw_total - accepted_total,
)

print(
    "datasets improved (adj edge):",
    improved,
)

print(
    "datasets worsened (adj edge):",
    worsened,
)

print(
    "datasets unchanged:",
    unchanged,
)


print(
    "\n========== ACTION HASH =========="
)

print(
    "development R1P50:",
    DEV_ACTION_SHA,
)

print(
    "Fold0 R1P50:",
    action_sha,
)


# ============================================================
# 14. Persist
# ============================================================

grid.to_pickle(
    GRID_OUT
)

metrics.to_pickle(
    DATASET_OUT
)

actions.to_pickle(
    ACTION_OUT
)


summary_payload = {
    "stage":
        "12.10G4",

    "gt_used":
        True,

    "confirmation":
        "Fold0 6bba",

    "policy":
        "R1P50",

    "policy_locked_before_fold0_gt":
        True,

    "bridge_min_prob":
        0.50,

    "incoming_col_rank":
        1,

    "outgoing_col_rank":
        1,

    "raw_eligible":
        raw_total,

    "accepted_actions":
        accepted_total,

    "action_sha256":
        action_sha,

    "development_action_sha256":
        DEV_ACTION_SHA,

    "baseline_reproduced":
        True,

    "baseline":
        base_summary,

    "r1p50":
        r1_summary,

    "delta_score":
        float(
            r1_summary["score"]
            -
            base_summary["score"]
        ),

    "improved_datasets_adj_edge":
        improved,

    "worsened_datasets_adj_edge":
        worsened,

    "unchanged_datasets_adj_edge":
        unchanged,
}


SUMMARY_OUT.write_text(
    json.dumps(
        summary_payload,
        indent=2,
        allow_nan=True,
    ),
    encoding="utf-8",
)


print(
    "\nwritten:",
    GRID_OUT
)

print(
    "written:",
    DATASET_OUT
)

print(
    "written:",
    ACTION_OUT
)

print(
    "written:",
    SUMMARY_OUT
)


print(
    "\n========== RUNTIME =========="
)

print(
    "elapsed min:",
    f"{(time.time()-started)/60:.1f}"
)


print(
    "\n========== 12.10G4 DECISION =========="
)

print(
    "LOCKED_R1P50_FOLD0_CONFIRMATION_EVALUATED: PASS"
)