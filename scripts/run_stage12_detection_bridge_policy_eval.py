# ============================================================
# RUN — Stage 12.10F Detection Bridge Recovery Policy Eval
# Purpose:
#   Evaluate the PREDECLARED suppressed-node recovery family
#   on folds1–4 6bba development data:
#
#       CURRENT_V2
#       R1P10
#       R1P25
#       R1P50
#
#   Common recovery contract:
#       B12 geometry universe
#       incoming column rank == 1
#       outgoing column rank == 1
#       deterministic greedy one-to-one endpoint assignment
#
#   Only bridge_min_prob varies:
#       R1P10 >= 0.10
#       R1P25 >= 0.25
#       R1P50 >= 0.50
#
# Changes files: YES — Stage12 evaluation artifacts
# Runs training/inference: NO
# Uses GT: YES — folds1–4 development evaluation
# Keep after running: YES
# ============================================================

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

PROJECT = Path(
    "/Users/heqiuyan/Desktop/biohub-cell-tracking"
)

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
    / "stage12_folds14_bridge_association_evidence.pkl"
)

GRID_OUT = (
    S12
    / "stage12_detection_bridge_policy_grid.pkl"
)

DATASET_OUT = (
    S12
    / "stage12_detection_bridge_policy_grid_datasets.pkl"
)

ACTION_OUT = (
    S12
    / "stage12_detection_bridge_policy_actions.pkl"
)

SUMMARY_OUT = (
    S12
    / "stage12_detection_bridge_policy_grid_summary.json"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]


import run_stage11_multiframe_extract as runner

from biohub_cell_tracking import frozen_v9 as fv9


# ============================================================
# 1. Frozen constants
# ============================================================

SCALE = np.array(
    [
        1.625,
        0.40625,
        0.40625,
    ],
    dtype=np.float64,
)


POLICIES = [
    {
        "policy_id": "R1P10",
        "min_prob": 0.10,
    },
    {
        "policy_id": "R1P25",
        "min_prob": 0.25,
    },
    {
        "policy_id": "R1P50",
        "min_prob": 0.50,
    },
]


# GT-blind 12.10E counts.
EXPECTED_RAW_ELIGIBLE = {
    "R1P10": 1663,
    "R1P25": 1460,
    "R1P50": 807,
}


# Exact previously frozen Stage12 development baseline,
# displayed to 9 decimals.
EXPECTED_CURRENT_V2 = {
    "edge_jaccard":
        0.797717778,

    "adj_edge_jaccard":
        0.789766600,

    "division_jaccard":
        0.021739130,

    "score":
        0.791940513,
}


# ============================================================
# 2. Development corpus
# ============================================================

cv = pd.read_csv(
    PROJECT
    / "configs/cv_manifest.csv"
)


corp = (
    cv[
        cv["fold"].isin(
            [1, 2, 3, 4]
        )
        &
        cv[
            "dataset"
        ].str.startswith(
            "6bba_"
        )
    ]
    .sort_values(
        [
            "fold",
            "dataset",
        ]
    )
    .reset_index(
        drop=True
    )
)


assert len(corp) == 103
assert corp["dataset"].nunique() == 103


names = (
    corp[
        "dataset"
    ]
    .tolist()
)


fold_map = dict(
    zip(
        corp["dataset"],
        corp["fold"],
    )
)


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
# 3. Frozen Stage11 / Stage12 current graph actions
# ============================================================

morph = pd.read_pickle(
    S11
    / "stage11_deployable_final_v9_policy_grid_actions.pkl"
)

morph = (
    morph[
        morph[
            "config_id"
        ].eq(
            "K3_R3_P100"
        )
    ]
    .copy()
)


v1 = pd.read_pickle(
    S12
    / "stage12_folds14_mutual_best_universe.pkl"
)

v1 = (
    v1[
        v1[
            "candidate_prob"
        ].ge(
            0.25
        )
    ]
    .copy()
)

assert len(v1) == 46951


v2 = pd.read_pickle(
    S12
    / "stage12_folds14_round2_reciprocal_universe.pkl"
)

v2 = (
    v2[
        v2[
            "candidate_prob"
        ].ge(
            0.25
        )
    ]
    .copy()
)

assert len(v2) == 18511


# ============================================================
# 4. Locked suppressed-node evidence
# ============================================================

evidence = pd.read_pickle(
    EVIDENCE_PATH
)


assert len(evidence) == 15074
assert evidence["dataset"].nunique() == 102


assert (
    "6bba_2540cd90"
    not in
    set(
        evidence[
            "dataset"
        ]
    )
)


key_cols = [
    "dataset",
    "t",
    "z_ds",
    "y_ds",
    "x_ds",
]


assert (
    evidence[
        key_cols
    ]
    .duplicated()
    .sum()
    == 0
)


rank1 = (
    evidence[
        "in_col_rank"
    ].eq(1)
    &
    evidence[
        "out_col_rank"
    ].eq(1)
)


assert int(
    rank1.sum()
) == 1664


for spec in POLICIES:

    n = int(
        (
            rank1
            &
            evidence[
                "bridge_min_prob"
            ].ge(
                spec[
                    "min_prob"
                ]
            )
        ).sum()
    )

    assert (
        n
        ==
        EXPECTED_RAW_ELIGIBLE[
            spec[
                "policy_id"
            ]
        ]
    )


# ============================================================
# 5. Frozen V9 policies
# ============================================================

policies = (
    fv9.build_frozen_v9_policies(
        names,
        P995,
        P9975,
        TRAIN_ROOT,
        prefix_map=prefix_map,
        config=fv9.FROZEN_V9_CONFIG,
        progress=False,
    )
)


# ============================================================
# 6. Graph reconstruction helpers
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
        .set_index(
            "node_id"
        )
    )


    edges = (
        g.edge_attrs(
            unpack=True
        )
        .to_pandas()
    )


    outdeg = (
        edges.groupby(
            "source_id"
        )
        .size()
        .to_dict()
    )


    indeg = (
        edges.groupby(
            "target_id"
        )
        .size()
        .to_dict()
    )


    return (
        nodes,
        edges,
        outdeg,
        indeg,
    )


def add_existing_actions(
    g,
    nodes,
    table,
    name,
):

    aa = (
        table[
            table[
                "dataset"
            ].eq(
                name
            )
        ]
    )


    for r in aa.itertuples(
        index=False
    ):

        source_id = int(
            r.source_pred_node_id
        )

        target_id = int(
            r.target_pred_node_id
        )


        dist = (
            fv9.recovered_edge_dist(
                nodes,
                source_id,
                target_id,
                fv9.FROZEN_V9_CONFIG,
            )
        )


        g.add_edge(
            source_id,
            target_id,
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


def build_current_v2(
    name,
):

    g, _, _ = (
        fv9.apply_frozen_v9(
            name,
            P995,
            P9975,
            policies,
            TRAIN_ROOT,
            config=
                fv9.FROZEN_V9_CONFIG,
        )
    )


    nodes, _, _, _ = (
        graph_state(
            g
        )
    )


    # Exact promoted order.
    add_existing_actions(
        g,
        nodes,
        morph,
        name,
    )

    add_existing_actions(
        g,
        nodes,
        v1,
        name,
    )

    add_existing_actions(
        g,
        nodes,
        v2,
        name,
    )


    return g


# ============================================================
# 7. Detection recovery policy
# ============================================================

def eligible_candidates(
    name,
    min_prob,
):

    aa = (
        evidence[
            evidence[
                "dataset"
            ].eq(
                name
            )
            &
            evidence[
                "in_col_rank"
            ].eq(1)
            &
            evidence[
                "out_col_rank"
            ].eq(1)
            &
            evidence[
                "bridge_min_prob"
            ].ge(
                min_prob
            )
        ]
        .copy()
    )


    # Deterministic priority.
    aa = (
        aa.sort_values(
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
        .reset_index(
            drop=True
        )
    )


    return aa


def physical_distance(
    a,
    b,
):

    a_xyz = np.array(
        [
            float(a["z"]),
            float(a["y"]),
            float(a["x"]),
        ],
        dtype=np.float64,
    )


    b_xyz = np.array(
        [
            float(b["z"]),
            float(b["y"]),
            float(b["x"]),
        ],
        dtype=np.float64,
    )


    return float(
        np.linalg.norm(
            (
                a_xyz
                -
                b_xyz
            )
            *
            SCALE
        )
    )


def apply_bridge_policy(
    g,
    name,
    policy_id,
    min_prob,
):

    candidates = (
        eligible_candidates(
            name,
            min_prob,
        )
    )


    nodes, _, outdeg, indeg = (
        graph_state(
            g
        )
    )


    used_sources = set()
    used_targets = set()


    selected = []


    for r in candidates.itertuples(
        index=False
    ):

        source_id = int(
            r.nearest_source_id
        )

        target_id = int(
            r.nearest_target_id
        )


        # These were defined as free endpoints in the frozen
        # current-V2 geometry universe.
        assert (
            outdeg.get(
                source_id,
                0,
            )
            == 0
        )

        assert (
            indeg.get(
                target_id,
                0,
            )
            == 0
        )


        # Deterministic one-to-one recovery.
        if (
            source_id
            in used_sources
        ):
            continue


        if (
            target_id
            in used_targets
        ):
            continue


        src = nodes.loc[
            source_id
        ]

        tgt = nodes.loc[
            target_id
        ]


        assert (
            int(src["t"])
            ==
            int(r.t) - 1
        )

        assert (
            int(tgt["t"])
            ==
            int(r.t) + 1
        )


        candidate_attrs = {
            "t":
                int(r.t),

            "z":
                float(r.z),

            "y":
                float(r.y),

            "x":
                float(r.x),
        }


        in_dist = (
            physical_distance(
                src,
                candidate_attrs,
            )
        )


        out_dist = (
            physical_distance(
                candidate_attrs,
                tgt,
            )
        )


        # Must reproduce frozen 12.10D geometry.
        assert np.isclose(
            in_dist,
            float(
                r.src_dist_um
            ),
            atol=1e-6,
        )


        assert np.isclose(
            out_dist,
            float(
                r.tgt_dist_um
            ),
            atol=1e-6,
        )


        new_node_id = (
            g.add_node(
                candidate_attrs,
                validate_keys=True,
            )
        )


        g.add_edge(
            source_id,
            new_node_id,
            {
                "edge_dist":
                    float(
                        in_dist
                    ),

                "edge_prob":
                    float(
                        r.in_prob
                    ),
            },
            validate_keys=True,
        )


        g.add_edge(
            new_node_id,
            target_id,
            {
                "edge_dist":
                    float(
                        out_dist
                    ),

                "edge_prob":
                    float(
                        r.out_prob
                    ),
            },
            validate_keys=True,
        )


        used_sources.add(
            source_id
        )

        used_targets.add(
            target_id
        )


        selected.append({
            "policy_id":
                policy_id,

            "dataset":
                name,

            "fold":
                int(
                    fold_map[
                        name
                    ]
                ),

            "source_id":
                source_id,

            "target_id":
                target_id,

            "new_node_id":
                int(
                    new_node_id
                ),

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
                float(
                    r.in_prob
                ),

            "out_prob":
                float(
                    r.out_prob
                ),

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


    return (
        selected,
        len(candidates),
    )


# ============================================================
# 8. Official metric evaluation
# ============================================================

def evaluate_prediction(
    g,
    name,
):

    # Load GT fresh because official metric evaluation mutates
    # graph-side schema.
    gt = runner.load_graph(
        TRAIN_ROOT
        / f"{name}.geff"
    )


    result = fv9.evaluate_graph(
        g,
        gt,
        SCALE,
        float(
            estimated_map[
                name
            ]
        ),
        config=
            fv9.FROZEN_V9_CONFIG,
    )


    return result


# ============================================================
# 9. Development evaluation
# ============================================================

metric_rows = []
action_rows = []

started = time.time()


for di, name in enumerate(
    names,
    start=1,
):

    fold = int(
        fold_map[
            name
        ]
    )


    # --------------------------------------------------------
    # CURRENT V2 baseline
    # --------------------------------------------------------

    g = build_current_v2(
        name
    )


    baseline_metrics = (
        evaluate_prediction(
            g,
            name,
        )
    )


    baseline_metrics.update({
        "dataset":
            name,

        "fold":
            fold,

        "policy_id":
            "CURRENT_V2",

        "raw_eligible":
            0,

        "accepted_actions":
            0,
    })


    metric_rows.append(
        baseline_metrics
    )


    # --------------------------------------------------------
    # Predeclared recovery family
    # --------------------------------------------------------

    for spec in POLICIES:

        policy_id = (
            spec[
                "policy_id"
            ]
        )

        min_prob = float(
            spec[
                "min_prob"
            ]
        )


        # Fresh graph for every policy. Never evaluate then
        # continue mutating an already metric-mutated graph.
        g = build_current_v2(
            name
        )


        selected, raw_eligible = (
            apply_bridge_policy(
                g,
                name,
                policy_id,
                min_prob,
            )
        )


        metrics = (
            evaluate_prediction(
                g,
                name,
            )
        )


        metrics.update({
            "dataset":
                name,

            "fold":
                fold,

            "policy_id":
                policy_id,

            "raw_eligible":
                int(
                    raw_eligible
                ),

            "accepted_actions":
                int(
                    len(
                        selected
                    )
                ),
        })


        metric_rows.append(
            metrics
        )


        action_rows.extend(
            selected
        )


    if (
        di == 1
        or di % 10 == 0
        or di == 103
    ):

        print(
            f"{di:3d}/103 | "
            f"{name} | "
            f"{(time.time()-started)/60:.1f} min"
        )


# ============================================================
# 10. Assemble results
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)

actions_df = pd.DataFrame(
    action_rows
)


assert len(metrics_df) == (
    103 * 4
)

assert (
    metrics_df[
        "dataset"
    ].nunique()
    == 103
)


for policy_id in [
    "CURRENT_V2",
    "R1P10",
    "R1P25",
    "R1P50",
]:

    assert (
        metrics_df[
            metrics_df[
                "policy_id"
            ].eq(
                policy_id
            )
        ][
            "dataset"
        ]
        .nunique()
        == 103
    )


# ============================================================
# 11. Aggregate summaries
# ============================================================

aggregate_rows = []


for policy_id in [
    "CURRENT_V2",
    "R1P10",
    "R1P25",
    "R1P50",
]:

    rr = (
        metrics_df[
            metrics_df[
                "policy_id"
            ].eq(
                policy_id
            )
        ]
        .to_dict(
            "records"
        )
    )


    summary = (
        fv9.summarise(
            rr
        )
    )


    summary[
        "policy_id"
    ] = policy_id


    summary[
        "raw_eligible"
    ] = int(
        metrics_df[
            metrics_df[
                "policy_id"
            ].eq(
                policy_id
            )
        ][
            "raw_eligible"
        ].sum()
    )


    summary[
        "accepted_actions"
    ] = int(
        metrics_df[
            metrics_df[
                "policy_id"
            ].eq(
                policy_id
            )
        ][
            "accepted_actions"
        ].sum()
    )


    aggregate_rows.append(
        summary
    )


grid = pd.DataFrame(
    aggregate_rows
)


baseline = (
    grid[
        grid[
            "policy_id"
        ].eq(
            "CURRENT_V2"
        )
    ]
    .iloc[0]
)


grid[
    "delta_score"
] = (
    grid[
        "score"
    ]
    -
    float(
        baseline[
            "score"
        ]
    )
)


grid[
    "delta_edge_jaccard"
] = (
    grid[
        "edge_jaccard"
    ]
    -
    float(
        baseline[
            "edge_jaccard"
        ]
    )
)


grid[
    "delta_adj_edge_jaccard"
] = (
    grid[
        "adj_edge_jaccard"
    ]
    -
    float(
        baseline[
            "adj_edge_jaccard"
        ]
    )
)


grid[
    "delta_node_recall"
] = (
    grid[
        "node_recall"
    ]
    -
    float(
        baseline[
            "node_recall"
        ]
    )
)


# ============================================================
# 12. Hard baseline reproduction
# ============================================================

assert int(
    baseline[
        "n"
    ]
) == 103


for key, expected in (
    EXPECTED_CURRENT_V2.items()
):

    actual = float(
        baseline[
            key
        ]
    )

    assert np.isclose(
        actual,
        expected,
        atol=5e-7,
        rtol=0.0,
    ), (
        key,
        actual,
        expected,
    )


# Raw GT-blind eligibility must still exactly reproduce 12.10E.
for spec in POLICIES:

    policy_id = (
        spec[
            "policy_id"
        ]
    )

    row = (
        grid[
            grid[
                "policy_id"
            ].eq(
                policy_id
            )
        ]
        .iloc[0]
    )


    assert int(
        row[
            "raw_eligible"
        ]
    ) == (
        EXPECTED_RAW_ELIGIBLE[
            policy_id
        ]
    )


# ============================================================
# 13. Per-fold summaries
# ============================================================

fold_rows = []


for fold in [
    1,
    2,
    3,
    4,
]:

    baseline_fold_rows = (
        metrics_df[
            metrics_df[
                "fold"
            ].eq(
                fold
            )
            &
            metrics_df[
                "policy_id"
            ].eq(
                "CURRENT_V2"
            )
        ]
        .to_dict(
            "records"
        )
    )


    baseline_fold = (
        fv9.summarise(
            baseline_fold_rows
        )
    )


    for policy_id in [
        "R1P10",
        "R1P25",
        "R1P50",
    ]:

        rr = (
            metrics_df[
                metrics_df[
                    "fold"
                ].eq(
                    fold
                )
                &
                metrics_df[
                    "policy_id"
                ].eq(
                    policy_id
                )
            ]
            .to_dict(
                "records"
            )
        )


        ss = (
            fv9.summarise(
                rr
            )
        )


        fold_rows.append({
            "fold":
                fold,

            "policy_id":
                policy_id,

            "baseline_score":
                float(
                    baseline_fold[
                        "score"
                    ]
                ),

            "score":
                float(
                    ss[
                        "score"
                    ]
                ),

            "delta_score":
                float(
                    ss[
                        "score"
                    ]
                    -
                    baseline_fold[
                        "score"
                    ]
                ),

            "delta_edge_jaccard":
                float(
                    ss[
                        "edge_jaccard"
                    ]
                    -
                    baseline_fold[
                        "edge_jaccard"
                    ]
                ),

            "delta_adj_edge_jaccard":
                float(
                    ss[
                        "adj_edge_jaccard"
                    ]
                    -
                    baseline_fold[
                        "adj_edge_jaccard"
                    ]
                ),

            "delta_node_recall":
                float(
                    ss[
                        "node_recall"
                    ]
                    -
                    baseline_fold[
                        "node_recall"
                    ]
                ),
        })


fold_df = pd.DataFrame(
    fold_rows
)


# ============================================================
# 14. Deterministic action hashes
# ============================================================

action_hashes = {}


for policy_id in [
    "R1P10",
    "R1P25",
    "R1P50",
]:

    aa = (
        actions_df[
            actions_df[
                "policy_id"
            ].eq(
                policy_id
            )
        ][
            [
                "dataset",
                "source_id",
                "target_id",
                "t",
                "z",
                "y",
                "x",
                "bridge_min_prob",
            ]
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
    )


    payload = (
        aa.to_dict(
            "records"
        )
    )


    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    ).encode(
        "utf-8"
    )


    action_hashes[
        policy_id
    ] = (
        hashlib.sha256(
            blob
        ).hexdigest()
    )


# ============================================================
# 15. Report
# ============================================================

print(
    "\n========== CURRENT V2 REPRODUCTION =========="
)

print(
    "edge_jaccard:",
    f"{float(baseline['edge_jaccard']):.9f}",
)

print(
    "adj_edge_jaccard:",
    f"{float(baseline['adj_edge_jaccard']):.9f}",
)

print(
    "division_jaccard:",
    f"{float(baseline['division_jaccard']):.9f}",
)

print(
    "node_recall:",
    f"{float(baseline['node_recall']):.9f}",
)

print(
    "score:",
    f"{float(baseline['score']):.9f}",
)


print(
    "\n========== PREDECLARED POLICY GRID =========="
)

cols = [
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

print(
    grid[
        cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


print(
    "\n========== POLICY DELTAS BY FOLD =========="
)

print(
    fold_df[
        [
            "fold",
            "policy_id",
            "delta_score",
            "delta_edge_jaccard",
            "delta_adj_edge_jaccard",
            "delta_node_recall",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.9f}",
    )
)


print(
    "\n========== ACTION HASHES =========="
)

for policy_id in [
    "R1P10",
    "R1P25",
    "R1P50",
]:

    print(
        policy_id,
        action_hashes[
            policy_id
        ],
    )


print(
    "\n========== ACTION CONFLICT LOSS =========="
)

for policy_id in [
    "R1P10",
    "R1P25",
    "R1P50",
]:

    row = (
        grid[
            grid[
                "policy_id"
            ].eq(
                policy_id
            )
        ]
        .iloc[0]
    )


    raw_n = int(
        row[
            "raw_eligible"
        ]
    )

    accepted_n = int(
        row[
            "accepted_actions"
        ]
    )


    print(
        f"{policy_id}: "
        f"{raw_n:,} -> "
        f"{accepted_n:,} "
        f"(dropped {raw_n-accepted_n:,})"
    )


# ============================================================
# 16. Persist
# ============================================================

grid.to_pickle(
    GRID_OUT
)

metrics_df.to_pickle(
    DATASET_OUT
)

actions_df.to_pickle(
    ACTION_OUT
)


summary_payload = {
    "stage":
        "12.10F",

    "gt_used":
        True,

    "development_datasets":
        103,

    "corpus":
        "folds1-4 6bba",

    "baseline":
        "CURRENT_V2",

    "predeclared_policies": {
        "R1P10": {
            "incoming_col_rank":
                1,

            "outgoing_col_rank":
                1,

            "bridge_min_prob":
                0.10,
        },

        "R1P25": {
            "incoming_col_rank":
                1,

            "outgoing_col_rank":
                1,

            "bridge_min_prob":
                0.25,
        },

        "R1P50": {
            "incoming_col_rank":
                1,

            "outgoing_col_rank":
                1,

            "bridge_min_prob":
                0.50,
        },
    },

    "selection":
        (
            "deterministic greedy descending "
            "bridge_min_prob; one-to-one existing "
            "free source and target endpoints"
        ),

    "baseline_reproduced":
        True,

    "action_hashes":
        action_hashes,

    "aggregate":
        grid.to_dict(
            "records"
        ),

    "folds":
        fold_df.to_dict(
            "records"
        ),
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
    "\n========== 12.10F DECISION =========="
)

print(
    "PREDECLARED_DETECTION_BRIDGE_GRID_EVALUATED: PASS"
)