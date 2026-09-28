# ============================================================
# RESULT — Stage 12.11A CURRENT_BEST Edge Residual Census
# Purpose:
#   Decompose every official edge FN in the frozen Stage12 CURRENT_BEST
#   40-dataset Fold0 pipeline into:
#
#       DETECTION_LIMITED
#           one or both GT edge endpoints have no matched
#           predicted node
#
#       LINKING_LIMITED
#           both GT endpoints already exist as matched predicted
#           nodes, but their correct edge is absent
#
#   Also classify exact official FPs using source-locked:
#
#       pred_valid == True
#       matched_edge_mask == False
#
#   No metric logic is reimplemented.
#
# Changes files: YES
#   reports/stage12_residual_audit/
#       stage12_current_best_edge_fn_residuals.pkl
#       stage12_current_best_edge_fp_residuals.pkl
#       stage12_current_best_edge_residual_datasets.pkl
#       stage12_current_best_edge_residual_summary.json
#
# Runs training/inference: NO
# Uses GT: YES — residual research audit
# Changes frozen Stage11 policy: NO
# Keep after running: YES
# ============================================================

from pathlib import Path
import hashlib
import json
import sys

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

PRED_ROOT = (
    PROJECT
    / "predictions/heqiuyan"
)

STAGE11_RPT = (
    PROJECT
    / "reports/stage11_division_probe"
)

RPT = (
    PROJECT
    / "reports/stage12_residual_audit"
)

RPT.mkdir(
    parents=True,
    exist_ok=True,
)

CV_PATH = (
    PROJECT
    / "configs/cv_manifest.csv"
)

MORPH_ACTION_PATH = (
    STAGE11_RPT
    / "stage11_fold0_morphdiv_v2_actions_locked.pkl"
)

V1_ACTION_PATH = (
    RPT
    / "stage12_fold0_mutual_cont_v1_actions_locked.pkl"
)

V2_ACTION_PATH = (
    RPT
    / "stage12_fold0_mutual_cont_v2_actions_locked.pkl"
)

R1P50_ACTION_PATH = (
    RPT
    / "stage12_fold0_r1p50_confirmation_actions.pkl"
)

CHECKPOINT_PATH = (
    RPT
    / "stage12_current_best_checkpoint.json"
)


FN_OUT = (
    RPT
    / "stage12_current_best_edge_fn_residuals.pkl"
)

FP_OUT = (
    RPT
    / "stage12_current_best_edge_fp_residuals.pkl"
)

DATASET_OUT = (
    RPT
    / "stage12_current_best_edge_residual_datasets.pkl"
)

SUMMARY_OUT = (
    RPT
    / "stage12_current_best_edge_residual_summary.json"
)


sys.path.insert(
    0,
    str(PROJECT / "src"),
)

sys.path.insert(
    0,
    str(PROJECT / "external/official/src"),
)

from biohub_cell_tracking import frozen_v9 as fv9
from tracking_cellmot.metrics import _evaluate_matched_graph


# ============================================================
# ============================================================
# 1. Verify frozen Stage12 CURRENT_BEST checkpoint
# ============================================================

checkpoint = json.loads(
    CHECKPOINT_PATH.read_text(
        encoding="utf-8"
    )
)

EXPECTED_PIPELINE = (
    "FULL_HYBRID_PLUS_MORPH_DIV_V2"
    "_PLUS_MUTUAL_CONT_V1"
    "_PLUS_MUTUAL_CONT_V2"
    "_PLUS_DET_BRIDGE_R1P50"
)

assert checkpoint["status"] == "CURRENT_BEST"
assert checkpoint["pipeline"] == EXPECTED_PIPELINE

assert abs(
    float(checkpoint["score"])
    - 0.786344478
) <= 5e-7

assert int(checkpoint["edge_tp"]) == 22806
assert int(checkpoint["edge_fp"]) == 2658
assert int(checkpoint["edge_fn"]) == 3425


# 2. Frozen current-best routes
# ============================================================

S9995_995 = (
    PRED_ROOT
    / "stage10_infer_sparse9995_256_det995"
    / "split_0"
)

S9995_9975 = (
    PRED_ROOT
    / "stage10_infer_sparse9995_256_det9975"
    / "split_0"
)

B256_995 = (
    PRED_ROOT
    / "stage10_infer_baseline_256_det995"
    / "split_0"
)

B256_9975 = (
    PRED_ROOT
    / "stage10_infer_baseline_256_det9975"
    / "split_0"
)


# ============================================================
# 3. Fold0 corpus
# ============================================================

cv = pd.read_csv(
    CV_PATH
)

cv0 = (
    cv[
        cv["fold"] == 0
    ]
    .sort_values(
        ["prefix", "dataset"]
    )
    .reset_index(drop=True)
)

assert len(cv0) == 40

names_44 = (
    cv0.loc[
        cv0["dataset"].str.startswith("44b6_"),
        "dataset",
    ]
    .tolist()
)

names_6 = (
    cv0.loc[
        cv0["dataset"].str.startswith("6bba_"),
        "dataset",
    ]
    .tolist()
)

assert len(names_44) == 15
assert len(names_6) == 25

prefix_map = dict(
    zip(
        cv0["dataset"],
        cv0["prefix"],
    )
)

estimated_nodes = dict(
    zip(
        cv0["dataset"],
        cv0["estimated_nodes"].astype(float),
    )
)


# ============================================================
# 4. Build Frozen-V9 route policies
# ============================================================

policies_44 = fv9.build_frozen_v9_policies(
    names_44,
    S9995_995,
    S9995_9975,
    TRAIN_ROOT,
    prefix_map=prefix_map,
    config=fv9.FROZEN_V9_CONFIG,
    progress=False,
)

policies_6 = fv9.build_frozen_v9_policies(
    names_6,
    B256_995,
    B256_9975,
    TRAIN_ROOT,
    prefix_map=prefix_map,
    config=fv9.FROZEN_V9_CONFIG,
    progress=False,
)

morph_actions = pd.read_pickle(
    MORPH_ACTION_PATH
)

v1_actions = pd.read_pickle(
    V1_ACTION_PATH
)

v2_actions = pd.read_pickle(
    V2_ACTION_PATH
)

r1p50_actions = pd.read_pickle(
    R1P50_ACTION_PATH
)


assert len(morph_actions) == 34
assert len(v1_actions) == 10107
assert len(v2_actions) == 4076
assert len(r1p50_actions) == 152


for table in [
    morph_actions,
    v1_actions,
    v2_actions,
    r1p50_actions,
]:
    assert set(
        table["dataset"]
    ).issubset(
        set(names_6)
    )


# ============================================================
# ============================================================
# 5. CURRENT_BEST graph builder
# ============================================================

def graph_state(graph):

    nodes = (
        graph.node_attrs(
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
        graph.edge_attrs(
            unpack=True
        )
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

    edge_set = set(
        zip(
            edges["source_id"].astype(int),
            edges["target_id"].astype(int),
        )
    )

    return (
        nodes,
        outdeg,
        indeg,
        edge_set,
    )


def add_continuation_actions(
    graph,
    nodes,
    table,
    name,
):

    ds = table[
        table["dataset"].eq(name)
    ]

    for a in ds.itertuples(
        index=False
    ):

        src = int(
            a.source_pred_node_id
        )

        tgt = int(
            a.target_pred_node_id
        )

        dist = fv9.recovered_edge_dist(
            nodes,
            src,
            tgt,
            fv9.FROZEN_V9_CONFIG,
        )

        graph.add_edge(
            src,
            tgt,
            {
                "edge_dist":
                    float(dist),

                "edge_prob":
                    float(
                        a.candidate_prob
                    ),
            },
            validate_keys=True,
        )


def _build_current_best_pre_g3h3e1(name):

    is_44 = name.startswith(
        "44b6_"
    )

    if is_44:

        p995 = S9995_995
        p9975 = S9995_9975
        policies = policies_44

    else:

        p995 = B256_995
        p9975 = B256_9975
        policies = policies_6


    graph, scale, _ = (
        fv9.apply_frozen_v9(
            name,
            p995,
            p9975,
            policies,
            TRAIN_ROOT,
            config=fv9.FROZEN_V9_CONFIG,
        )
    )


    # 44b6 route is unchanged after Frozen-V9.
    if is_44:
        return graph, scale


    nodes, outdeg, indeg, edge_set = (
        graph_state(graph)
    )


    # --------------------------------------------------------
    # MORPH-DIV-V2
    # Exact original Stage11 insertion semantics.
    # --------------------------------------------------------

    morph = (
        morph_actions[
            morph_actions["dataset"].eq(name)
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


    for a in morph.itertuples(
        index=False
    ):

        src = int(
            a.source_pred_node_id
        )

        tgt = int(
            a.target_pred_node_id
        )

        assert outdeg.get(src, 0) == 1
        assert indeg.get(tgt, 0) == 0
        assert (src, tgt) not in edge_set

        dist = fv9.recovered_edge_dist(
            nodes,
            src,
            tgt,
            fv9.FROZEN_V9_CONFIG,
        )

        graph.add_edge(
            src,
            tgt,
            {
                "edge_dist":
                    float(dist),

                "edge_prob":
                    float(
                        a.candidate_prob
                    ),
            },
        )

        outdeg[src] = 2
        indeg[tgt] = 1
        edge_set.add(
            (src, tgt)
        )


    # --------------------------------------------------------
    # MUTUAL-CONT-V1
    # --------------------------------------------------------

    add_continuation_actions(
        graph,
        nodes,
        v1_actions,
        name,
    )


    # --------------------------------------------------------
    # MUTUAL-CONT-V2
    # --------------------------------------------------------

    add_continuation_actions(
        graph,
        nodes,
        v2_actions,
        name,
    )


    # Re-read exact post-V2 graph state.
    nodes, outdeg, indeg, edge_set = (
        graph_state(graph)
    )


    # --------------------------------------------------------
    # Confirmed DET-BRIDGE-R1P50
    # --------------------------------------------------------

    bridge = (
        r1p50_actions[
            r1p50_actions[
                "dataset"
            ].eq(name)
        ]
    )


    for a in bridge.itertuples(
        index=False
    ):

        src = int(
            a.source_id
        )

        tgt = int(
            a.target_id
        )

        assert outdeg.get(src, 0) == 0
        assert indeg.get(tgt, 0) == 0


        new_id = graph.add_node(
            {
                "t":
                    int(a.t),

                "z":
                    float(a.z),

                "y":
                    float(a.y),

                "x":
                    float(a.x),
            },
            validate_keys=True,
        )


        graph.add_edge(
            src,
            new_id,
            {
                "edge_dist":
                    float(
                        a.src_dist_um
                    ),

                "edge_prob":
                    float(
                        a.in_prob
                    ),
            },
            validate_keys=True,
        )


        graph.add_edge(
            new_id,
            tgt,
            {
                "edge_dist":
                    float(
                        a.tgt_dist_um
                    ),

                "edge_prob":
                    float(
                        a.out_prob
                    ),
            },
            validate_keys=True,
        )


        # Locked confirmation actions are conflict-free.
        outdeg[src] = 1
        indeg[new_id] = 1

        outdeg[new_id] = 1
        indeg[tgt] = 1


    return graph, scale


# ============================================================
# Stage 12.11G3H3E1 confirmed CURRENT_BEST augmentation
#
# Policy:
#   ASYM1x5_Q9990_NO_TARGET_CUT_V1
#
# Development score delta:
#   +0.000050616471
#
# Locked Fold0 score delta:
#   +0.000054118252
#
# Locked Fold0 TP/FP/FN delta:
#   0 / -2 / 0
#
# Scientific selection is CLOSED.
# Fold0 is consumed.
# No policy retuning is permitted here.
# ============================================================

def build_current_best(name):
    graph, scale = (
        _build_current_best_pre_g3h3e1(
            name
        )
    )

    # Confirmed branch applies only to 6bba.
    # 44b6 remains exact historical CURRENT_BEST.
    if name.startswith("44b6_"):
        return graph, scale

    assert name.startswith(
        "6bba_"
    ), name

    from biohub_cell_tracking.stage12_confirmed_rewire import (
        apply_confirmed_rewire_policy,
    )

    artifact_dir = (
        PROJECT
        / "reports"
        / "stage12_residual_audit"
    )

    graph, _rewire_stats = (
        apply_confirmed_rewire_policy(
            name,
            graph,
            pre_rewire_builder=
                _build_current_best_pre_g3h3e1,
            artifact_dir=artifact_dir,
        )
    )

    return graph, scale


# 6. Residual audit
# ============================================================

metric_records = []
dataset_rows = []
fn_rows = []
fp_rows = []


for i, name in enumerate(
    cv0["dataset"],
    start=1,
):

    prefix = prefix_map[name]

    graph, scale = build_current_best(
        name
    )

    gt = fv9.load_graph(
        TRAIN_ROOT
        / f"{name}.geff"
    )

    # Official evaluation performs node matching in place.
    metric = fv9.evaluate_graph(
        graph,
        gt,
        scale,
        estimated_nodes[name],
        fv9.FROZEN_V9_CONFIG,
    )

    metric_records.append(
        metric
    )

    # Exact official pred_valid / matched mask logic.
    edge_eval = (
        _evaluate_matched_graph(
            graph,
            gt,
        )
        .to_pandas()
    )

    pred_nodes = (
        graph.node_attrs(
            unpack=True
        )
        .to_pandas()
    )

    pred_edges = (
        graph.edge_attrs(
            unpack=True
        )
        .to_pandas()
    )

    gt_edges = (
        gt.edge_attrs(
            unpack=True
        )
        .to_pandas()
    )


    # --------------------------------------------------------
    # Node matching maps
    # --------------------------------------------------------

    pred_nodes[
        "match_node_id"
    ] = (
        pd.to_numeric(
            pred_nodes[
                "match_node_id"
            ],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    matched_nodes = (
        pred_nodes[
            pred_nodes[
                "match_node_id"
            ] >= 0
        ]
    )

    # Official matching is one-to-one.
    assert not matched_nodes[
        "match_node_id"
    ].duplicated().any()

    pred_to_gt = dict(
        zip(
            matched_nodes[
                "node_id"
            ].astype(int),

            matched_nodes[
                "match_node_id"
            ].astype(int),
        )
    )

    gt_to_pred = {
        gt_id: pred_id
        for pred_id, gt_id
        in pred_to_gt.items()
    }


    # --------------------------------------------------------
    # Exact TP / FP sets from official edge accounting
    # --------------------------------------------------------

    matched_mask = (
        edge_eval[
            "matched_edge_mask"
        ].astype(bool)
    )

    valid_mask = (
        edge_eval[
            "pred_valid"
        ].astype(bool)
    )

    tp_edges = edge_eval[
        matched_mask
    ]

    valid_fp_edges = edge_eval[
        valid_mask
        &
        ~matched_mask
    ]


    assert len(tp_edges) == int(
        metric["edge_tp"]
    )

    assert len(valid_fp_edges) == int(
        metric["edge_fp"]
    )


    # --------------------------------------------------------
    # Convert TP prediction edges back to GT edge identities
    # --------------------------------------------------------

    matched_gt_pairs = set()

    for e in tp_edges.itertuples(
        index=False
    ):

        ps = int(
            e.source_id
        )

        pt = int(
            e.target_id
        )

        assert ps in pred_to_gt
        assert pt in pred_to_gt

        matched_gt_pairs.add(
            (
                pred_to_gt[ps],
                pred_to_gt[pt],
            )
        )


    assert (
        len(matched_gt_pairs)
        == int(metric["edge_tp"])
    )


    gt_edge_set = set(
        zip(
            gt_edges[
                "source_id"
            ].astype(int),

            gt_edges[
                "target_id"
            ].astype(int),
        )
    )

    fn_pairs = (
        gt_edge_set
        - matched_gt_pairs
    )

    assert len(fn_pairs) == int(
        metric["edge_fn"]
    )


    # --------------------------------------------------------
    # Degree context
    # --------------------------------------------------------

    pred_outdeg = (
        pred_edges.groupby(
            "source_id"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    pred_indeg = (
        pred_edges.groupby(
            "target_id"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    gt_outdeg = (
        gt_edges.groupby(
            "source_id"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    valid_fp_out = (
        valid_fp_edges.groupby(
            "source_id"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    valid_fp_in = (
        valid_fp_edges.groupby(
            "target_id"
        )
        .size()
        .astype(int)
        .to_dict()
    )


    # --------------------------------------------------------
    # FN root-cause decomposition
    # --------------------------------------------------------

    for gsrc, gtgt in sorted(
        fn_pairs
    ):

        ps = gt_to_pred.get(
            gsrc
        )

        pt = gt_to_pred.get(
            gtgt
        )

        src_present = (
            ps is not None
        )

        tgt_present = (
            pt is not None
        )

        if (
            not src_present
            and not tgt_present
        ):
            cause = (
                "DETECTION_BOTH_MISSING"
            )

        elif not src_present:
            cause = (
                "DETECTION_SOURCE_MISSING"
            )

        elif not tgt_present:
            cause = (
                "DETECTION_TARGET_MISSING"
            )

        else:
            cause = (
                "LINKING_LIMITED"
            )


        if cause == "LINKING_LIMITED":

            src_out = int(
                pred_outdeg.get(
                    ps,
                    0,
                )
            )

            tgt_in = int(
                pred_indeg.get(
                    pt,
                    0,
                )
            )

            if tgt_in == 0:
                topology = (
                    f"SRC_OUT{src_out}_TARGET_FREE"
                )
            else:
                topology = (
                    f"SRC_OUT{src_out}_TARGET_OCCUPIED"
                )

            src_fp = int(
                valid_fp_out.get(
                    ps,
                    0,
                )
            )

            tgt_fp = int(
                valid_fp_in.get(
                    pt,
                    0,
                )
            )

        else:

            src_out = (
                int(
                    pred_outdeg.get(
                        ps,
                        0,
                    )
                )
                if ps is not None
                else -1
            )

            tgt_in = (
                int(
                    pred_indeg.get(
                        pt,
                        0,
                    )
                )
                if pt is not None
                else -1
            )

            topology = (
                "DETECTION_LIMITED"
            )

            src_fp = 0
            tgt_fp = 0


        fn_rows.append({
            "dataset":
                name,

            "prefix":
                prefix,

            "gt_source_id":
                int(gsrc),

            "gt_target_id":
                int(gtgt),

            "pred_source_id":
                (
                    int(ps)
                    if ps is not None
                    else -1
                ),

            "pred_target_id":
                (
                    int(pt)
                    if pt is not None
                    else -1
                ),

            "cause":
                cause,

            "gt_parent_outdegree":
                int(
                    gt_outdeg.get(
                        gsrc,
                        0,
                    )
                ),

            "pred_source_outdegree":
                src_out,

            "pred_target_indegree":
                tgt_in,

            "link_topology":
                topology,

            "source_has_valid_fp_out":
                bool(
                    src_fp > 0
                ),

            "target_has_valid_fp_in":
                bool(
                    tgt_fp > 0
                ),

            "swap_supported":
                bool(
                    src_fp > 0
                    or tgt_fp > 0
                ),
        })


    # --------------------------------------------------------
    # Exact official FP endpoint-support decomposition
    # --------------------------------------------------------

    for e in valid_fp_edges.itertuples(
        index=False
    ):

        ps = int(
            e.source_id
        )

        pt = int(
            e.target_id
        )

        sg = pred_to_gt.get(
            ps
        )

        tg = pred_to_gt.get(
            pt
        )

        if (
            sg is not None
            and tg is not None
        ):
            support = (
                "BOTH_ENDPOINTS_GT_MATCHED"
            )

        elif sg is not None:
            support = (
                "SOURCE_ONLY_GT_MATCHED"
            )

        elif tg is not None:
            support = (
                "TARGET_ONLY_GT_MATCHED"
            )

        else:
            support = (
                "NEITHER_ENDPOINT_GT_MATCHED"
            )


        fp_rows.append({
            "dataset":
                name,

            "prefix":
                prefix,

            "source_pred_node_id":
                ps,

            "target_pred_node_id":
                pt,

            "source_gt_node_id":
                (
                    int(sg)
                    if sg is not None
                    else -1
                ),

            "target_gt_node_id":
                (
                    int(tg)
                    if tg is not None
                    else -1
                ),

            "endpoint_support":
                support,

            "pred_source_outdegree":
                int(
                    pred_outdeg.get(
                        ps,
                        0,
                    )
                ),

            "pred_target_indegree":
                int(
                    pred_indeg.get(
                        pt,
                        0,
                    )
                ),
        })


    dataset_rows.append({
        "dataset":
            name,

        "prefix":
            prefix,

        "edge_tp":
            int(
                metric["edge_tp"]
            ),

        "edge_fp":
            int(
                metric["edge_fp"]
            ),

        "edge_fn":
            int(
                metric["edge_fn"]
            ),

        "node_recall":
            float(
                metric["node_recall"]
            ),
    })


    if (
        i == 1
        or i % 10 == 0
        or i == 40
    ):
        print(
            f"{i:2d}/40 datasets"
        )


# ============================================================
# 7. Global consistency gates
# ============================================================

fn_df = pd.DataFrame(
    fn_rows
)

fp_df = pd.DataFrame(
    fp_rows
)

dataset_df = pd.DataFrame(
    dataset_rows
)


total_tp = int(
    dataset_df[
        "edge_tp"
    ].sum()
)

total_fp = int(
    dataset_df[
        "edge_fp"
    ].sum()
)

total_fn = int(
    dataset_df[
        "edge_fn"
    ].sum()
)


assert len(fn_df) == total_fn
assert len(fp_df) == total_fp

assert total_tp == 22806
assert total_fp == 2658
assert total_fn == 3425


summary_metric = fv9.summarise(
    metric_records
)


print(
    "\n========== CURRENT-BEST REPRODUCTION =========="
)

print(
    "edge TP/FP/FN:",
    total_tp,
    "/",
    total_fp,
    "/",
    total_fn,
)

print(
    "edge J       :",
    f"{summary_metric['edge_jaccard']:.9f}",
)

print(
    "adj edge J   :",
    f"{summary_metric['adj_edge_jaccard']:.9f}",
)

print(
    "score        :",
    f"{summary_metric['score']:.9f}",
)


assert abs(
    summary_metric[
        "edge_jaccard"
    ]
    - float(checkpoint["edge_jaccard"])
) <= 1e-9

assert abs(
    summary_metric[
        "score"
    ]
    - float(checkpoint["score"])
) <= 1e-9

assert abs(
    summary_metric[
        "adj_edge_jaccard"
    ]
    - float(
        checkpoint[
            "adj_edge_jaccard"
        ]
    )
) <= 1e-9

assert abs(
    summary_metric[
        "node_recall"
    ]
    - float(
        checkpoint[
            "node_recall"
        ]
    )
) <= 1e-9

assert abs(
    summary_metric[
        "division_jaccard"
    ]
    - float(
        checkpoint[
            "division_jaccard"
        ]
    )
) <= 1e-9


# ============================================================
# 8. FN root cause
# ============================================================

fn_df[
    "root_cause"
] = np.where(
    fn_df[
        "cause"
    ].eq(
        "LINKING_LIMITED"
    ),
    "LINKING_LIMITED",
    "DETECTION_LIMITED",
)


print(
    "\n========== EDGE FN ROOT CAUSE =========="
)

root_counts = (
    fn_df[
        "root_cause"
    ]
    .value_counts()
)

for key in [
    "DETECTION_LIMITED",
    "LINKING_LIMITED",
]:

    n = int(
        root_counts.get(
            key,
            0,
        )
    )

    print(
        f"{key:20s}: "
        f"{n:5d} "
        f"({n / total_fn:.3%})"
    )


print(
    "\n========== DETECTION-LIMITED DETAIL =========="
)

print(
    fn_df.loc[
        fn_df[
            "root_cause"
        ]
        == "DETECTION_LIMITED",
        "cause",
    ]
    .value_counts()
    .to_string()
)


print(
    "\n========== FN ROOT CAUSE BY PREFIX =========="
)

print(
    pd.crosstab(
        fn_df["prefix"],
        fn_df["root_cause"],
    ).to_string()
)


# ============================================================
# 9. Linking-limited topology
# ============================================================

link = fn_df[
    fn_df[
        "root_cause"
    ]
    == "LINKING_LIMITED"
].copy()


print(
    "\n========== LINKING-LIMITED TOPOLOGY =========="
)

print(
    link[
        "link_topology"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n========== LINKING FN — GT PARENT TYPE =========="
)

print(
    link[
        "gt_parent_outdegree"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\n========== LINKING FN — VALID-FP SUBSTITUTION SUPPORT =========="
)

print(
    "linking FNs:",
    len(link),
)

print(
    "source has valid FP out:",
    int(
        link[
            "source_has_valid_fp_out"
        ].sum()
    ),
)

print(
    "target has valid FP in :",
    int(
        link[
            "target_has_valid_fp_in"
        ].sum()
    ),
)

print(
    "either side supports swap:",
    int(
        link[
            "swap_supported"
        ].sum()
    ),
)

if len(link):

    print(
        "swap-supported fraction:",
        f"{link['swap_supported'].mean():.3%}"
    )


# ============================================================
# 10. Official FP anatomy
# ============================================================

print(
    "\n========== OFFICIAL VALID FP ENDPOINT SUPPORT =========="
)

print(
    fp_df[
        "endpoint_support"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n========== VALID FP SUPPORT BY PREFIX =========="
)

print(
    pd.crosstab(
        fp_df["prefix"],
        fp_df["endpoint_support"],
    ).to_string()
)


# ============================================================
# 11. Persist
# ============================================================

fn_df.to_pickle(
    FN_OUT
)

fp_df.to_pickle(
    FP_OUT
)

dataset_df.to_pickle(
    DATASET_OUT
)


payload = {
    "stage":
        "12.11A",

    "current_best_score":
        float(
            summary_metric[
                "score"
            ]
        ),

    "edge_tp":
        total_tp,

    "edge_fp":
        total_fp,

    "edge_fn":
        total_fn,

    "detection_limited_fn":
        int(
            (
                fn_df[
                    "root_cause"
                ]
                == "DETECTION_LIMITED"
            ).sum()
        ),

    "linking_limited_fn":
        int(
            (
                fn_df[
                    "root_cause"
                ]
                == "LINKING_LIMITED"
            ).sum()
        ),

    "linking_swap_supported":
        int(
            link[
                "swap_supported"
            ].sum()
        ),
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
    FN_OUT
)

print(
    "written:",
    FP_OUT
)

print(
    "written:",
    DATASET_OUT
)

print(
    "written:",
    SUMMARY_OUT
)


print(
    "\n========== 12.11A DECISION =========="
)

print(
    "R1P50_CURRENT_BEST_EDGE_RESIDUAL_CENSUS: PASS"
)