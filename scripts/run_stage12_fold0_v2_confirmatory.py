# ============================================================
# RUN — Stage12 Fold0 MUTUAL-CONT-V2 Confirmatory Evaluation
# Purpose:
#   Evaluate the already SHA-locked Fold0 MUTUAL-CONT-V2
#   actions against the exact official metric.
#
#   Compare:
#
#     V1 BASELINE
#       Baseline256 -> Frozen V9 -> MORPH-DIV-V2
#       -> MUTUAL-CONT-V1
#
#     V2
#       V1 BASELINE -> MUTUAL-CONT-V2
#
#   V2 policy parameters are immutable after this evaluation.
#
# Changes files: YES — Stage12 confirmatory artifacts
# Runs training/inference: NO
# Uses GT: YES — locked Fold0 confirmatory evaluation
# Changes policy: NO
# Keep after running: YES
# ============================================================

from pathlib import Path
import hashlib
import json
import sys

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

CV_PATH = PROJECT / "configs/cv_manifest.csv"

S11 = PROJECT / "reports/stage11_division_probe"
S12 = PROJECT / "reports/stage12_residual_audit"

MORPH_PATH = (
    S11
    / "stage11_fold0_morphdiv_v2_actions_locked.pkl"
)

V1_PATH = (
    S12
    / "stage12_fold0_mutual_cont_v1_actions_locked.pkl"
)

V2_PATH = (
    S12
    / "stage12_fold0_mutual_cont_v2_actions_locked.pkl"
)

V2_LOCK_PATH = (
    S12
    / "stage12_fold0_mutual_cont_v2_action_lock.json"
)

DATASET_OUT = (
    S12
    / "stage12_fold0_mutual_cont_v2_confirmatory_datasets.pkl"
)

SUMMARY_OUT = (
    S12
    / "stage12_fold0_mutual_cont_v2_confirmatory_summary.json"
)


sys.path.insert(
    0,
    str(PROJECT / "src"),
)

from biohub_cell_tracking import frozen_v9 as fv9


# ============================================================
# 1. Verify immutable V2 action lock
# ============================================================

v2 = (
    pd.read_pickle(V2_PATH)
    .sort_values(
        [
            "dataset",
            "t",
            "source_pred_node_id",
            "target_pred_node_id",
        ]
    )
    .reset_index(drop=True)
)

lock = json.loads(
    V2_LOCK_PATH.read_text(
        encoding="utf-8"
    )
)


EXPECTED_SHA = (
    "e59cad5260e7b222e067b6b4e1dbddb9"
    "5f04541978e30e548ab662bc9f62af35"
)

assert len(v2) == 4076

assert (
    lock["status"]
    ==
    "FOLD0_V2_ACTIONS_LOCKED_BEFORE_CONFIRMATORY_GT_EVALUATION"
)

assert lock["fold0_gt_used"] is False
assert lock["action_sha256"] == EXPECTED_SHA


LOCK_COLS = [
    "dataset",
    "t",
    "source_pred_node_id",
    "target_pred_node_id",
    "candidate_prob",
]

blob = (
    v2[LOCK_COLS]
    .to_csv(
        index=False,
        float_format="%.17g",
    )
    .encode()
)

actual_sha = hashlib.sha256(
    blob
).hexdigest()

assert actual_sha == EXPECTED_SHA


# ============================================================
# 2. Fold0 corpus
# ============================================================

cv = pd.read_csv(CV_PATH)

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

names = corp["dataset"].tolist()

prefix_map = dict(
    zip(
        corp["dataset"],
        corp["prefix"],
    )
)

estimated_nodes = dict(
    zip(
        corp["dataset"],
        corp["estimated_nodes"].astype(float),
    )
)


# ============================================================
# 3. Locked predecessor actions
# ============================================================

morph = pd.read_pickle(
    MORPH_PATH
)

v1 = pd.read_pickle(
    V1_PATH
)

assert len(morph) == 34
assert len(v1) == 10107


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
# 4. Graph helpers
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
        .astype(int)
        .to_dict()
    )

    indeg = (
        edges.groupby("target_id")
        .size()
        .astype(int)
        .to_dict()
    )

    eset = set(
        zip(
            edges["source_id"].astype(int),
            edges["target_id"].astype(int),
        )
    )

    return nodes, outdeg, indeg, eset


def add_continuations(
    g,
    aa,
):

    if aa.empty:
        return g

    nodes, outdeg, indeg, eset = (
        graph_state(g)
    )

    for a in aa.itertuples(index=False):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert outdeg.get(s, 0) == 0
        assert indeg.get(q, 0) == 0
        assert (s, q) not in eset

        assert (
            int(nodes.loc[q, "t"])
            ==
            int(nodes.loc[s, "t"]) + 1
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
                    float(a.candidate_prob),
            },
        )

        outdeg[s] = 1
        indeg[q] = 1
        eset.add((s, q))

    return g


def build_v1_baseline(name):

    g, scale, _ = fv9.apply_frozen_v9(
        name,
        P995,
        P9975,
        policies,
        TRAIN_ROOT,
        config=fv9.FROZEN_V9_CONFIG,
    )

    # --------------------------------------------------------
    # MORPH-DIV-V2
    # --------------------------------------------------------

    aa_morph = (
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

    if not aa_morph.empty:

        nodes, outdeg, indeg, eset = (
            graph_state(g)
        )

        for a in aa_morph.itertuples(
            index=False
        ):

            s = int(a.source_pred_node_id)
            q = int(a.target_pred_node_id)

            assert outdeg.get(s, 0) == 1
            assert indeg.get(q, 0) == 0
            assert (s, q) not in eset

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
                        float(a.candidate_prob),
                },
            )

            outdeg[s] = 2
            indeg[q] = 1
            eset.add((s, q))


    # --------------------------------------------------------
    # MUTUAL-CONT-V1
    # --------------------------------------------------------

    aa_v1 = (
        v1[
            v1["dataset"].eq(name)
        ]
        .sort_values(
            [
                "t",
                "source_pred_node_id",
                "target_pred_node_id",
            ]
        )
    )

    g = add_continuations(
        g,
        aa_v1,
    )

    return g, scale


# ============================================================
# 5. Locked confirmatory evaluation
# ============================================================

rows = []


for i, name in enumerate(
    names,
    start=1,
):

    baseline, scale = (
        build_v1_baseline(name)
    )

    aa_v2 = (
        v2[
            v2["dataset"].eq(name)
        ]
        .sort_values(
            [
                "t",
                "source_pred_node_id",
                "target_pred_node_id",
            ]
        )
    )

    new_graph = add_continuations(
        baseline.copy(),
        aa_v2,
    )


    # Fresh GT for both paths because evaluator mutates schema.
    gt_base = fv9.load_graph(
        TRAIN_ROOT
        / f"{name}.geff"
    )

    gt_new = fv9.load_graph(
        TRAIN_ROOT
        / f"{name}.geff"
    )


    base_metric = fv9.evaluate_graph(
        baseline.copy(),
        gt_base,
        scale,
        estimated_nodes[name],
        fv9.FROZEN_V9_CONFIG,
    )

    new_metric = fv9.evaluate_graph(
        new_graph,
        gt_new,
        scale,
        estimated_nodes[name],
        fv9.FROZEN_V9_CONFIG,
    )


    rows.append({
        "dataset":
            name,

        "actions":
            int(len(aa_v2)),

        **{
            f"base_{k}": v
            for k, v
            in base_metric.items()
        },

        **{
            f"new_{k}": v
            for k, v
            in new_metric.items()
        },
    })


    if (
        i == 1
        or i % 5 == 0
        or i == 25
    ):

        print(
            f"{i:2d}/25"
        )


df = pd.DataFrame(rows)


# ============================================================
# 6. Exact aggregate metrics
# ============================================================

base_records = [
    {
        k.removeprefix("base_"): row[k]
        for k in df.columns
        if k.startswith("base_")
    }
    for _, row in df.iterrows()
]

new_records = [
    {
        k.removeprefix("new_"): row[k]
        for k in df.columns
        if k.startswith("new_")
    }
    for _, row in df.iterrows()
]


base = fv9.summarise(
    base_records
)

new = fv9.summarise(
    new_records
)


# Exact locked V1 Fold0 checkpoint.
assert abs(
    base["score"]
    - 0.786581383
) <= 1e-9


base_tp = int(
    df["base_edge_tp"].sum()
)

base_fp = int(
    df["base_edge_fp"].sum()
)

base_fn = int(
    df["base_edge_fn"].sum()
)

new_tp = int(
    df["new_edge_tp"].sum()
)

new_fp = int(
    df["new_edge_fp"].sum()
)

new_fn = int(
    df["new_edge_fn"].sum()
)


print(
    "\n========== LOCKED V1 FOLD0 BASELINE =========="
)

print(
    "edge TP/FP/FN:",
    base_tp,
    "/",
    base_fp,
    "/",
    base_fn,
)

print(
    "edge J      :",
    f"{base['edge_jaccard']:.9f}"
)

print(
    "adj edge J  :",
    f"{base['adj_edge_jaccard']:.9f}"
)

print(
    "division J  :",
    f"{base['division_jaccard']:.9f}"
)

print(
    "score       :",
    f"{base['score']:.9f}"
)


print(
    "\n========== MUTUAL-CONT-V2 FOLD0 CONFIRMATORY =========="
)

print(
    "actions     :",
    len(v2)
)

print(
    "edge TP/FP/FN:",
    new_tp,
    "/",
    new_fp,
    "/",
    new_fn,
)

print(
    "edge J      :",
    f"{new['edge_jaccard']:.9f}"
)

print(
    "adj edge J  :",
    f"{new['adj_edge_jaccard']:.9f}"
)

print(
    "division J  :",
    f"{new['division_jaccard']:.9f}"
)

print(
    "score       :",
    f"{new['score']:.9f}"
)


print(
    "\n========== V2 CONFIRMATORY DELTAS =========="
)

print(
    "edge TP:",
    f"{new_tp-base_tp:+d}"
)

print(
    "edge FP:",
    f"{new_fp-base_fp:+d}"
)

print(
    "edge FN:",
    f"{new_fn-base_fn:+d}"
)

print(
    "edge J:",
    f"{new['edge_jaccard']-base['edge_jaccard']:+.9f}"
)

print(
    "adj edge J:",
    f"{new['adj_edge_jaccard']-base['adj_edge_jaccard']:+.9f}"
)

print(
    "division J:",
    f"{new['division_jaccard']-base['division_jaccard']:+.9f}"
)

print(
    "score:",
    f"{new['score']-base['score']:+.9f}"
)


# ============================================================
# 7. Dataset stability
# ============================================================

df["delta_adj_edge"] = (
    df["new_adj_edge_jaccard"]
    - df["base_adj_edge_jaccard"]
)

df["delta_edge"] = (
    df["new_edge_jaccard"]
    - df["base_edge_jaccard"]
)


improved = int(
    (df["delta_adj_edge"] > 0).sum()
)

unchanged = int(
    (df["delta_adj_edge"] == 0).sum()
)

worsened = int(
    (df["delta_adj_edge"] < 0).sum()
)


print(
    "\n========== V2 DATASET EDGE EFFECT =========="
)

print(
    "improved:",
    improved
)

print(
    "unchanged:",
    unchanged
)

print(
    "worsened:",
    worsened
)


# ============================================================
# 8. Persist immutable confirmatory result
# ============================================================

df.to_pickle(
    DATASET_OUT
)


payload = {
    "stage":
        "12.07B",

    "policy":
        "MUTUAL-CONT-V2",

    "status":
        "LOCKED_FOLD0_CONFIRMATORY_EVALUATION",

    "action_sha256":
        actual_sha,

    "actions":
        int(len(v2)),

    "baseline": {
        "edge_tp":
            base_tp,

        "edge_fp":
            base_fp,

        "edge_fn":
            base_fn,

        "edge_jaccard":
            float(base["edge_jaccard"]),

        "adj_edge_jaccard":
            float(base["adj_edge_jaccard"]),

        "division_jaccard":
            float(base["division_jaccard"]),

        "score":
            float(base["score"]),
    },

    "new": {
        "edge_tp":
            new_tp,

        "edge_fp":
            new_fp,

        "edge_fn":
            new_fn,

        "edge_jaccard":
            float(new["edge_jaccard"]),

        "adj_edge_jaccard":
            float(new["adj_edge_jaccard"]),

        "division_jaccard":
            float(new["division_jaccard"]),

        "score":
            float(new["score"]),
    },

    "delta": {
        "edge_tp":
            int(new_tp-base_tp),

        "edge_fp":
            int(new_fp-base_fp),

        "edge_fn":
            int(new_fn-base_fn),

        "adj_edge_jaccard":
            float(
                new["adj_edge_jaccard"]
                - base["adj_edge_jaccard"]
            ),

        "division_jaccard":
            float(
                new["division_jaccard"]
                - base["division_jaccard"]
            ),

        "score":
            float(
                new["score"]
                - base["score"]
            ),
    },

    "dataset_effect": {
        "improved":
            improved,

        "unchanged":
            unchanged,

        "worsened":
            worsened,
    },

    "policy_retuned_after_lock":
        False,
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
    DATASET_OUT
)

print(
    "written:",
    SUMMARY_OUT
)


print(
    "\n========== 12.07B DECISION =========="
)

print(
    "MUTUAL_CONT_V2_LOCKED_CONFIRMATORY_TEST: PASS"
)