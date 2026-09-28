# ============================================================
# RUN — Stage12 Round-2 Reciprocal Probability Policy Grid
# Purpose:
#   Evaluate a PREDECLARED probability-only policy family for
#   the GT-blind Round-2 reciprocal universe on folds1–4.
#
#   Baseline:
#       Baseline256 -> Frozen V9 -> MORPH-DIV-V2
#       -> MUTUAL-CONT-V1
#
#   Candidate family:
#       Round-2 reciprocal candidates with p >=
#       {0.10,0.20,0.25,0.30,0.35,0.40,0.45}
#
# Changes files: YES — Stage12 policy-grid artifacts
# Runs training/inference: NO
# Uses GT: YES — folds1–4 development evaluation only
# Uses Fold0: NO
# Keep after running: YES
# ============================================================

from pathlib import Path
import json
import sys
import time

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
      "stage11_fullgraph_baseline256_folds1to4_det995/split_0"
)

P9975 = (
    PROJECT
    / "predictions/heqiuyan/"
      "stage11_fullgraph_baseline256_folds1to4_det9975/split_0"
)

CV_PATH = PROJECT / "configs/cv_manifest.csv"

S11 = PROJECT / "reports/stage11_division_probe"
S12 = PROJECT / "reports/stage12_residual_audit"

V1_PATH = (
    S12 / "stage12_folds14_mutual_best_universe.pkl"
)

R2_PATH = (
    S12 / "stage12_folds14_round2_reciprocal_universe.pkl"
)

GRID_OUT = (
    S12 / "stage12_round2_probability_grid.pkl"
)

DATASET_OUT = (
    S12 / "stage12_round2_probability_grid_datasets.pkl"
)

SUMMARY_OUT = (
    S12 / "stage12_round2_probability_grid_summary.json"
)


sys.path.insert(
    0,
    str(PROJECT / "src"),
)

from biohub_cell_tracking import frozen_v9 as fv9


THRESHOLDS = [
    .10,
    .20,
    .25,
    .30,
    .35,
    .40,
    .45,
]


# ============================================================
# 1. Development corpus
# ============================================================

cv = pd.read_csv(CV_PATH)

corp = (
    cv[
        cv["fold"].isin([1, 2, 3, 4])
        &
        cv["dataset"].str.startswith("6bba_")
    ]
    .sort_values(["fold", "dataset"])
    .reset_index(drop=True)
)

assert len(corp) == 103

names = corp["dataset"].tolist()

fold_map = dict(
    zip(
        corp["dataset"],
        corp["fold"],
    )
)

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
# 2. Frozen Stage11 + V1 artifacts
# ============================================================

morph_all = pd.read_pickle(
    S11
    / "stage11_deployable_final_v9_policy_grid_actions.pkl"
)

morph = (
    morph_all[
        morph_all["config_id"].eq(
            "K3_R3_P100"
        )
    ]
    .copy()
)


v1_universe = pd.read_pickle(
    V1_PATH
)

v1 = (
    v1_universe[
        v1_universe[
            "candidate_prob"
        ].ge(0.25)
    ]
    .copy()
)

assert len(v1) == 46951


r2 = pd.read_pickle(
    R2_PATH
)

assert len(r2) == 75480
assert r2["dataset"].nunique() == 103


# ============================================================
# 3. Frozen-V9 policies
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
# 4. Graph utilities
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


def add_zero_zero_edges(
    g,
    aa,
):

    if aa.empty:
        return g

    nodes, outdeg, indeg, eset = (
        graph_state(g)
    )

    for a in aa.itertuples(
        index=False
    ):

        s = int(
            a.source_pred_node_id
        )

        q = int(
            a.target_pred_node_id
        )

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
                    float(
                        a.candidate_prob
                    ),
            },
        )

        outdeg[s] = 1
        indeg[q] = 1
        eset.add((s, q))

    return g


def build_stage11(name):

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

    if aa.empty:
        return g, scale

    nodes, outdeg, indeg, eset = (
        graph_state(g)
    )

    for a in aa.itertuples(
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

    return g, scale


def build_v1_baseline(name):

    g, scale = build_stage11(
        name
    )

    aa = (
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

    g = add_zero_zero_edges(
        g,
        aa,
    )

    return g, scale


# ============================================================
# 5. Exact development evaluation
# ============================================================

configs = [
    ("BASELINE_V1", None)
] + [
    (
        f"R2_P{int(p * 100):02d}",
        p,
    )
    for p in THRESHOLDS
]


rows = []

started = time.time()


for di, name in enumerate(
    names,
    start=1,
):

    baseline, scale = (
        build_v1_baseline(name)
    )

    ds_r2 = (
        r2[
            r2["dataset"].eq(name)
        ]
        .sort_values(
            [
                "t",
                "source_pred_node_id",
                "target_pred_node_id",
            ]
        )
    )


    for config_id, threshold in configs:

        if threshold is None:

            graph = baseline.copy()
            actions = 0

        else:

            selected = (
                ds_r2[
                    ds_r2[
                        "candidate_prob"
                    ].ge(threshold)
                ]
            )

            graph = add_zero_zero_edges(
                baseline.copy(),
                selected,
            )

            actions = len(selected)


        gt = fv9.load_graph(
            TRAIN_ROOT
            / f"{name}.geff"
        )

        metric = fv9.evaluate_graph(
            graph,
            gt,
            scale,
            estimated_nodes[name],
            fv9.FROZEN_V9_CONFIG,
        )


        rows.append({
            "config_id":
                config_id,

            "threshold":
                (
                    float(threshold)
                    if threshold is not None
                    else float("nan")
                ),

            "dataset":
                name,

            "fold":
                int(fold_map[name]),

            "actions":
                int(actions),

            **metric,
        })


    if (
        di == 1
        or di % 10 == 0
        or di == 103
    ):

        print(
            f"{di:3d}/103 | "
            f"{(time.time()-started)/60:.1f} min"
        )


results = pd.DataFrame(rows)

assert len(results) == 103 * 8


# ============================================================
# 6. Baseline reproduction
# ============================================================

base_rows = (
    results[
        results["config_id"].eq(
            "BASELINE_V1"
        )
    ]
)

base = fv9.summarise(
    base_rows.to_dict(
        "records"
    )
)


assert abs(
    base["score"]
    - 0.788841246
) <= 1e-9


# ============================================================
# 7. Aggregate grid
# ============================================================

grid_rows = []


for config_id, threshold in configs[1:]:

    rr = (
        results[
            results["config_id"].eq(
                config_id
            )
        ]
    )

    s = fv9.summarise(
        rr.to_dict(
            "records"
        )
    )

    grid_rows.append({
        "config_id":
            config_id,

        "threshold":
            float(threshold),

        "actions":
            int(
                rr["actions"].sum()
            ),

        "datasets_acted":
            int(
                (
                    rr["actions"] > 0
                ).sum()
            ),

        "edge_tp":
            int(
                rr["edge_tp"].sum()
            ),

        "edge_fp":
            int(
                rr["edge_fp"].sum()
            ),

        "edge_fn":
            int(
                rr["edge_fn"].sum()
            ),

        "edge_jaccard":
            float(
                s["edge_jaccard"]
            ),

        "adj_edge_jaccard":
            float(
                s["adj_edge_jaccard"]
            ),

        "division_jaccard":
            float(
                s["division_jaccard"]
            ),

        "score":
            float(
                s["score"]
            ),

        "delta_adj_edge":
            float(
                s["adj_edge_jaccard"]
                - base["adj_edge_jaccard"]
            ),

        "delta_division":
            float(
                s["division_jaccard"]
                - base["division_jaccard"]
            ),

        "delta_score":
            float(
                s["score"]
                - base["score"]
            ),
    })


grid = (
    pd.DataFrame(grid_rows)
    .sort_values(
        "score",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. Fold stability
# ============================================================

fold_rows = []


for config_id, threshold in configs[1:]:

    for fold in [1, 2, 3, 4]:

        b = (
            results[
                results["config_id"].eq(
                    "BASELINE_V1"
                )
                &
                results["fold"].eq(fold)
            ]
        )

        r = (
            results[
                results["config_id"].eq(
                    config_id
                )
                &
                results["fold"].eq(fold)
            ]
        )

        sb = fv9.summarise(
            b.to_dict("records")
        )

        sr = fv9.summarise(
            r.to_dict("records")
        )

        fold_rows.append({
            "config_id":
                config_id,

            "threshold":
                float(threshold),

            "fold":
                fold,

            "actions":
                int(
                    r["actions"].sum()
                ),

            "delta_score":
                float(
                    sr["score"]
                    - sb["score"]
                ),

            "delta_adj_edge":
                float(
                    sr["adj_edge_jaccard"]
                    - sb["adj_edge_jaccard"]
                ),

            "delta_division":
                float(
                    sr["division_jaccard"]
                    - sb["division_jaccard"]
                ),
        })


fold_df = pd.DataFrame(
    fold_rows
)


# ============================================================
# 9. Report
# ============================================================

print(
    "\n========== V1 DEVELOPMENT BASELINE =========="
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
    "\n========== ROUND-2 PROBABILITY GRID =========="
)

cols = [
    "config_id",
    "threshold",
    "actions",
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "adj_edge_jaccard",
    "division_jaccard",
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


best_id = str(
    grid.iloc[0]["config_id"]
)


print(
    "\n========== BEST CONFIG FOLD STABILITY =========="
)

print(
    fold_df[
        fold_df["config_id"].eq(
            best_id
        )
    ]
    .sort_values("fold")
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


print(
    "\n========== DELTA SCORE BY THRESHOLD / FOLD =========="
)

print(
    fold_df.pivot(
        index="threshold",
        columns="fold",
        values="delta_score",
    )
    .to_string(
        float_format=lambda x:
            f"{x:+.9f}",
    )
)


# ============================================================
# 10. Persist
# ============================================================

grid.to_pickle(
    GRID_OUT
)

results.to_pickle(
    DATASET_OUT
)


payload = {
    "stage":
        "12.06C",

    "baseline":
        "MUTUAL-CONT-V1",

    "baseline_score":
        float(base["score"]),

    "thresholds":
        THRESHOLDS,

    "best_config":
        best_id,

    "best_score":
        float(
            grid.iloc[0]["score"]
        ),

    "best_delta_score":
        float(
            grid.iloc[0]["delta_score"]
        ),

    "fold0_used":
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
    GRID_OUT
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
    "\n========== 12.06C DECISION =========="
)

print(
    "ROUND2_PROBABILITY_GRID_EVALUATED: PASS"
)