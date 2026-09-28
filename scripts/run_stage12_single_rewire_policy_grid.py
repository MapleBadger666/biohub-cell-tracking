# ============================================================
# RUN — Stage 12.09D Single-Rewire Development Policy Grid
# Purpose:
#   Evaluate a SMALL PREDECLARED single-rewire family on
#   folds1–4 after the locked Stage12 pipeline:
#
#       Frozen V9
#       -> MORPH-DIV-V2
#       -> MUTUAL-CONT-V1
#       -> MUTUAL-CONT-V2
#
#   Only conflicts introduced by MUTUAL-CONT-V1/V2 are eligible.
#
#   SOURCE_REPLACE and TARGET_REPLACE are evaluated separately.
#
#   Fixed delta-raw gates:
#       >=  0.00
#       >= -0.25
#       >= -0.50
#
#   TARGET_REPLACE additionally excludes conflict parents with
#   outdegree=2 to protect predicted divisions.
#
# Changes files: YES — Stage12 development-grid artifacts
# Runs training/inference: NO
# Uses GT: YES — folds1–4 development only
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

UNIVERSE_PATH = (
    S12
    / "stage12_folds14_single_rewire_universe.pkl"
)

V1_PATH = (
    S12
    / "stage12_folds14_mutual_best_universe.pkl"
)

V2_PATH = (
    S12
    / "stage12_folds14_round2_reciprocal_universe.pkl"
)

GRID_OUT = (
    S12
    / "stage12_single_rewire_policy_grid.pkl"
)

DATASET_OUT = (
    S12
    / "stage12_single_rewire_policy_grid_datasets.pkl"
)

SUMMARY_OUT = (
    S12
    / "stage12_single_rewire_policy_grid_summary.json"
)


sys.path.insert(
    0,
    str(PROJECT / "src"),
)

from biohub_cell_tracking import frozen_v9 as fv9


# ============================================================
# 1. PREDECLARED policy family
# ============================================================

ALLOWED_ORIGINS = {
    "MUTUAL_CONT_V1",
    "MUTUAL_CONT_V2",
}

DELTA_RAW_GATES = [
    0.00,
    -0.25,
    -0.50,
]


configs = [
    {
        "config_id": "BASELINE_V2",
        "repair_class": None,
        "delta_raw_min": None,
    }
]

for cls, prefix in [
    ("SOURCE_REPLACE", "SR"),
    ("TARGET_REPLACE", "TR"),
]:
    for gate in DELTA_RAW_GATES:

        suffix = (
            "P000"
            if gate == 0.0
            else f"M{int(abs(gate) * 100):03d}"
        )

        configs.append({
            "config_id":
                f"{prefix}_DR_{suffix}",

            "repair_class":
                cls,

            "delta_raw_min":
                float(gate),
        })


# ============================================================
# 2. Development corpus
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
# 3. Frozen Stage11 / V1 / V2 actions
# ============================================================

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
    V1_PATH
)

v1 = (
    v1_universe[
        v1_universe["candidate_prob"].ge(0.25)
    ]
    .copy()
)

assert len(v1) == 46951


v2_universe = pd.read_pickle(
    V2_PATH
)

v2 = (
    v2_universe[
        v2_universe["candidate_prob"].ge(0.25)
    ]
    .copy()
)

assert len(v2) == 18511


rewire = pd.read_pickle(
    UNIVERSE_PATH
)

assert len(rewire) == 261249
assert rewire["dataset"].nunique() == 103


# ============================================================
# 4. Pre-filter ONLY by rules fixed before GT evaluation
# ============================================================

eligible = (
    rewire[
        rewire["conflict_origin"].isin(
            ALLOWED_ORIGINS
        )
    ]
    .copy()
)


# TARGET_REPLACE safety gate:
# never remove one daughter from an outdegree-2 parent.
eligible = eligible[
    ~(
        eligible["repair_class"].eq(
            "TARGET_REPLACE"
        )
        &
        eligible["conflict_parent_is_division"]
    )
].copy()


print(
    "========== PREDECLARED ELIGIBLE UNIVERSE =========="
)

print(
    pd.crosstab(
        eligible["repair_class"],
        eligible["conflict_origin"],
    ).to_string()
)


print(
    "\n========== ACTION COUNTS BEFORE GT =========="
)

for cfg in configs[1:]:

    aa = eligible[
        eligible["repair_class"].eq(
            cfg["repair_class"]
        )
        &
        eligible[
            "candidate_minus_conflict_raw"
        ].ge(
            cfg["delta_raw_min"]
        )
    ]

    print(
        f"{cfg['config_id']:12s}: "
        f"{len(aa):6d}"
    )


# ============================================================
# 5. Frozen-V9 policies
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
# 6. Graph-state helpers
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

    out_neighbors = {}

    for r in edges.itertuples(
        index=False
    ):

        s = int(r.source_id)
        q = int(r.target_id)

        out_neighbors.setdefault(
            s,
            set(),
        ).add(q)

    in_neighbors = {}

    for r in edges.itertuples(
        index=False
    ):

        s = int(r.source_id)
        q = int(r.target_id)

        in_neighbors.setdefault(
            q,
            set(),
        ).add(s)

    return (
        nodes,
        out_neighbors,
        in_neighbors,
    )


def add_edge_checked(
    g,
    nodes,
    out_neighbors,
    in_neighbors,
    s,
    q,
    prob,
):

    s = int(s)
    q = int(q)

    assert q not in out_neighbors.get(
        s,
        set(),
    )

    assert s not in in_neighbors.get(
        q,
        set(),
    )

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
                float(prob),
        },
    )

    out_neighbors.setdefault(
        s,
        set(),
    ).add(q)

    in_neighbors.setdefault(
        q,
        set(),
    ).add(s)


# ============================================================
# 7. Build exact current V2 development baseline
# ============================================================

def build_v2_baseline(name):

    g, scale, _ = fv9.apply_frozen_v9(
        name,
        P995,
        P9975,
        policies,
        TRAIN_ROOT,
        config=fv9.FROZEN_V9_CONFIG,
    )

    (
        nodes,
        out_neighbors,
        in_neighbors,
    ) = graph_state(g)


    # --------------------------------------------------------
    # MORPH-DIV-V2
    # --------------------------------------------------------

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

    for a in aa.itertuples(index=False):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert len(
            out_neighbors.get(s, set())
        ) == 1

        assert len(
            in_neighbors.get(q, set())
        ) == 0

        add_edge_checked(
            g,
            nodes,
            out_neighbors,
            in_neighbors,
            s,
            q,
            a.candidate_prob,
        )


    # --------------------------------------------------------
    # MUTUAL-CONT-V1
    # --------------------------------------------------------

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

    for a in aa.itertuples(index=False):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert len(
            out_neighbors.get(s, set())
        ) == 0

        assert len(
            in_neighbors.get(q, set())
        ) == 0

        add_edge_checked(
            g,
            nodes,
            out_neighbors,
            in_neighbors,
            s,
            q,
            a.candidate_prob,
        )


    # --------------------------------------------------------
    # MUTUAL-CONT-V2
    # --------------------------------------------------------

    aa = (
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

    for a in aa.itertuples(index=False):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert len(
            out_neighbors.get(s, set())
        ) == 0

        assert len(
            in_neighbors.get(q, set())
        ) == 0

        add_edge_checked(
            g,
            nodes,
            out_neighbors,
            in_neighbors,
            s,
            q,
            a.candidate_prob,
        )


    return g, scale


# ============================================================
# 8. Exact single-class rewiring
# ============================================================

def apply_rewires(
    baseline,
    aa,
    repair_class,
):

    g = baseline.copy()

    if aa.empty:
        return g


    (
        nodes,
        out_neighbors,
        in_neighbors,
    ) = graph_state(g)


    # Strong static uniqueness contract.
    assert (
        aa[
            [
                "conflict_source_id",
                "conflict_target_id",
            ]
        ]
        .duplicated()
        .sum()
        == 0
    )


    if repair_class == "SOURCE_REPLACE":

        assert (
            aa[
                "source_pred_node_id"
            ]
            .duplicated()
            .sum()
            == 0
        )

        assert (
            aa[
                "target_pred_node_id"
            ]
            .duplicated()
            .sum()
            == 0
        )


    if repair_class == "TARGET_REPLACE":

        assert (
            aa[
                "target_pred_node_id"
            ]
            .duplicated()
            .sum()
            == 0
        )

        # Division-protection contract.
        assert (
            ~aa[
                "conflict_parent_is_division"
            ]
        ).all()


    for a in aa.itertuples(
        index=False
    ):

        s = int(
            a.source_pred_node_id
        )

        q = int(
            a.target_pred_node_id
        )

        cs = int(
            a.conflict_source_id
        )

        cq = int(
            a.conflict_target_id
        )


        assert cq in out_neighbors.get(
            cs,
            set(),
        )

        assert cs in in_neighbors.get(
            cq,
            set(),
        )


        if repair_class == "SOURCE_REPLACE":

            assert s == cs

            assert len(
                out_neighbors.get(s, set())
            ) == 1

            assert len(
                in_neighbors.get(q, set())
            ) == 0


        elif repair_class == "TARGET_REPLACE":

            assert q == cq

            assert len(
                out_neighbors.get(s, set())
            ) == 0

            assert len(
                in_neighbors.get(q, set())
            ) == 1

            # Explicit safety gate.
            assert len(
                out_neighbors.get(cs, set())
            ) == 1


        else:
            raise ValueError(
                repair_class
            )


        # Remove retained conflict edge.
        g.remove_edge(
            cs,
            cq,
        )

        out_neighbors[
            cs
        ].remove(cq)

        in_neighbors[
            cq
        ].remove(cs)


        # Add replacement edge.
        add_edge_checked(
            g,
            nodes,
            out_neighbors,
            in_neighbors,
            s,
            q,
            a.candidate_prob,
        )


    return g


# ============================================================
# 9. Exact development evaluation
# ============================================================

rows = []

started = time.time()


for di, name in enumerate(
    names,
    start=1,
):

    baseline, scale = (
        build_v2_baseline(name)
    )


    ds_eligible = (
        eligible[
            eligible["dataset"].eq(name)
        ]
    )


    for cfg in configs:

        config_id = cfg[
            "config_id"
        ]


        if cfg["repair_class"] is None:

            graph = baseline.copy()
            actions = 0


        else:

            selected = (
                ds_eligible[
                    ds_eligible[
                        "repair_class"
                    ].eq(
                        cfg[
                            "repair_class"
                        ]
                    )
                    &
                    ds_eligible[
                        "candidate_minus_conflict_raw"
                    ].ge(
                        cfg[
                            "delta_raw_min"
                        ]
                    )
                ]
                .sort_values(
                    [
                        "t",
                        "source_pred_node_id",
                        "target_pred_node_id",
                    ]
                )
            )


            graph = apply_rewires(
                baseline,
                selected,
                cfg["repair_class"],
            )

            actions = len(
                selected
            )


        # Evaluator mutates schema:
        # always load a fresh GT graph.
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

            "repair_class":
                (
                    cfg["repair_class"]
                    if cfg["repair_class"]
                    is not None
                    else "BASELINE"
                ),

            "delta_raw_min":
                (
                    float(
                        cfg[
                            "delta_raw_min"
                        ]
                    )
                    if cfg[
                        "delta_raw_min"
                    ] is not None
                    else float("nan")
                ),

            "dataset":
                name,

            "fold":
                int(
                    fold_map[name]
                ),

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


results = pd.DataFrame(
    rows
)

assert len(
    results
) == 103 * len(configs)


# ============================================================
# 10. Exact baseline reproduction
# ============================================================

base_rows = results[
    results["config_id"].eq(
        "BASELINE_V2"
    )
]

base = fv9.summarise(
    base_rows.to_dict(
        "records"
    )
)


print(
    "\n========== V2 DEVELOPMENT BASELINE =========="
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


assert abs(
    base["score"]
    - 0.791940513
) <= 1e-9


# ============================================================
# 11. Aggregate grid
# ============================================================

grid_rows = []


for cfg in configs[1:]:

    rr = results[
        results["config_id"].eq(
            cfg["config_id"]
        )
    ]

    s = fv9.summarise(
        rr.to_dict(
            "records"
        )
    )


    grid_rows.append({
        "config_id":
            cfg["config_id"],

        "repair_class":
            cfg["repair_class"],

        "delta_raw_min":
            float(
                cfg[
                    "delta_raw_min"
                ]
            ),

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
                s[
                    "adj_edge_jaccard"
                ]
            ),

        "division_jaccard":
            float(
                s[
                    "division_jaccard"
                ]
            ),

        "score":
            float(
                s["score"]
            ),

        "delta_adj_edge":
            float(
                s[
                    "adj_edge_jaccard"
                ]
                -
                base[
                    "adj_edge_jaccard"
                ]
            ),

        "delta_division":
            float(
                s[
                    "division_jaccard"
                ]
                -
                base[
                    "division_jaccard"
                ]
            ),

        "delta_score":
            float(
                s["score"]
                - base["score"]
            ),
    })


grid = (
    pd.DataFrame(
        grid_rows
    )
    .sort_values(
        "score",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. Fold stability
# ============================================================

fold_rows = []


for cfg in configs[1:]:

    for fold in [
        1,
        2,
        3,
        4,
    ]:

        b = results[
            results["config_id"].eq(
                "BASELINE_V2"
            )
            &
            results["fold"].eq(fold)
        ]

        r = results[
            results["config_id"].eq(
                cfg["config_id"]
            )
            &
            results["fold"].eq(fold)
        ]


        sb = fv9.summarise(
            b.to_dict(
                "records"
            )
        )

        sr = fv9.summarise(
            r.to_dict(
                "records"
            )
        )


        fold_rows.append({
            "config_id":
                cfg["config_id"],

            "repair_class":
                cfg["repair_class"],

            "delta_raw_min":
                float(
                    cfg[
                        "delta_raw_min"
                    ]
                ),

            "fold":
                fold,

            "actions":
                int(
                    r[
                        "actions"
                    ].sum()
                ),

            "delta_score":
                float(
                    sr["score"]
                    -
                    sb["score"]
                ),

            "delta_adj_edge":
                float(
                    sr[
                        "adj_edge_jaccard"
                    ]
                    -
                    sb[
                        "adj_edge_jaccard"
                    ]
                ),

            "delta_division":
                float(
                    sr[
                        "division_jaccard"
                    ]
                    -
                    sb[
                        "division_jaccard"
                    ]
                ),
        })


fold_df = pd.DataFrame(
    fold_rows
)


# ============================================================
# 13. Reports
# ============================================================

print(
    "\n========== SINGLE-REWIRE POLICY GRID =========="
)

cols = [
    "config_id",
    "repair_class",
    "delta_raw_min",
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


print(
    "\n========== DELTA SCORE BY CONFIG / FOLD =========="
)

print(
    fold_df.pivot(
        index="config_id",
        columns="fold",
        values="delta_score",
    )
    .to_string(
        float_format=lambda x:
            f"{x:+.9f}",
    )
)


print(
    "\n========== SOURCE-REPLACE FOLD STABILITY =========="
)

print(
    fold_df[
        fold_df[
            "repair_class"
        ].eq(
            "SOURCE_REPLACE"
        )
    ]
    .sort_values(
        [
            "delta_raw_min",
            "fold",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


print(
    "\n========== TARGET-REPLACE FOLD STABILITY =========="
)

print(
    fold_df[
        fold_df[
            "repair_class"
        ].eq(
            "TARGET_REPLACE"
        )
    ]
    .sort_values(
        [
            "delta_raw_min",
            "fold",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


# ============================================================
# 14. Persist
# ============================================================

grid.to_pickle(
    GRID_OUT
)

results.to_pickle(
    DATASET_OUT
)


payload = {
    "stage":
        "12.09D",

    "baseline":
        (
            "MUTUAL_CONT_V1_PLUS_"
            "MUTUAL_CONT_V2"
        ),

    "baseline_score":
        float(
            base["score"]
        ),

    "allowed_conflict_origins":
        sorted(
            ALLOWED_ORIGINS
        ),

    "delta_raw_gates":
        DELTA_RAW_GATES,

    "target_replace_division_parent_excluded":
        True,

    "best_config":
        str(
            grid.iloc[0][
                "config_id"
            ]
        ),

    "best_score":
        float(
            grid.iloc[0][
                "score"
            ]
        ),

    "best_delta_score":
        float(
            grid.iloc[0][
                "delta_score"
            ]
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
    "\n========== 12.09D DECISION =========="
)

print(
    "SINGLE_REWIRE_DEVELOPMENT_GRID_EVALUATED: PASS"
)