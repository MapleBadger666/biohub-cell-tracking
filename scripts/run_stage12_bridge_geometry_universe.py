# ============================================================
# RUN — Stage 12.10D Full Temporal Bridge Geometry Universe
# Purpose:
#   For every suppressed Baseline256 peak on folds1–4 6bba,
#   compute nearest free-endpoint temporal bridge geometry:
#
#       free source at t-1
#           -> suppressed peak at t
#           -> free target at t+1
#
#   Cache geometry for later association scoring.
#
# Changes files: YES — Stage12 geometry artifacts
# Runs training/inference: NO
# Uses GT: NO
# Modifies prediction graphs: NO
# Keep after running: YES
# ============================================================

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


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

S11 = PROJECT / "reports/stage11_division_probe"
S12 = PROJECT / "reports/stage12_residual_audit"

PEAK_PATH = (
    S12
    / "stage12_folds14_suppressed_peak_universe.pkl"
)

OUT = (
    S12
    / "stage12_folds14_bridge_geometry_universe.pkl"
)

B12_OUT = (
    S12
    / "stage12_folds14_bridge_geometry_b12.pkl"
)

SUMMARY_OUT = (
    S12
    / "stage12_folds14_bridge_geometry_summary.json"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]

from biohub_cell_tracking import frozen_v9 as fv9


SCALE = np.array(
    [1.625, 0.40625, 0.40625],
    dtype=np.float64,
)


# ============================================================
# 1. Frozen deployment corpus
# ============================================================

cv = pd.read_csv(
    PROJECT
    / "configs/cv_manifest.csv"
)

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

prefix_map = {
    name: "6bba"
    for name in names
}


# ============================================================
# 2. Frozen postprocessing actions
# ============================================================

morph = pd.read_pickle(
    S11
    / "stage11_deployable_final_v9_policy_grid_actions.pkl"
)

morph = morph[
    morph["config_id"].eq("K3_R3_P100")
].copy()


v1 = pd.read_pickle(
    S12
    / "stage12_folds14_mutual_best_universe.pkl"
)

v1 = v1[
    v1["candidate_prob"].ge(0.25)
].copy()

assert len(v1) == 46951


v2 = pd.read_pickle(
    S12
    / "stage12_folds14_round2_reciprocal_universe.pkl"
)

v2 = v2[
    v2["candidate_prob"].ge(0.25)
].copy()

assert len(v2) == 18511


peaks = pd.read_pickle(
    PEAK_PATH
)

assert len(peaks) == 416395
assert peaks["dataset"].nunique() == 103


# ============================================================
# 3. Frozen V9 policies
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
# 4. Exact current V2 graph
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
        g.edge_attrs(unpack=True)
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

    return nodes, outdeg, indeg


def add_actions(
    g,
    nodes,
    table,
    name,
):

    aa = table[
        table["dataset"].eq(name)
    ]

    for r in aa.itertuples(index=False):

        s = int(r.source_pred_node_id)
        q = int(r.target_pred_node_id)

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
                    float(r.candidate_prob),
            },
        )


def build_current(name):

    g, _, _ = fv9.apply_frozen_v9(
        name,
        P995,
        P9975,
        policies,
        TRAIN_ROOT,
        config=fv9.FROZEN_V9_CONFIG,
    )

    nodes, _, _ = graph_state(g)

    add_actions(
        g,
        nodes,
        morph,
        name,
    )

    add_actions(
        g,
        nodes,
        v1,
        name,
    )

    add_actions(
        g,
        nodes,
        v2,
        name,
    )

    return g


# ============================================================
# 5. Geometry extraction
# ============================================================

parts = []

started = time.time()


for di, name in enumerate(
    names,
    start=1,
):

    g = build_current(name)

    nodes, outdeg, indeg = (
        graph_state(g)
    )

    nodes = nodes.reset_index()

    pp = (
        peaks[
            peaks["dataset"].eq(name)
        ]
        .copy()
    )

    dataset_rows = []


    for t, cc in pp.groupby("t"):

        t = int(t)

        # Requires both temporal sides.
        if t <= 0:
            continue


        src = nodes[
            nodes["t"].eq(t - 1)
            &
            nodes["node_id"].map(
                lambda x:
                    outdeg.get(
                        int(x),
                        0,
                    ) == 0
            )
        ]


        tgt = nodes[
            nodes["t"].eq(t + 1)
            &
            nodes["node_id"].map(
                lambda x:
                    indeg.get(
                        int(x),
                        0,
                    ) == 0
            )
        ]


        if len(src) == 0 or len(tgt) == 0:
            continue


        c_xyz = (
            cc[
                ["z", "y", "x"]
            ]
            .to_numpy(
                dtype=np.float64
            )
            *
            SCALE
        )

        s_xyz = (
            src[
                ["z", "y", "x"]
            ]
            .to_numpy(
                dtype=np.float64
            )
            *
            SCALE
        )

        q_xyz = (
            tgt[
                ["z", "y", "x"]
            ]
            .to_numpy(
                dtype=np.float64
            )
            *
            SCALE
        )


        stree = cKDTree(s_xyz)
        qtree = cKDTree(q_xyz)


        # Always request nearest two so we can quantify
        # endpoint ambiguity.
        sd, si = stree.query(
            c_xyz,
            k=min(2, len(src)),
        )

        qd, qi = qtree.query(
            c_xyz,
            k=min(2, len(tgt)),
        )


        if sd.ndim == 1:

            sd = sd[:, None]
            si = si[:, None]


        if qd.ndim == 1:

            qd = qd[:, None]
            qi = qi[:, None]


        src1 = src.iloc[
            si[:, 0]
        ].reset_index(drop=True)

        tgt1 = tgt.iloc[
            qi[:, 0]
        ].reset_index(drop=True)


        s1 = s_xyz[
            si[:, 0]
        ]

        q1 = q_xyz[
            qi[:, 0]
        ]


        midpoint = (
            s1 + q1
        ) / 2.0


        midpoint_residual = (
            np.linalg.norm(
                c_xyz - midpoint,
                axis=1,
            )
        )


        endpoint_span = (
            np.linalg.norm(
                q1 - s1,
                axis=1,
            )
        )


        balance = (
            np.abs(
                sd[:, 0]
                -
                qd[:, 0]
            )
            /
            (
                sd[:, 0]
                +
                qd[:, 0]
                +
                1e-9
            )
        )


        src_margin = (
            sd[:, 1] - sd[:, 0]
            if sd.shape[1] >= 2
            else np.full(
                len(cc),
                np.inf,
            )
        )

        tgt_margin = (
            qd[:, 1] - qd[:, 0]
            if qd.shape[1] >= 2
            else np.full(
                len(cc),
                np.inf,
            )
        )


        block = cc.copy().reset_index(
            drop=True
        )

        block[
            "nearest_source_id"
        ] = (
            src1["node_id"]
            .to_numpy(
                dtype=np.int64
            )
        )

        block[
            "nearest_target_id"
        ] = (
            tgt1["node_id"]
            .to_numpy(
                dtype=np.int64
            )
        )

        block["src_dist_um"] = (
            sd[:, 0]
        )

        block["tgt_dist_um"] = (
            qd[:, 0]
        )

        block["src_margin_um"] = (
            src_margin
        )

        block["tgt_margin_um"] = (
            tgt_margin
        )

        block[
            "midpoint_residual_um"
        ] = midpoint_residual

        block[
            "endpoint_span_um"
        ] = endpoint_span

        block[
            "step_balance"
        ] = balance


        dataset_rows.append(
            block
        )


    if dataset_rows:

        parts.append(
            pd.concat(
                dataset_rows,
                ignore_index=True,
            )
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
# 6. Universe
# ============================================================

geom = pd.concat(
    parts,
    ignore_index=True,
)


assert (
    geom[
        [
            "dataset",
            "t",
            "z",
            "y",
            "x",
        ]
    ]
    .duplicated()
    .sum()
    == 0
)


b12 = geom[
    geom["src_dist_um"].le(12.0)
    &
    geom["tgt_dist_um"].le(12.0)
].copy()


# ============================================================
# 7. GT-blind reports
# ============================================================

print(
    "\n========== FULL BRIDGE GEOMETRY UNIVERSE =========="
)

print(
    "input suppressed peaks:",
    len(peaks)
)

print(
    "temporally evaluable:",
    len(geom)
)

print(
    "B12:",
    len(b12)
)

print(
    "B12 fraction of evaluable:",
    f"{len(b12)/len(geom):.3%}"
)

print(
    "B12 fraction of all suppressed:",
    f"{len(b12)/len(peaks):.3%}"
)


print(
    "\n========== B12 QUANTILES =========="
)

cols = [
    "detector_prob",
    "src_dist_um",
    "tgt_dist_um",
    "src_margin_um",
    "tgt_margin_um",
    "midpoint_residual_um",
    "endpoint_span_um",
    "step_balance",
]

print(
    b12[
        cols
    ]
    .quantile(
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
    .to_string()
)


print(
    "\n========== B12 COUNTS BY FOLD =========="
)

print(
    b12.groupby("fold")
    .size()
    .to_string()
)


print(
    "\n========== B12 DETECTOR THRESHOLDS =========="
)

for p in [
    0.99,
    0.98,
    0.95,
    0.90,
]:

    n = int(
        b12[
            "detector_prob"
        ].gt(p)
        .sum()
    )

    print(
        f"p > {p:.2f}: "
        f"{n:,}"
    )


print(
    "\n========== B12 MIDPOINT GATES =========="
)

for residual in [
    2.0,
    4.0,
    6.0,
    8.0,
]:

    n = int(
        b12[
            "midpoint_residual_um"
        ]
        .le(residual)
        .sum()
    )

    print(
        f"midpoint <= {residual:.1f} um: "
        f"{n:,}"
    )


print(
    "\n========== B12 STEP GATES =========="
)

for d in [
    6.0,
    8.0,
    10.0,
    12.0,
]:

    n = int(
        (
            b12[
                "src_dist_um"
            ].le(d)
            &
            b12[
                "tgt_dist_um"
            ].le(d)
        ).sum()
    )

    print(
        f"both <= {d:.1f} um: "
        f"{n:,}"
    )


# ============================================================
# 8. Persist
# ============================================================

geom.to_pickle(
    OUT
)

b12.to_pickle(
    B12_OUT
)


payload = {
    "stage":
        "12.10D",

    "gt_used":
        False,

    "datasets":
        103,

    "suppressed_peaks":
        int(len(peaks)),

    "temporally_evaluable":
        int(len(geom)),

    "b12_candidates":
        int(len(b12)),

    "b12_fraction_all":
        float(
            len(b12)
            /
            len(peaks)
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
    OUT
)

print(
    "written:",
    B12_OUT
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
    "\n========== 12.10D DECISION =========="
)

print(
    "FULL_TEMPORAL_BRIDGE_GEOMETRY_EXTRACTED: PASS"
)