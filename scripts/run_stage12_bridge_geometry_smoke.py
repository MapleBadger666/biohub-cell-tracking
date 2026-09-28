# ============================================================
# RUN — Stage 12.10D-S1 Temporal Bridge Geometry Smoke
# Purpose:
#   On one 6bba pilot per folds1–4, attach suppressed peaks to
#   nearest FREE endpoints of the current locked V2 graph:
#
#       free source at t-1
#            -> suppressed peak at t
#            -> free target at t+1
#
#   Measure GT-blind geometry only.
#
# Changes files: NO
# Runs training/inference: NO
# Uses GT: NO
# Modifies graphs: only temporary in memory
# Keep after running: YES
# ============================================================

from pathlib import Path
import sys

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


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]

from biohub_cell_tracking import frozen_v9 as fv9


# ============================================================
# 1. Four fixed pilots
# ============================================================

PILOTS = [
    "6bba_05b6850b",
    "6bba_05db0fb1",
    "6bba_062c8d37",
    "6bba_07e24132",
]

SCALE = np.array(
    [1.625, 0.40625, 0.40625],
    dtype=np.float64,
)


# ============================================================
# 2. Locked Stage11/V1/V2 action sets
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
    S12
    / "stage12_folds14_suppressed_peak_universe.pkl"
)

peaks = peaks[
    peaks["dataset"].isin(PILOTS)
].copy()


# ============================================================
# 3. Frozen V9 policies
# ============================================================

prefix_map = {
    name: "6bba"
    for name in PILOTS
}

policies = fv9.build_frozen_v9_policies(
    PILOTS,
    P995,
    P9975,
    TRAIN_ROOT,
    prefix_map=prefix_map,
    config=fv9.FROZEN_V9_CONFIG,
    progress=False,
)


# ============================================================
# 4. Build exact current V2 graph
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

    # Exact promoted order.
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
# 5. Nearest free endpoint geometry
# ============================================================

rows = []


for name in PILOTS:

    g = build_current(name)

    nodes, outdeg, indeg = (
        graph_state(g)
    )

    nodes = nodes.reset_index()

    pp = peaks[
        peaks["dataset"].eq(name)
    ].copy()


    for t, cc in pp.groupby("t"):

        t = int(t)

        if t <= 0:
            continue


        src = nodes[
            (
                nodes["t"].eq(t - 1)
                &
                nodes["node_id"]
                .map(
                    lambda x:
                        outdeg.get(
                            int(x),
                            0,
                        ) == 0
                )
            )
        ]


        tgt = nodes[
            (
                nodes["t"].eq(t + 1)
                &
                nodes["node_id"]
                .map(
                    lambda x:
                        indeg.get(
                            int(x),
                            0,
                        ) == 0
                )
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


        sd, si = stree.query(
            c_xyz,
            k=2,
        )

        qd, qi = qtree.query(
            c_xyz,
            k=2,
        )


        if sd.ndim == 1:
            sd = sd[:, None]
            si = si[:, None]

        if qd.ndim == 1:
            qd = qd[:, None]
            qi = qi[:, None]


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


        for j, r in enumerate(
            cc.itertuples(index=False)
        ):

            rows.append({
                "dataset":
                    name,

                "t":
                    int(t),

                "detector_prob":
                    float(
                        r.detector_prob
                    ),

                "src_dist_um":
                    float(sd[j, 0]),

                "src_margin_um":
                    float(
                        sd[j, 1]
                        -
                        sd[j, 0]
                    ),

                "tgt_dist_um":
                    float(qd[j, 0]),

                "tgt_margin_um":
                    float(
                        qd[j, 1]
                        -
                        qd[j, 0]
                    ),

                "midpoint_residual_um":
                    float(
                        midpoint_residual[j]
                    ),

                "endpoint_span_um":
                    float(
                        endpoint_span[j]
                    ),

                "step_balance":
                    float(
                        balance[j]
                    ),
            })


audit = pd.DataFrame(rows)


# ============================================================
# 6. GT-blind diagnostics
# ============================================================

print(
    "========== TEMPORAL BRIDGE GEOMETRY =========="
)

print(
    "candidates evaluated:",
    len(audit)
)


print(
    "\n========== TWO-SIDED STEP GATES =========="
)

for d in [
    6.0,
    8.0,
    10.0,
    12.0,
]:

    mask = (
        audit["src_dist_um"].le(d)
        &
        audit["tgt_dist_um"].le(d)
    )

    print(
        f"both steps <= {d:4.1f} um: "
        f"{int(mask.sum()):,} "
        f"({mask.mean():.3%})"
    )


broad = audit[
    audit["src_dist_um"].le(12)
    &
    audit["tgt_dist_um"].le(12)
]


print(
    "\n========== B12 GEOMETRY QUANTILES =========="
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
    broad[
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
    "\n========== B12 COUNTS BY DATASET =========="
)

print(
    broad.groupby("dataset")
    .size()
    .to_string()
)


print(
    "\n========== STRICTER BRIDGE SHAPES =========="
)

for residual in [
    2.0,
    4.0,
    6.0,
]:

    mask = (
        broad[
            "midpoint_residual_um"
        ].le(residual)
    )

    print(
        f"B12 + midpoint <= {residual:.1f} um: "
        f"{int(mask.sum()):,}"
    )


print(
    "\n========== 12.10D-S1 DECISION =========="
)

print(
    "TEMPORAL_BRIDGE_GEOMETRY_SMOKE: PASS"
)