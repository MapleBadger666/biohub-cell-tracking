# ============================================================
# RUN — Stage 12.09B Full Single-Rewire Universe Extractor
# Purpose:
#   Extract the complete GT-blind SOURCE_REPLACE and
#   TARGET_REPLACE candidate universes on folds1–4 after:
#
#       Frozen V9
#       -> MORPH-DIV-V2
#       -> MUTUAL-CONT-V1
#       -> MUTUAL-CONT-V2
#
#   Cache candidate/conflict association evidence and graph
#   provenance so later policy research requires NO replay.
#
# Changes files: YES
#   reports/stage12_residual_audit/
#       stage12_folds14_single_rewire_universe.pkl
#       stage12_folds14_single_rewire_universe_summary.json
#
# Runs training: NO
# Runs inference: YES — parity-verified fast association replay
# Uses GT: NO
# Executes rewiring: NO
# Keep after running: YES
# ============================================================

from collections import defaultdict
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
      "stage11_fullgraph_baseline256_folds1to4_det995/"
      "split_0"
)

P9975 = (
    PROJECT
    / "predictions/heqiuyan/"
      "stage11_fullgraph_baseline256_folds1to4_det9975/"
      "split_0"
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

V1_PATH = (
    S12
    / "stage12_folds14_mutual_best_universe.pkl"
)

V2_PATH = (
    S12
    / "stage12_folds14_round2_reciprocal_universe.pkl"
)

OUT = (
    S12
    / "stage12_folds14_single_rewire_universe.pkl"
)

SUMMARY_OUT = (
    S12
    / "stage12_folds14_single_rewire_universe_summary.json"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]

from biohub_cell_tracking import frozen_v9 as fv9

import run_stage11_multiframe_extract as runner
import stage12_iterative_reciprocal as ir

from replay_stage11_association import DatasetReplay


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
    .sort_values(
        ["fold", "dataset"]
    )
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


# ============================================================
# 2. Frozen actions
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


v2_universe = pd.read_pickle(
    V2_PATH
)

v2 = (
    v2_universe[
        v2_universe[
            "candidate_prob"
        ].ge(0.25)
    ]
    .copy()
)

assert len(v2) == 18511


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
# 4. Small utilities
# ============================================================

def finite_float(x):

    try:
        y = float(x)
    except (TypeError, ValueError):
        return np.nan

    return (
        y
        if np.isfinite(y)
        else np.nan
    )


def rank_desc(values, value):

    values = np.asarray(
        values
    )

    return int(
        1
        + np.sum(
            values > value
        )
    )


def top_margin(
    values,
    selected_index,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 2:
        return np.nan

    selected = float(
        values[selected_index]
    )

    second = float(
        np.partition(
            values,
            -2,
        )[-2]
    )

    return selected - second


# ============================================================
# 5. Exact post-V2 graph state + edge provenance
# ============================================================

def post_v2_state(name):

    g, scale, _ = fv9.apply_frozen_v9(
        name,
        P995,
        P9975,
        policies,
        TRAIN_ROOT,
        config=fv9.FROZEN_V9_CONFIG,
    )

    scale = np.asarray(
        scale,
        dtype=float,
    )


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


    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)

    edge_meta = {}


    for rec in edges.to_dict(
        "records"
    ):

        s = int(
            rec["source_id"]
        )

        q = int(
            rec["target_id"]
        )

        out_neighbors[s].add(q)
        in_neighbors[q].add(s)

        edge_meta[(s, q)] = {
            "origin":
                "FROZEN_V9",

            "graph_edge_prob":
                finite_float(
                    rec.get(
                        "edge_prob",
                        np.nan,
                    )
                ),

            "graph_edge_dist":
                finite_float(
                    rec.get(
                        "edge_dist",
                        np.nan,
                    )
                ),
        }


    def physical_distance(
        s,
        q,
    ):

        a = (
            nodes.loc[
                s,
                ["z", "y", "x"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        b = (
            nodes.loc[
                q,
                ["z", "y", "x"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        return float(
            np.linalg.norm(
                (b - a)
                * scale
            )
        )


    def add_state_edge(
        s,
        q,
        origin,
        prob,
    ):

        s = int(s)
        q = int(q)

        assert (
            q not in out_neighbors[s]
        )

        assert (
            s not in in_neighbors[q]
        )

        out_neighbors[s].add(q)
        in_neighbors[q].add(s)

        edge_meta[(s, q)] = {
            "origin":
                origin,

            "graph_edge_prob":
                float(prob),

            "graph_edge_dist":
                physical_distance(
                    s,
                    q,
                ),
        }


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

    for a in aa.itertuples(
        index=False
    ):

        s = int(
            a.source_pred_node_id
        )

        q = int(
            a.target_pred_node_id
        )

        assert len(
            out_neighbors[s]
        ) == 1

        assert len(
            in_neighbors[q]
        ) == 0

        add_state_edge(
            s,
            q,
            "MORPH_DIV_V2",
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

    for a in aa.itertuples(
        index=False
    ):

        s = int(
            a.source_pred_node_id
        )

        q = int(
            a.target_pred_node_id
        )

        assert len(
            out_neighbors[s]
        ) == 0

        assert len(
            in_neighbors[q]
        ) == 0

        add_state_edge(
            s,
            q,
            "MUTUAL_CONT_V1",
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

    for a in aa.itertuples(
        index=False
    ):

        s = int(
            a.source_pred_node_id
        )

        q = int(
            a.target_pred_node_id
        )

        assert len(
            out_neighbors[s]
        ) == 0

        assert len(
            in_neighbors[q]
        ) == 0

        add_state_edge(
            s,
            q,
            "MUTUAL_CONT_V2",
            a.candidate_prob,
        )


    outdeg = {
        int(k): len(v)
        for k, v in out_neighbors.items()
    }

    indeg = {
        int(k): len(v)
        for k, v in in_neighbors.items()
    }


    return (
        nodes,
        outdeg,
        indeg,
        out_neighbors,
        in_neighbors,
        edge_meta,
        scale,
    )


# ============================================================
# 6. Model
# ============================================================

device = runner.resolve_device(
    "mps"
)

model, window_size, downsample = (
    runner.load_model(
        CKPT,
        device,
    )
)

assert window_size == 2
assert tuple(
    downsample
) == (1, 4, 4)


# ============================================================
# 7. Extraction
# ============================================================

records = []

started = time.time()


for di, name in enumerate(
    names,
    start=1,
):

    (
        nodes,
        outdeg,
        indeg,
        out_neighbors,
        in_neighbors,
        edge_meta,
        scale,
    ) = post_v2_state(
        name
    )


    final_ids = set(
        map(
            int,
            nodes.index,
        )
    )


    raw_graph = runner.load_graph(
        P995
        / f"{name}.geff"
    )

    raw_nodes = (
        raw_graph.node_attrs(
            attr_keys=[
                "node_id",
                "t",
                "z",
                "y",
                "x",
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


    def distance_um(
        s,
        q,
    ):

        a = (
            nodes.loc[
                s,
                ["z", "y", "x"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        b = (
            nodes.loc[
                q,
                ["z", "y", "x"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        return float(
            np.linalg.norm(
                (b - a)
                * scale
            )
        )


    for t in range(
        replay.T - 1
    ):

        (
            src_ids,
            tgt_ids,
            raw,
            probs,
        ) = ir.fast_association_pair(
            replay,
            raw_nodes,
            t,
            model,
            device,
            return_raw=True,
        )


        if probs is None:
            continue


        assert (
            raw.shape
            == probs.shape
        )


        src_lookup = {
            int(node_id): i
            for i, node_id
            in enumerate(src_ids)
        }

        tgt_lookup = {
            int(node_id): j
            for j, node_id
            in enumerate(tgt_ids)
        }


        # ====================================================
        # Process one repair class
        # ====================================================

        for repair_class in [
            "SOURCE_REPLACE",
            "TARGET_REPLACE",
        ]:


            if (
                repair_class
                == "SOURCE_REPLACE"
            ):

                src_idx = np.asarray(
                    [
                        i
                        for i, s
                        in enumerate(src_ids)
                        if (
                            int(s) in final_ids
                            and outdeg.get(
                                int(s),
                                0,
                            ) == 1
                        )
                    ],
                    dtype=int,
                )

                tgt_idx = np.asarray(
                    [
                        j
                        for j, q
                        in enumerate(tgt_ids)
                        if (
                            int(q) in final_ids
                            and indeg.get(
                                int(q),
                                0,
                            ) == 0
                        )
                    ],
                    dtype=int,
                )


            else:

                src_idx = np.asarray(
                    [
                        i
                        for i, s
                        in enumerate(src_ids)
                        if (
                            int(s) in final_ids
                            and outdeg.get(
                                int(s),
                                0,
                            ) == 0
                        )
                    ],
                    dtype=int,
                )

                tgt_idx = np.asarray(
                    [
                        j
                        for j, q
                        in enumerate(tgt_ids)
                        if (
                            int(q) in final_ids
                            and indeg.get(
                                int(q),
                                0,
                            ) == 1
                        )
                    ],
                    dtype=int,
                )


            if (
                len(src_idx) == 0
                or len(tgt_idx) == 0
            ):
                continue


            sub_prob = probs[
                np.ix_(
                    src_idx,
                    tgt_idx,
                )
            ]

            sub_raw = raw[
                np.ix_(
                    src_idx,
                    tgt_idx,
                )
            ]


            row_top = np.argmax(
                sub_prob,
                axis=1,
            )

            col_top = np.argmax(
                sub_prob,
                axis=0,
            )


            for a, b in enumerate(
                row_top
            ):

                if int(
                    col_top[b]
                ) != a:
                    continue


                i = int(
                    src_idx[a]
                )

                j = int(
                    tgt_idx[b]
                )


                s = int(
                    src_ids[i]
                )

                q = int(
                    tgt_ids[j]
                )


                candidate_prob = float(
                    probs[i, j]
                )

                candidate_raw = float(
                    raw[i, j]
                )


                # --------------------------------------------
                # Identify the unique conflicting graph edge.
                # --------------------------------------------

                if (
                    repair_class
                    == "SOURCE_REPLACE"
                ):

                    assert outdeg.get(
                        s,
                        0,
                    ) == 1

                    assert indeg.get(
                        q,
                        0,
                    ) == 0

                    conflict_source = s

                    conflict_target = int(
                        next(
                            iter(
                                out_neighbors[s]
                            )
                        )
                    )


                else:

                    assert outdeg.get(
                        s,
                        0,
                    ) == 0

                    assert indeg.get(
                        q,
                        0,
                    ) == 1

                    conflict_target = q

                    conflict_source = int(
                        next(
                            iter(
                                in_neighbors[q]
                            )
                        )
                    )


                assert (
                    conflict_source,
                    conflict_target,
                ) in edge_meta


                # All current competition edges should remain
                # adjacent-frame edges.
                assert int(
                    nodes.loc[
                        conflict_target,
                        "t",
                    ]
                ) == (
                    int(
                        nodes.loc[
                            conflict_source,
                            "t",
                        ]
                    )
                    + 1
                )


                meta = edge_meta[
                    (
                        conflict_source,
                        conflict_target,
                    )
                ]


                # --------------------------------------------
                # Candidate ranks in ORIGINAL dense matrix.
                # --------------------------------------------

                candidate_prob_row_rank = (
                    rank_desc(
                        probs[i, :],
                        candidate_prob,
                    )
                )

                candidate_prob_col_rank = (
                    rank_desc(
                        probs[:, j],
                        candidate_prob,
                    )
                )

                candidate_raw_row_rank = (
                    rank_desc(
                        raw[i, :],
                        candidate_raw,
                    )
                )

                candidate_raw_col_rank = (
                    rank_desc(
                        raw[:, j],
                        candidate_raw,
                    )
                )


                # --------------------------------------------
                # Reciprocal-submatrix margins.
                # --------------------------------------------

                candidate_prob_row_margin = (
                    top_margin(
                        sub_prob[a, :],
                        b,
                    )
                )

                candidate_prob_col_margin = (
                    top_margin(
                        sub_prob[:, b],
                        a,
                    )
                )

                candidate_raw_row_margin = (
                    top_margin(
                        sub_raw[a, :],
                        b,
                    )
                )

                candidate_raw_col_margin = (
                    top_margin(
                        sub_raw[:, b],
                        a,
                    )
                )


                # --------------------------------------------
                # Conflict association evidence.
                #
                # Some Frozen-V9 union endpoints may not exist
                # in the P995 dense association matrix.
                # --------------------------------------------

                conflict_in_matrix = (
                    conflict_source
                    in src_lookup
                    and conflict_target
                    in tgt_lookup
                )


                if conflict_in_matrix:

                    ci = int(
                        src_lookup[
                            conflict_source
                        ]
                    )

                    cj = int(
                        tgt_lookup[
                            conflict_target
                        ]
                    )

                    conflict_prob = float(
                        probs[ci, cj]
                    )

                    conflict_raw = float(
                        raw[ci, cj]
                    )

                    conflict_prob_row_rank = (
                        rank_desc(
                            probs[ci, :],
                            conflict_prob,
                        )
                    )

                    conflict_prob_col_rank = (
                        rank_desc(
                            probs[:, cj],
                            conflict_prob,
                        )
                    )

                    conflict_raw_row_rank = (
                        rank_desc(
                            raw[ci, :],
                            conflict_raw,
                        )
                    )

                    conflict_raw_col_rank = (
                        rank_desc(
                            raw[:, cj],
                            conflict_raw,
                        )
                    )

                    delta_prob = (
                        candidate_prob
                        - conflict_prob
                    )

                    delta_raw = (
                        candidate_raw
                        - conflict_raw
                    )


                else:

                    conflict_prob = np.nan
                    conflict_raw = np.nan

                    conflict_prob_row_rank = np.nan
                    conflict_prob_col_rank = np.nan

                    conflict_raw_row_rank = np.nan
                    conflict_raw_col_rank = np.nan

                    delta_prob = np.nan
                    delta_raw = np.nan


                candidate_distance = (
                    distance_um(
                        s,
                        q,
                    )
                )

                conflict_distance = (
                    distance_um(
                        conflict_source,
                        conflict_target,
                    )
                )


                records.append({
                    "dataset":
                        name,

                    "fold":
                        int(
                            fold_map[name]
                        ),

                    "t":
                        int(t),

                    "repair_class":
                        repair_class,

                    # Candidate edge.
                    "source_pred_node_id":
                        s,

                    "target_pred_node_id":
                        q,

                    "candidate_prob":
                        candidate_prob,

                    "candidate_raw_logit":
                        candidate_raw,

                    "candidate_prob_row_rank":
                        candidate_prob_row_rank,

                    "candidate_prob_column_rank":
                        candidate_prob_col_rank,

                    "candidate_raw_row_rank":
                        candidate_raw_row_rank,

                    "candidate_raw_column_rank":
                        candidate_raw_col_rank,

                    "candidate_prob_row_margin":
                        float(
                            candidate_prob_row_margin
                        ),

                    "candidate_prob_column_margin":
                        float(
                            candidate_prob_col_margin
                        ),

                    "candidate_raw_row_margin":
                        float(
                            candidate_raw_row_margin
                        ),

                    "candidate_raw_column_margin":
                        float(
                            candidate_raw_col_margin
                        ),

                    "candidate_distance_um":
                        candidate_distance,

                    # Candidate topology.
                    "candidate_source_outdegree":
                        int(
                            outdeg.get(
                                s,
                                0,
                            )
                        ),

                    "candidate_target_indegree":
                        int(
                            indeg.get(
                                q,
                                0,
                            )
                        ),

                    # Conflict edge.
                    "conflict_source_id":
                        conflict_source,

                    "conflict_target_id":
                        conflict_target,

                    "conflict_origin":
                        meta["origin"],

                    "conflict_graph_edge_prob":
                        meta[
                            "graph_edge_prob"
                        ],

                    "conflict_graph_edge_dist":
                        meta[
                            "graph_edge_dist"
                        ],

                    "conflict_source_outdegree":
                        int(
                            outdeg.get(
                                conflict_source,
                                0,
                            )
                        ),

                    "conflict_target_indegree":
                        int(
                            indeg.get(
                                conflict_target,
                                0,
                            )
                        ),

                    "conflict_in_association_matrix":
                        bool(
                            conflict_in_matrix
                        ),

                    "conflict_prob":
                        float(
                            conflict_prob
                        ),

                    "conflict_raw_logit":
                        float(
                            conflict_raw
                        ),

                    "conflict_prob_row_rank":
                        conflict_prob_row_rank,

                    "conflict_prob_column_rank":
                        conflict_prob_col_rank,

                    "conflict_raw_row_rank":
                        conflict_raw_row_rank,

                    "conflict_raw_column_rank":
                        conflict_raw_col_rank,

                    # Candidate vs conflict.
                    "candidate_minus_conflict_prob":
                        float(
                            delta_prob
                        ),

                    "candidate_minus_conflict_raw":
                        float(
                            delta_raw
                        ),

                    "conflict_distance_um":
                        conflict_distance,

                    "candidate_minus_conflict_distance_um":
                        (
                            candidate_distance
                            - conflict_distance
                        ),

                    # Particularly important for TARGET_REPLACE:
                    # removing an edge from outdegree-2 parent
                    # can destroy a predicted division.
                    "conflict_parent_outdegree":
                        int(
                            outdeg.get(
                                conflict_source,
                                0,
                            )
                        ),

                    "conflict_parent_is_division":
                        bool(
                            outdeg.get(
                                conflict_source,
                                0,
                            )
                            == 2
                        ),

                    # Size of reciprocal ranking universe.
                    "repair_submatrix_sources":
                        int(
                            len(src_idx)
                        ),

                    "repair_submatrix_targets":
                        int(
                            len(tgt_idx)
                        ),
                })


    if (
        di == 1
        or di % 10 == 0
        or di == 103
    ):

        print(
            f"{di:3d}/103 | "
            f"candidates={len(records):,} | "
            f"{(time.time()-started)/60:.1f} min"
        )


    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ============================================================
# 8. Frozen observable universe
# ============================================================

universe = (
    pd.DataFrame(records)
    .sort_values(
        [
            "fold",
            "dataset",
            "t",
            "repair_class",
            "source_pred_node_id",
            "target_pred_node_id",
        ]
    )
    .reset_index(drop=True)
)


assert len(universe) > 0

assert (
    universe[
        "dataset"
    ].nunique()
    == 103
)

assert set(
    universe[
        "repair_class"
    ].unique()
) == {
    "SOURCE_REPLACE",
    "TARGET_REPLACE",
}


# Candidate identities must be unique within a repair class.
assert (
    universe[
        [
            "dataset",
            "repair_class",
            "source_pred_node_id",
            "target_pred_node_id",
        ]
    ]
    .duplicated()
    .sum()
    == 0
)


# Exact topology contract.
sr = universe[
    universe[
        "repair_class"
    ].eq(
        "SOURCE_REPLACE"
    )
]

tr = universe[
    universe[
        "repair_class"
    ].eq(
        "TARGET_REPLACE"
    )
]

assert (
    sr[
        "candidate_source_outdegree"
    ].eq(1).all()
)

assert (
    sr[
        "candidate_target_indegree"
    ].eq(0).all()
)

assert (
    tr[
        "candidate_source_outdegree"
    ].eq(0).all()
)

assert (
    tr[
        "candidate_target_indegree"
    ].eq(1).all()
)


# ============================================================
# 9. Report
# ============================================================

elapsed = (
    time.time()
    - started
)


print(
    "\n========== FULL SINGLE-REWIRE UNIVERSE =========="
)

print(
    "datasets:",
    universe[
        "dataset"
    ].nunique(),
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
    "\n========== CANDIDATES BY CLASS =========="
)

print(
    universe[
        "repair_class"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n========== CANDIDATES BY FOLD / CLASS =========="
)

print(
    pd.crosstab(
        universe[
            "fold"
        ],
        universe[
            "repair_class"
        ],
    ).to_string()
)


print(
    "\n========== CANDIDATE EVIDENCE =========="
)

for cls in [
    "SOURCE_REPLACE",
    "TARGET_REPLACE",
]:

    cc = universe[
        universe[
            "repair_class"
        ].eq(cls)
    ]

    print(
        f"\n{cls}"
    )

    for col in [
        "candidate_prob",
        "candidate_raw_logit",
        "candidate_prob_row_margin",
        "candidate_prob_column_margin",
        "candidate_raw_row_margin",
        "candidate_raw_column_margin",
        "candidate_distance_um",
    ]:

        print(
            col,
            np.round(
                cc[col]
                .quantile(
                    [
                        .10,
                        .25,
                        .50,
                        .75,
                        .90,
                    ]
                )
                .to_numpy(),
                6,
            )
        )


print(
    "\n========== CONFLICT ORIGIN =========="
)

print(
    pd.crosstab(
        universe[
            "repair_class"
        ],
        universe[
            "conflict_origin"
        ],
    ).to_string()
)


print(
    "\n========== CONFLICT MATRIX COVERAGE =========="
)

coverage = (
    universe.groupby(
        "repair_class"
    )[
        "conflict_in_association_matrix"
    ]
    .agg(
        ["sum", "count", "mean"]
    )
)

print(
    coverage.to_string(
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print(
    "\n========== CANDIDATE VS CONFLICT =========="
)

covered = universe[
    universe[
        "conflict_in_association_matrix"
    ]
]


for cls in [
    "SOURCE_REPLACE",
    "TARGET_REPLACE",
]:

    cc = covered[
        covered[
            "repair_class"
        ].eq(cls)
    ]

    print(
        f"\n{cls}"
    )

    for col in [
        "candidate_minus_conflict_prob",
        "candidate_minus_conflict_raw",
        "candidate_minus_conflict_distance_um",
    ]:

        print(
            col,
            np.round(
                cc[col]
                .quantile(
                    [
                        .10,
                        .25,
                        .50,
                        .75,
                        .90,
                    ]
                )
                .to_numpy(),
                6,
            )
        )


print(
    "\n========== TARGET-REPLACE CONFLICT PARENT =========="
)

print(
    tr[
        "conflict_parent_outdegree"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "division-parent conflicts:",
    int(
        tr[
            "conflict_parent_is_division"
        ].sum()
    ),
    "/",
    len(tr),
)


print(
    "\n========== RUNTIME =========="
)

print(
    "elapsed min:",
    f"{elapsed/60:.1f}"
)


# ============================================================
# 10. Persist
# ============================================================

universe.to_pickle(
    OUT
)


payload = {
    "stage":
        "12.09B",

    "baseline":
        (
            "FROZEN_V9_MORPH_DIV_V2_"
            "MUTUAL_CONT_V1_MUTUAL_CONT_V2"
        ),

    "datasets":
        int(
            universe[
                "dataset"
            ].nunique()
        ),

    "candidates":
        int(
            len(universe)
        ),

    "candidate_counts_by_class": {
        str(k): int(v)
        for k, v in (
            universe[
                "repair_class"
            ]
            .value_counts()
            .items()
        )
    },

    "runtime_minutes":
        float(
            elapsed / 60
        ),

    "gt_used":
        False,

    "rewiring_executed":
        False,

    "cached_evidence": [
        "candidate_prob",
        "candidate_raw_logit",
        "candidate_original_ranks",
        "candidate_reciprocal_margins",
        "candidate_distance_um",
        "conflict_edge_identity",
        "conflict_origin",
        "conflict_prob",
        "conflict_raw_logit",
        "conflict_original_ranks",
        "candidate_minus_conflict_prob",
        "candidate_minus_conflict_raw",
        "conflict_distance_um",
        "conflict_parent_outdegree",
    ],
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
    "\n========== 12.09B DECISION =========="
)

print(
    "FULL_SINGLE_REWIRE_UNIVERSE: PASS"
)