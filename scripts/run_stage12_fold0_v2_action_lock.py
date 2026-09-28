# ============================================================
# RUN — Stage12 Fold0 MUTUAL-CONT-V2 Blind Action Lock
# Purpose:
#   Apply locked MUTUAL-CONT-V2 to all Fold0 6bba datasets
#   WITHOUT reading Fold0 GT.
#
#   Starting state:
#       Baseline256 -> Frozen V9 -> MORPH-DIV-V2
#       -> locked MUTUAL-CONT-V1 occupancy
#
#   V2:
#       free-source x free-target submatrix
#       reciprocal row/column Top1
#       original association probability >= 0.25
#
# Changes files: YES — Stage12 locked action artifacts
# Runs training: NO
# Runs inference: YES — parity-verified fast association replay
# Uses GT: NO
# Keep after running: YES
# ============================================================

from pathlib import Path
import hashlib
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
      "stage10_infer_baseline_256_det995/split_0"
)

P9975 = (
    PROJECT
    / "predictions/heqiuyan/"
      "stage10_infer_baseline_256_det9975/split_0"
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

MORPH_PATH = (
    S11
    / "stage11_fold0_morphdiv_v2_actions_locked.pkl"
)

V1_ACTION_PATH = (
    S12
    / "stage12_fold0_mutual_cont_v1_actions_locked.pkl"
)

V1_LOCK_PATH = (
    S12
    / "stage12_fold0_mutual_cont_v1_action_lock.json"
)

V2_POLICY_PATH = (
    S12
    / "stage12_mutual_cont_v2_policy_lock.json"
)

OUT = (
    S12
    / "stage12_fold0_mutual_cont_v2_actions_locked.pkl"
)

LOCK_OUT = (
    S12
    / "stage12_fold0_mutual_cont_v2_action_lock.json"
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
# 1. Verify locked predecessor + V2 policy
# ============================================================

v1_lock = json.loads(
    V1_LOCK_PATH.read_text(
        encoding="utf-8"
    )
)

assert (
    v1_lock["status"]
    == "FOLD0_ACTIONS_LOCKED_BEFORE_GT_EVALUATION"
)

assert (
    v1_lock["action_sha256"]
    ==
    "28a54a53524eea310cd4f3d10d258f05"
    "cc73e706c96695a4b2f0d5362c454a2c"
)


v2_policy = json.loads(
    V2_POLICY_PATH.read_text(
        encoding="utf-8"
    )
)

assert (
    v2_policy["policy_name"]
    == "MUTUAL-CONT-V2"
)

assert (
    v2_policy["status"]
    == "LOCKED_BEFORE_FOLD0_CONFIRMATORY_EVALUATION"
)

assert abs(
    float(
        v2_policy[
            "rule"
        ][
            "candidate_prob_min"
        ]
    )
    - 0.25
) < 1e-12

assert (
    v2_policy[
        "development"
    ][
        "action_sha256"
    ]
    ==
    "f238d073d33029e3814ba608acb63159"
    "d9521e1749bbad4a5db038485491dd78"
)

THRESHOLD = 0.25


# ============================================================
# 2. Fold0 6bba corpus
# ============================================================

cv = pd.read_csv(
    CV_PATH
)

corp = (
    cv[
        cv["fold"].eq(0)
        &
        cv["dataset"].str.startswith(
            "6bba_"
        )
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


# ============================================================
# 3. Frozen Stage11 morphology + V1 actions
# ============================================================

morph = pd.read_pickle(
    MORPH_PATH
)

assert len(morph) == 34


v1 = pd.read_pickle(
    V1_ACTION_PATH
)

assert len(v1) == 10107
assert v1["dataset"].nunique() == 25


# ============================================================
# 4. Frozen-V9 policies
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
# 5. Exact topology AFTER Stage11 + V1
# ============================================================

def post_v1_topology(name):

    g, _, _ = fv9.apply_frozen_v9(
        name,
        P995,
        P9975,
        policies,
        TRAIN_ROOT,
        config=fv9.FROZEN_V9_CONFIG,
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


    # --------------------------------------------------------
    # Locked MORPH-DIV-V2 occupancy
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

    for a in aa_morph.itertuples(
        index=False
    ):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert outdeg.get(s, 0) == 1
        assert indeg.get(q, 0) == 0

        outdeg[s] = 2
        indeg[q] = 1


    # --------------------------------------------------------
    # Locked MUTUAL-CONT-V1 occupancy
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

    for a in aa_v1.itertuples(
        index=False
    ):

        s = int(a.source_pred_node_id)
        q = int(a.target_pred_node_id)

        assert outdeg.get(s, 0) == 0
        assert indeg.get(q, 0) == 0

        outdeg[s] = 1
        indeg[q] = 1


    return (
        nodes,
        outdeg,
        indeg,
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
assert tuple(downsample) == (1, 4, 4)


# ============================================================
# 7. Blind Fold0 V2 extraction
# ============================================================

records = []

started = time.time()


for di, name in enumerate(
    names,
    start=1,
):

    nodes, post_out, post_in = (
        post_v1_topology(name)
    )

    final_ids = set(
        map(
            int,
            nodes.index,
        )
    )


    raw_graph = runner.load_graph(
        P995 / f"{name}.geff"
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


    for t in range(
        replay.T - 1
    ):

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


        free_sources = {
            int(s)
            for s in src_ids
            if (
                int(s) in final_ids
                and post_out.get(
                    int(s),
                    0,
                ) == 0
            )
        }

        free_targets = {
            int(q)
            for q in tgt_ids
            if (
                int(q) in final_ids
                and post_in.get(
                    int(q),
                    0,
                ) == 0
            )
        }


        pairs = ir.residual_reciprocal_pairs(
            src_ids,
            tgt_ids,
            probs,
            free_sources,
            free_targets,
        )


        for s, q, p in pairs:

            # Locked V2 probability gate.
            if p < THRESHOLD:
                continue


            records.append({
                "dataset":
                    name,

                "t":
                    int(t),

                "source_pred_node_id":
                    int(s),

                "target_pred_node_id":
                    int(q),

                "candidate_prob":
                    float(p),

                "post_v1_source_outdegree":
                    0,

                "post_v1_target_indegree":
                    0,

                "round2_row_rank":
                    1,

                "round2_column_rank":
                    1,
            })


    if (
        di == 1
        or di % 5 == 0
        or di == 25
    ):

        print(
            f"{di:2d}/25 | "
            f"actions={len(records):,} | "
            f"{(time.time()-started)/60:.1f} min"
        )


    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


actions = (
    pd.DataFrame(records)
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


# ============================================================
# 8. Deployment contract
# ============================================================

assert len(actions) > 0

assert actions["dataset"].nunique() == 25

assert (
    actions[
        "candidate_prob"
    ]
    .ge(0.25)
    .all()
)

assert (
    actions[
        "post_v1_source_outdegree"
    ]
    .eq(0)
    .all()
)

assert (
    actions[
        "post_v1_target_indegree"
    ]
    .eq(0)
    .all()
)

assert (
    actions[
        "round2_row_rank"
    ]
    .eq(1)
    .all()
)

assert (
    actions[
        "round2_column_rank"
    ]
    .eq(1)
    .all()
)


# Reciprocal matching must be one-to-one for each endpoint role.
assert (
    actions[
        [
            "dataset",
            "source_pred_node_id",
        ]
    ]
    .duplicated()
    .sum()
    == 0
)

assert (
    actions[
        [
            "dataset",
            "target_pred_node_id",
        ]
    ]
    .duplicated()
    .sum()
    == 0
)


# ============================================================
# 9. Canonical SHA
# ============================================================

LOCK_COLS = [
    "dataset",
    "t",
    "source_pred_node_id",
    "target_pred_node_id",
    "candidate_prob",
]

blob = (
    actions[
        LOCK_COLS
    ]
    .to_csv(
        index=False,
        float_format="%.17g",
    )
    .encode()
)

action_sha = hashlib.sha256(
    blob
).hexdigest()


# ============================================================
# 10. Report — still NO Fold0 GT metric
# ============================================================

print(
    "\n========== FOLD0 V2 BLIND ACTION SET =========="
)

print(
    "datasets total :",
    len(names)
)

print(
    "datasets acted :",
    actions["dataset"].nunique()
)

print(
    "actions        :",
    len(actions)
)

print(
    "actions/dataset median:",
    f"{actions.groupby('dataset').size().median():.1f}"
)


print(
    "\n========== V2 ACTION PROBABILITY =========="
)

print(
    "q10/q25/q50/q75/q90:",
    np.round(
        actions[
            "candidate_prob"
        ]
        .quantile(
            [.10, .25, .50, .75, .90]
        )
        .to_numpy(),
        6,
    )
)


print(
    "\n========== V2 ACTIONS BY DATASET =========="
)

counts = (
    actions.groupby("dataset")
    .size()
)

print(
    "min   :",
    int(counts.min())
)

print(
    "median:",
    f"{counts.median():.1f}"
)

print(
    "max   :",
    int(counts.max())
)


print(
    "\n========== V2 ACTION SHA LOCK =========="
)

print(
    "SHA256:",
    action_sha
)


# ============================================================
# 11. Persist BEFORE confirmatory GT evaluation
# ============================================================

actions.to_pickle(
    OUT
)


payload = {
    "stage":
        "12.07A",

    "policy":
        "MUTUAL-CONT-V2",

    "status":
        "FOLD0_V2_ACTIONS_LOCKED_BEFORE_CONFIRMATORY_GT_EVALUATION",

    "predecessor":
        "MUTUAL-CONT-V1",

    "threshold":
        0.25,

    "datasets_total":
        int(len(names)),

    "datasets_acted":
        int(
            actions["dataset"].nunique()
        ),

    "actions":
        int(len(actions)),

    "action_sha256":
        action_sha,

    "v1_fold0_action_sha256":
        v1_lock["action_sha256"],

    "v2_development_action_sha256":
        v2_policy[
            "development"
        ][
            "action_sha256"
        ],

    "fold0_gt_used":
        False,
}


LOCK_OUT.write_text(
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
    LOCK_OUT
)


print(
    "\n========== 12.07A DECISION =========="
)

print(
    "FOLD0_MUTUAL_CONT_V2_ACTIONS_BLIND_LOCKED: PASS"
)