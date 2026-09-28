# ============================================================
# RUN — Stage 12.10E-S1 Augmented Association Evidence Smoke
# Purpose:
#   Score B12 suppressed peaks using the frozen Baseline256
#   transformer in full augmented-node context:
#
#       existing P995 nodes at t-1
#               -> suppressed candidates at t
#               -> existing P995 nodes at t+1
#
#   For each candidate measure association evidence toward the
#   geometry-selected free endpoints.
#
#   Before using the augmented scorer, verify zero-candidate
#   parity against DatasetReplay.replay_pair().
#
# Changes files: NO
# Runs training: NO
# Runs inference: YES — four pilot datasets
# Uses GT: NO
# Modifies persisted graphs: NO
# Keep after running: YES
# ============================================================

from pathlib import Path
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
      "stage11_fullgraph_baseline256_folds1to4_det995/split_0"
)

CKPT = (
    PROJECT
    / "models/official_baseline/"
      "stage10_pilot_baseline_256/"
      "split_0/edge_predictor_best.pth"
)

S12 = (
    PROJECT
    / "reports/stage12_residual_audit"
)

B12_PATH = (
    S12
    / "stage12_folds14_bridge_geometry_b12.pkl"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]


import run_stage11_multiframe_extract as runner

from predict_unet_transformer_mps import (
    _detect_cells_pooled,
)

from replay_stage11_association import (
    DatasetReplay,
)

from train_unet_transformer_mps import (
    extract_pos_features,
)


PILOTS = [
    "6bba_05b6850b",
    "6bba_05db0fb1",
    "6bba_062c8d37",
    "6bba_07e24132",
]


# ============================================================
# 1. Frozen candidate set
# ============================================================

b12 = pd.read_pickle(
    B12_PATH
)

b12 = (
    b12[
        b12["dataset"].isin(PILOTS)
    ]
    .copy()
    .sort_values(
        [
            "dataset",
            "t",
            "z_ds",
            "y_ds",
            "x_ds",
        ]
    )
    .reset_index(drop=True)
)


print(
    "========== PILOT B12 =========="
)

print(
    "candidates:",
    len(b12)
)

print(
    b12.groupby("dataset")
    .size()
    .to_string()
)


# ============================================================
# 2. Frozen model
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

model.eval()

assert window_size == 2
assert tuple(downsample) == (1, 4, 4)

ds_arr = np.asarray(
    downsample,
    dtype=np.int64,
)


# ============================================================
# 3. Helpers
# ============================================================

def to_numpy(x):

    if isinstance(
        x,
        torch.Tensor,
    ):

        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


def first_seen_pair_context(
    replay,
    pair_t,
):
    """
    Exact replay semantics for pair pair_t -> pair_t+1.

    Detection coordinates:
      source pair_t:
          first window containing pair_t
      target pair_t+1:
          current [pair_t, pair_t+1] window

    UNet features:
          current [pair_t, pair_t+1] window
    """

    with torch.no_grad():

        if pair_t >= 1:

            _, det_prev = replay._encode(
                [
                    pair_t - 1,
                    pair_t,
                ]
            )

            c_src = _detect_cells_pooled(
                det_prev[1][0],
                pair_t,
                replay.cfg.det_threshold,
                replay.pool_k,
            )

            unet_out, det_pair = (
                replay._encode(
                    [
                        pair_t,
                        pair_t + 1,
                    ]
                )
            )

        else:

            unet_out, det_pair = (
                replay._encode(
                    [0, 1]
                )
            )

            c_src = _detect_cells_pooled(
                det_pair[0][0],
                0,
                replay.cfg.det_threshold,
                replay.pool_k,
            )


        c_tgt = _detect_cells_pooled(
            det_pair[1][0],
            pair_t + 1,
            replay.cfg.det_threshold,
            replay.pool_k,
        )


    return (
        c_src,
        c_tgt,
        unet_out,
    )


def score_context(
    replay,
    unet_out,
    c_src,
    c_tgt,
):
    """
    Exact transformer scoring for arbitrary source/target
    coordinate sets in the frozen pair feature maps.
    """

    assert len(c_src) > 0
    assert len(c_tgt) > 0


    p_src = (
        torch.from_numpy(
            c_src[
                :, 1:
            ].astype(
                np.float32
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    p_tgt = (
        torch.from_numpy(
            c_tgt[
                :, 1:
            ].astype(
                np.float32
            )
        )
        .unsqueeze(0)
        .to(device)
    )


    window_shape = (
        replay.window,
        *replay.image_shape[1:],
    )


    src_rel = c_src.copy()

    tgt_rel = c_tgt.copy()

    src_rel[:, 0] = 0
    tgt_rel[:, 0] = 1


    pos_src = (
        torch.from_numpy(
            extract_pos_features(
                src_rel,
                window_shape,
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    pos_tgt = (
        torch.from_numpy(
            extract_pos_features(
                tgt_rel,
                window_shape,
            )
        )
        .unsqueeze(0)
        .to(device)
    )


    mask_src = torch.ones(
        1,
        len(c_src),
        dtype=torch.bool,
        device=device,
    )

    mask_tgt = torch.ones(
        1,
        len(c_tgt),
        dtype=torch.bool,
        device=device,
    )


    feat_src = (
        replay.model._index_features(
            unet_out[:, 0],
            p_src,
            mask_src,
        )
    )

    feat_tgt = (
        replay.model._index_features(
            unet_out[:, 1],
            p_tgt,
            mask_tgt,
        )
    )


    with torch.no_grad():

        edge_logits = (
            replay.model.predict_edges(
                feat_src,
                feat_tgt,
                p_src
                *
                replay.ds_arr_t,
                p_tgt
                *
                replay.ds_arr_t,
                pos_src,
                pos_tgt,
                mask_src,
                mask_tgt,
            )
        )


        raw = edge_logits[0]

        probs = torch.softmax(
            raw,
            dim=0,
        )


    return (
        raw.detach()
        .cpu()
        .numpy(),

        probs.detach()
        .cpu()
        .numpy(),
    )


def original_coord_map(
    coords_ds,
):

    xyz = (
        coords_ds[
            :, 1:
        ].astype(
            np.int64
        )
        *
        ds_arr
    )

    return {
        (
            int(coords_ds[i, 0]),
            int(xyz[i, 0]),
            int(xyz[i, 1]),
            int(xyz[i, 2]),
        ): i

        for i in range(
            len(coords_ds)
        )
    }


def rank_desc(
    values,
    index,
):

    value = values[index]

    return (
        1
        +
        int(
            np.sum(
                values > value
            )
        )
    )


def competitor_margin(
    values,
    index,
):

    if len(values) <= 1:
        return np.inf

    value = values[index]

    others = np.delete(
        values,
        index,
    )

    return float(
        value
        -
        np.max(others)
    )


# ============================================================
# 4. Smoke scoring
# ============================================================

rows = []

parity_rows = []

started = time.time()


for di, name in enumerate(
    PILOTS,
    start=1,
):

    replay = DatasetReplay(
        name,
        TRAIN_ROOT,
        model,
        window_size,
        downsample,
        device,
    )


    # --------------------------------------------------------
    # Endpoint node-id -> persisted coordinate mapping.
    # Current V2 graph has no new nodes, only edge mutations,
    # so every geometry endpoint must still be a P995 node.
    # --------------------------------------------------------

    graph = runner.load_graph(
        P995
        / f"{name}.geff"
    )

    node_df = (
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


    cc_dataset = (
        b12[
            b12["dataset"].eq(name)
        ]
    )


    assert (
        cc_dataset[
            "nearest_source_id"
        ]
        .isin(
            node_df.index
        )
        .all()
    )

    assert (
        cc_dataset[
            "nearest_target_id"
        ]
        .isin(
            node_df.index
        )
        .all()
    )


    # --------------------------------------------------------
    # One no-augmentation parity check per dataset.
    # --------------------------------------------------------

    probe_t = int(
        cc_dataset.iloc[0][
            "t"
        ]
    )

    probe_pair = (
        probe_t - 1
    )


    base_src, base_tgt, base_raw, base_prob = (
        replay.replay_pair(
            probe_pair
        )
    )


    (
        src_check,
        tgt_check,
        unet_check,
    ) = first_seen_pair_context(
        replay,
        probe_pair,
    )


    assert np.array_equal(
        base_src,
        src_check,
    )

    assert np.array_equal(
        base_tgt,
        tgt_check,
    )


    raw_check, prob_check = (
        score_context(
            replay,
            unet_check,
            src_check,
            tgt_check,
        )
    )


    raw_base_np = to_numpy(
        base_raw
    )

    prob_base_np = to_numpy(
        base_prob
    )


    raw_delta = float(
        np.max(
            np.abs(
                raw_check
                -
                raw_base_np
            )
        )
    )

    prob_delta = float(
        np.max(
            np.abs(
                prob_check
                -
                prob_base_np
            )
        )
    )


    parity_rows.append({
        "dataset":
            name,

        "raw_max_abs_delta":
            raw_delta,

        "prob_max_abs_delta":
            prob_delta,
    })


    assert raw_delta <= 1e-5

    assert prob_delta <= 1e-6


    # --------------------------------------------------------
    # Score candidate frames.
    # --------------------------------------------------------

    for t, cc in (
        cc_dataset.groupby("t")
    ):

        t = int(t)

        assert 1 <= t < replay.T - 1


        cand = np.column_stack(
            [
                np.full(
                    len(cc),
                    t,
                    dtype=np.int16,
                ),

                cc[
                    "z_ds"
                ].to_numpy(
                    dtype=np.int16
                ),

                cc[
                    "y_ds"
                ].to_numpy(
                    dtype=np.int16
                ),

                cc[
                    "x_ds"
                ].to_numpy(
                    dtype=np.int16
                ),
            ]
        )


        # ====================================================
        # Incoming pair:
        #
        #     P995(t-1) -> [P995(t) + suppressed(t)]
        # ====================================================

        (
            in_src,
            in_tgt,
            in_unet,
        ) = first_seen_pair_context(
            replay,
            t - 1,
        )


        in_src_map = (
            original_coord_map(
                in_src
            )
        )


        # Suppressed candidates must truly be absent from P995.
        existing_tgt = {
            tuple(row)
            for row in (
                np.column_stack(
                    [
                        in_tgt[:, 0],
                        (
                            in_tgt[
                                :, 1:
                            ].astype(
                                np.int64
                            )
                            *
                            ds_arr
                        ),
                    ]
                )
            )
        }


        for c in cand:

            c_orig = (
                int(c[0]),
                int(c[1] * ds_arr[0]),
                int(c[2] * ds_arr[1]),
                int(c[3] * ds_arr[2]),
            )

            assert (
                c_orig
                not in existing_tgt
            )


        in_aug_tgt = np.concatenate(
            [
                in_tgt,
                cand,
            ],
            axis=0,
        )


        in_raw, in_prob = (
            score_context(
                replay,
                in_unet,
                in_src,
                in_aug_tgt,
            )
        )


        in_candidate_start = (
            len(in_tgt)
        )


        # ====================================================
        # Outgoing pair:
        #
        #     [P995(t) + suppressed(t)] -> P995(t+1)
        # ====================================================

        (
            out_src,
            out_tgt,
            out_unet,
        ) = first_seen_pair_context(
            replay,
            t,
        )


        out_tgt_map = (
            original_coord_map(
                out_tgt
            )
        )


        out_aug_src = np.concatenate(
            [
                out_src,
                cand,
            ],
            axis=0,
        )


        out_raw, out_prob = (
            score_context(
                replay,
                out_unet,
                out_aug_src,
                out_tgt,
            )
        )


        out_candidate_start = (
            len(out_src)
        )


        # ====================================================
        # Candidate-specific evidence
        # ====================================================

        cc_local = (
            cc.reset_index(drop=True)
        )


        for j, r in enumerate(
            cc_local.itertuples(
                index=False
            )
        ):

            src_node = node_df.loc[
                int(
                    r.nearest_source_id
                )
            ]

            tgt_node = node_df.loc[
                int(
                    r.nearest_target_id
                )
            ]


            src_key = (
                int(src_node.t),
                int(src_node.z),
                int(src_node.y),
                int(src_node.x),
            )

            tgt_key = (
                int(tgt_node.t),
                int(tgt_node.z),
                int(tgt_node.y),
                int(tgt_node.x),
            )


            assert src_key in in_src_map

            assert tgt_key in out_tgt_map


            s_idx = int(
                in_src_map[
                    src_key
                ]
            )

            q_idx = int(
                out_tgt_map[
                    tgt_key
                ]
            )

            in_c_idx = (
                in_candidate_start
                +
                j
            )

            out_c_idx = (
                out_candidate_start
                +
                j
            )


            # ----------------------------------------------
            # Incoming selected edge:
            # nearest source -> candidate target
            # ----------------------------------------------

            in_edge_raw = float(
                in_raw[
                    s_idx,
                    in_c_idx,
                ]
            )

            in_edge_prob = float(
                in_prob[
                    s_idx,
                    in_c_idx,
                ]
            )

            in_col_rank = rank_desc(
                in_raw[
                    :,
                    in_c_idx,
                ],
                s_idx,
            )

            in_col_margin = (
                competitor_margin(
                    in_raw[
                        :,
                        in_c_idx,
                    ],
                    s_idx,
                )
            )

            in_row_rank = rank_desc(
                in_raw[
                    s_idx,
                    :,
                ],
                in_c_idx,
            )

            in_row_margin = (
                competitor_margin(
                    in_raw[
                        s_idx,
                        :,
                    ],
                    in_c_idx,
                )
            )


            # ----------------------------------------------
            # Outgoing selected edge:
            # candidate source -> nearest target
            # ----------------------------------------------

            out_edge_raw = float(
                out_raw[
                    out_c_idx,
                    q_idx,
                ]
            )

            out_edge_prob = float(
                out_prob[
                    out_c_idx,
                    q_idx,
                ]
            )

            out_col_rank = rank_desc(
                out_raw[
                    :,
                    q_idx,
                ],
                out_c_idx,
            )

            out_col_margin = (
                competitor_margin(
                    out_raw[
                        :,
                        q_idx,
                    ],
                    out_c_idx,
                )
            )

            out_row_rank = rank_desc(
                out_raw[
                    out_c_idx,
                    :,
                ],
                q_idx,
            )

            out_row_margin = (
                competitor_margin(
                    out_raw[
                        out_c_idx,
                        :,
                    ],
                    q_idx,
                )
            )


            rows.append({
                "dataset":
                    name,

                "fold":
                    int(r.fold),

                "t":
                    int(t),

                "z":
                    int(r.z),

                "y":
                    int(r.y),

                "x":
                    int(r.x),

                "detector_prob":
                    float(
                        r.detector_prob
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

                "nearest_source_id":
                    int(
                        r.nearest_source_id
                    ),

                "nearest_target_id":
                    int(
                        r.nearest_target_id
                    ),

                "in_raw":
                    in_edge_raw,

                "in_prob":
                    in_edge_prob,

                "in_col_rank":
                    int(
                        in_col_rank
                    ),

                "in_col_margin_raw":
                    float(
                        in_col_margin
                    ),

                "in_row_rank":
                    int(
                        in_row_rank
                    ),

                "in_row_margin_raw":
                    float(
                        in_row_margin
                    ),

                "out_raw":
                    out_edge_raw,

                "out_prob":
                    out_edge_prob,

                "out_col_rank":
                    int(
                        out_col_rank
                    ),

                "out_col_margin_raw":
                    float(
                        out_col_margin
                    ),

                "out_row_rank":
                    int(
                        out_row_rank
                    ),

                "out_row_margin_raw":
                    float(
                        out_row_margin
                    ),

                "bridge_min_prob":
                    float(
                        min(
                            in_edge_prob,
                            out_edge_prob,
                        )
                    ),

                "bridge_geom_mean_prob":
                    float(
                        np.sqrt(
                            in_edge_prob
                            *
                            out_edge_prob
                        )
                    ),
            })


    print(
        f"{di}/4 | "
        f"{name} | "
        f"{len(cc_dataset):,} candidates | "
        f"{(time.time()-started)/60:.1f} min"
    )


    if torch.backends.mps.is_available():

        torch.mps.empty_cache()


# ============================================================
# 5. Reports
# ============================================================

audit = pd.DataFrame(
    rows
)

parity = pd.DataFrame(
    parity_rows
)


print(
    "\n========== AUGMENTED SCORER PARITY =========="
)

print(
    parity.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9g}",
    )
)


print(
    "\n========== BRIDGE ASSOCIATION EVIDENCE =========="
)

print(
    "candidates:",
    len(audit)
)


cols = [
    "detector_prob",
    "in_prob",
    "out_prob",
    "bridge_min_prob",
    "bridge_geom_mean_prob",
    "in_col_margin_raw",
    "out_col_margin_raw",
]

print(
    audit[
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
    "\n========== RECIPROCAL RANK COUNTS =========="
)

for k in [
    1,
    2,
    3,
]:

    mask = (
        audit[
            "in_col_rank"
        ].le(k)
        &
        audit[
            "out_col_rank"
        ].le(k)
    )

    print(
        f"both column ranks <= {k}: "
        f"{int(mask.sum()):,} "
        f"({mask.mean():.3%})"
    )


print(
    "\n========== TWO-SIDED PROBABILITY GATES =========="
)

for p in [
    0.10,
    0.25,
    0.50,
]:

    mask = (
        audit[
            "in_prob"
        ].ge(p)
        &
        audit[
            "out_prob"
        ].ge(p)
    )

    print(
        f"both probs >= {p:.2f}: "
        f"{int(mask.sum()):,} "
        f"({mask.mean():.3%})"
    )


print(
    "\n========== RANK1 + PROBABILITY =========="
)

rank1 = (
    audit[
        "in_col_rank"
    ].eq(1)
    &
    audit[
        "out_col_rank"
    ].eq(1)
)


for p in [
    0.10,
    0.25,
    0.50,
]:

    mask = (
        rank1
        &
        audit[
            "bridge_min_prob"
        ].ge(p)
    )

    print(
        f"rank1 both + min prob >= {p:.2f}: "
        f"{int(mask.sum()):,}"
    )


print(
    "\n========== COUNTS BY DATASET =========="
)

print(
    audit.groupby(
        "dataset"
    )
    .size()
    .to_string()
)


print(
    "\n========== RUNTIME =========="
)

print(
    "elapsed min:",
    f"{(time.time()-started)/60:.1f}"
)


print(
    "\n========== 12.10E-S1 DECISION =========="
)

print(
    "AUGMENTED_ASSOCIATION_EVIDENCE_SMOKE: PASS"
)