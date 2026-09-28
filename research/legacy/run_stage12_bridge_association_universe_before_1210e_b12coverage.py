# ============================================================
# RUN — Stage 12.10E Full Augmented Association Evidence
# Purpose:
#   Score all 15,074 locked B12 suppressed-peak candidates on
#   folds1–4 using the frozen Baseline256 transformer.
#
#   Incoming evidence:
#       P995(t-1) -> [P995(t) + B12(t)]
#
#   Outgoing evidence:
#       [P995(t) + B12(t)] -> P995(t+1)
#
#   Candidates are jointly augmented PER FRAME, one side at a
#   time, matching the semantics validated in 12.10E-S1.
#
#   No GT is used. No graph is modified.
#
# Changes files: YES — Stage12 neural-evidence artifacts
# Runs training: NO
# Runs inference: YES — 103 development datasets
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

OUT = (
    S12
    / "stage12_folds14_bridge_association_evidence.pkl"
)

DATASET_OUT = (
    S12
    / "stage12_folds14_bridge_association_dataset_summary.pkl"
)

SUMMARY_OUT = (
    S12
    / "stage12_folds14_bridge_association_summary.json"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]


import run_stage11_multiframe_extract as runner

from predict_unet_transformer_mps import (
    _detect_cells_pooled,
    _load_frame,
)

from replay_stage11_association import (
    DatasetReplay,
)

from stage11_multiframe_utils import (
    normalize_window,
    encode_windows_with_tta,
)

from train_unet_transformer_mps import (
    extract_pos_features,
)


# ============================================================
# 1. Frozen contracts
# ============================================================

BATCH_SIZE = 4

EXPECTED_SHA256 = (
    "8bddad68aaabf240d36eab9cda158f12"
    "d4969fa1cd6b21277570043ef83ab689"
)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


actual_sha = sha256_file(
    CKPT
)

assert actual_sha == EXPECTED_SHA256


# ============================================================
# 2. Locked B12 universe
# ============================================================

b12 = pd.read_pickle(
    B12_PATH
)

assert len(b12) == 15074
assert b12["dataset"].nunique() == 103


b12 = (
    b12
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


key_cols = [
    "dataset",
    "t",
    "z_ds",
    "y_ds",
    "x_ds",
]


assert (
    b12[
        key_cols
    ]
    .duplicated()
    .sum()
    == 0
)


names = (
    b12[
        "dataset"
    ]
    .drop_duplicates()
    .tolist()
)

assert len(names) == 103


# ============================================================
# 3. Frozen model
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
# 4. Helpers
# ============================================================

def original_coord_map(
    coords_ds,
):

    xyz = (
        coords_ds[
            :, 1:
        ].astype(np.int64)
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


def original_coord_set(
    coords_ds,
):

    return set(
        original_coord_map(
            coords_ds
        ).keys()
    )


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


def score_context(
    replay,
    unet_out,
    c_src,
    c_tgt,
):

    assert len(c_src) > 0
    assert len(c_tgt) > 0


    p_src = (
        torch.from_numpy(
            c_src[
                :, 1:
            ].astype(np.float32)
        )
        .unsqueeze(0)
        .to(device)
    )

    p_tgt = (
        torch.from_numpy(
            c_tgt[
                :, 1:
            ].astype(np.float32)
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

        logits = (
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

        raw = logits[0]

        prob = torch.softmax(
            raw,
            dim=0,
        )


    return (
        raw.detach()
        .cpu()
        .numpy(),

        prob.detach()
        .cpu()
        .numpy(),
    )


def candidate_array(
    cc,
    t,
):

    return np.column_stack(
        [
            np.full(
                len(cc),
                int(t),
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


# ============================================================
# 5. Full scoring
# ============================================================

incoming_rows = []
outgoing_rows = []
dataset_rows = []

started = time.time()


for di, name in enumerate(
    names,
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


    cc_dataset = (
        b12[
            b12["dataset"].eq(name)
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Persisted P995 nodes.
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


    persisted_by_t = {}

    for t, nn in (
        node_df.reset_index()
        .groupby("t")
    ):

        persisted_by_t[
            int(t)
        ] = {
            (
                int(r.t),
                int(r.z),
                int(r.y),
                int(r.x),
            )
            for r
            in nn.itertuples(
                index=False
            )
        }


    candidates_by_t = {
        int(t):
            frame.sort_values(
                [
                    "z_ds",
                    "y_ds",
                    "x_ds",
                ]
            ).reset_index(drop=True)

        for t, frame
        in cc_dataset.groupby("t")
    }


    frame_dets = {}

    parity_frames = 0


    # --------------------------------------------------------
    # Sequential pair stream; batched UNet/TTA.
    # --------------------------------------------------------

    pair_indices = list(
        range(
            replay.T - 1
        )
    )


    for start in range(
        0,
        len(pair_indices),
        BATCH_SIZE,
    ):

        batch = pair_indices[
            start:
            start + BATCH_SIZE
        ]


        windows = []

        for pair_t in batch:

            pair = torch.stack(
                [
                    _load_frame(
                        replay.zarr,
                        ti,
                        replay.target_shape,
                        replay.downsample,
                    )
                    for ti in (
                        pair_t,
                        pair_t + 1,
                    )
                ]
            )

            windows.append(
                normalize_window(
                    pair,
                    replay,
                )
            )


        imgs = torch.stack(
            windows
        ).to(device)


        with torch.no_grad():

            unet_batch, det_batch = (
                encode_windows_with_tta(
                    model,
                    imgs,
                )
            )


        for b, pair_t in enumerate(
            batch
        ):

            # ------------------------------------------------
            # Exact first-seen P995 frame detections.
            # ------------------------------------------------

            if pair_t == 0:

                frame_dets[0] = (
                    _detect_cells_pooled(
                        det_batch[b, 0],
                        0,
                        replay.cfg.det_threshold,
                        replay.pool_k,
                    )
                )


                assert (
                    original_coord_set(
                        frame_dets[0]
                    )
                    ==
                    persisted_by_t.get(
                        0,
                        set(),
                    )
                )

                parity_frames += 1


            frame_dets[
                pair_t + 1
            ] = (
                _detect_cells_pooled(
                    det_batch[b, 1],
                    pair_t + 1,
                    replay.cfg.det_threshold,
                    replay.pool_k,
                )
            )


            assert (
                original_coord_set(
                    frame_dets[
                        pair_t + 1
                    ]
                )
                ==
                persisted_by_t.get(
                    pair_t + 1,
                    set(),
                )
            )

            parity_frames += 1


            base_src = (
                frame_dets[
                    pair_t
                ]
            )

            base_tgt = (
                frame_dets[
                    pair_t + 1
                ]
            )

            unet_pair = (
                unet_batch[
                    b:
                    b + 1
                ]
            )


            # =================================================
            # OUTGOING:
            #
            # [P995(t) + candidates(t)] -> P995(t+1)
            # =================================================

            if pair_t in candidates_by_t:

                cc = (
                    candidates_by_t[
                        pair_t
                    ]
                )

                cand = candidate_array(
                    cc,
                    pair_t,
                )


                base_src_set = (
                    original_coord_set(
                        base_src
                    )
                )


                for c in cand:

                    key = (
                        int(c[0]),
                        int(
                            c[1]
                            *
                            ds_arr[0]
                        ),
                        int(
                            c[2]
                            *
                            ds_arr[1]
                        ),
                        int(
                            c[3]
                            *
                            ds_arr[2]
                        ),
                    )

                    assert (
                        key
                        not in base_src_set
                    )


                aug_src = np.concatenate(
                    [
                        base_src,
                        cand,
                    ],
                    axis=0,
                )


                raw, prob = score_context(
                    replay,
                    unet_pair,
                    aug_src,
                    base_tgt,
                )


                tgt_map = (
                    original_coord_map(
                        base_tgt
                    )
                )


                offset = len(
                    base_src
                )


                for j, r in enumerate(
                    cc.itertuples(
                        index=False
                    )
                ):

                    tgt_node = (
                        node_df.loc[
                            int(
                                r.nearest_target_id
                            )
                        ]
                    )


                    tgt_key = (
                        int(tgt_node.t),
                        int(tgt_node.z),
                        int(tgt_node.y),
                        int(tgt_node.x),
                    )


                    assert tgt_key in tgt_map

                    q_idx = int(
                        tgt_map[
                            tgt_key
                        ]
                    )

                    c_idx = (
                        offset
                        +
                        j
                    )


                    outgoing_rows.append({
                        "dataset":
                            name,

                        "t":
                            int(r.t),

                        "z_ds":
                            int(r.z_ds),

                        "y_ds":
                            int(r.y_ds),

                        "x_ds":
                            int(r.x_ds),

                        "out_raw":
                            float(
                                raw[
                                    c_idx,
                                    q_idx,
                                ]
                            ),

                        "out_prob":
                            float(
                                prob[
                                    c_idx,
                                    q_idx,
                                ]
                            ),

                        "out_col_rank":
                            int(
                                rank_desc(
                                    raw[
                                        :,
                                        q_idx,
                                    ],
                                    c_idx,
                                )
                            ),

                        "out_col_margin_raw":
                            float(
                                competitor_margin(
                                    raw[
                                        :,
                                        q_idx,
                                    ],
                                    c_idx,
                                )
                            ),

                        "out_row_rank":
                            int(
                                rank_desc(
                                    raw[
                                        c_idx,
                                        :,
                                    ],
                                    q_idx,
                                )
                            ),

                        "out_row_margin_raw":
                            float(
                                competitor_margin(
                                    raw[
                                        c_idx,
                                        :,
                                    ],
                                    q_idx,
                                )
                            ),
                    })


            # =================================================
            # INCOMING:
            #
            # P995(t) -> [P995(t+1) + candidates(t+1)]
            # =================================================

            target_time = (
                pair_t + 1
            )


            if target_time in candidates_by_t:

                cc = (
                    candidates_by_t[
                        target_time
                    ]
                )

                cand = candidate_array(
                    cc,
                    target_time,
                )


                base_tgt_set = (
                    original_coord_set(
                        base_tgt
                    )
                )


                for c in cand:

                    key = (
                        int(c[0]),
                        int(
                            c[1]
                            *
                            ds_arr[0]
                        ),
                        int(
                            c[2]
                            *
                            ds_arr[1]
                        ),
                        int(
                            c[3]
                            *
                            ds_arr[2]
                        ),
                    )

                    assert (
                        key
                        not in base_tgt_set
                    )


                aug_tgt = np.concatenate(
                    [
                        base_tgt,
                        cand,
                    ],
                    axis=0,
                )


                raw, prob = score_context(
                    replay,
                    unet_pair,
                    base_src,
                    aug_tgt,
                )


                src_map = (
                    original_coord_map(
                        base_src
                    )
                )


                offset = len(
                    base_tgt
                )


                for j, r in enumerate(
                    cc.itertuples(
                        index=False
                    )
                ):

                    src_node = (
                        node_df.loc[
                            int(
                                r.nearest_source_id
                            )
                        ]
                    )


                    src_key = (
                        int(src_node.t),
                        int(src_node.z),
                        int(src_node.y),
                        int(src_node.x),
                    )


                    assert src_key in src_map

                    s_idx = int(
                        src_map[
                            src_key
                        ]
                    )

                    c_idx = (
                        offset
                        +
                        j
                    )


                    incoming_rows.append({
                        "dataset":
                            name,

                        "t":
                            int(r.t),

                        "z_ds":
                            int(r.z_ds),

                        "y_ds":
                            int(r.y_ds),

                        "x_ds":
                            int(r.x_ds),

                        "in_raw":
                            float(
                                raw[
                                    s_idx,
                                    c_idx,
                                ]
                            ),

                        "in_prob":
                            float(
                                prob[
                                    s_idx,
                                    c_idx,
                                ]
                            ),

                        "in_col_rank":
                            int(
                                rank_desc(
                                    raw[
                                        :,
                                        c_idx,
                                    ],
                                    s_idx,
                                )
                            ),

                        "in_col_margin_raw":
                            float(
                                competitor_margin(
                                    raw[
                                        :,
                                        c_idx,
                                    ],
                                    s_idx,
                                )
                            ),

                        "in_row_rank":
                            int(
                                rank_desc(
                                    raw[
                                        s_idx,
                                        :,
                                    ],
                                    c_idx,
                                )
                            ),

                        "in_row_margin_raw":
                            float(
                                competitor_margin(
                                    raw[
                                        s_idx,
                                        :,
                                    ],
                                    c_idx,
                                )
                            ),
                    })


        del (
            imgs,
            unet_batch,
            det_batch,
        )


    # Exactly all T frames must reproduce P995.
    assert parity_frames == replay.T


    dataset_rows.append({
        "dataset":
            name,

        "fold":
            int(
                cc_dataset[
                    "fold"
                ].iloc[0]
            ),

        "b12_candidates":
            int(
                len(
                    cc_dataset
                )
            ),

        "parity_frames":
            int(
                parity_frames
            ),

        "total_frames":
            int(
                replay.T
            ),
    })


    if (
        di == 1
        or di % 10 == 0
        or di == 103
    ):

        print(
            f"{di:3d}/103 | "
            f"{name} | "
            f"B12={len(cc_dataset):,} | "
            f"{(time.time()-started)/60:.1f} min"
        )


    if torch.backends.mps.is_available():

        torch.mps.empty_cache()


# ============================================================
# 6. Assemble exact evidence table
# ============================================================

incoming = pd.DataFrame(
    incoming_rows
)

outgoing = pd.DataFrame(
    outgoing_rows
)


assert (
    incoming[
        key_cols
    ]
    .duplicated()
    .sum()
    == 0
)

assert (
    outgoing[
        key_cols
    ]
    .duplicated()
    .sum()
    == 0
)


evidence = (
    b12.merge(
        incoming,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
    .merge(
        outgoing,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
)


assert len(evidence) == 15074

required = [
    "in_raw",
    "in_prob",
    "in_col_rank",
    "out_raw",
    "out_prob",
    "out_col_rank",
]

assert (
    evidence[
        required
    ]
    .notna()
    .all()
    .all()
)


for col in [
    "in_col_rank",
    "in_row_rank",
    "out_col_rank",
    "out_row_rank",
]:

    evidence[col] = (
        evidence[col]
        .astype(np.int64)
    )


evidence[
    "bridge_min_prob"
] = np.minimum(
    evidence[
        "in_prob"
    ],
    evidence[
        "out_prob"
    ],
)


evidence[
    "bridge_geom_mean_prob"
] = np.sqrt(
    evidence[
        "in_prob"
    ]
    *
    evidence[
        "out_prob"
    ]
)


evidence[
    "reciprocal_col_rank"
] = np.maximum(
    evidence[
        "in_col_rank"
    ],
    evidence[
        "out_col_rank"
    ],
)


# ============================================================
# 7. GT-blind reports
# ============================================================

print(
    "\n========== FULL BRIDGE ASSOCIATION EVIDENCE =========="
)

print(
    "datasets:",
    evidence[
        "dataset"
    ].nunique()
)

print(
    "candidates:",
    len(
        evidence
    )
)


print(
    "\n========== ASSOCIATION QUANTILES =========="
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
    evidence[
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
    "\n========== RECIPROCAL COLUMN RANKS =========="
)

for k in [
    1,
    2,
    3,
]:

    mask = (
        evidence[
            "in_col_rank"
        ].le(k)
        &
        evidence[
            "out_col_rank"
        ].le(k)
    )

    print(
        f"both ranks <= {k}: "
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
        evidence[
            "in_prob"
        ].ge(p)
        &
        evidence[
            "out_prob"
        ].ge(p)
    )

    print(
        f"both probs >= {p:.2f}: "
        f"{int(mask.sum()):,} "
        f"({mask.mean():.3%})"
    )


print(
    "\n========== RANK1 + MIN PROB =========="
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


for p in [
    0.10,
    0.25,
    0.50,
]:

    mask = (
        rank1
        &
        evidence[
            "bridge_min_prob"
        ].ge(p)
    )

    print(
        f"rank1 both + min_prob >= {p:.2f}: "
        f"{int(mask.sum()):,}"
    )


print(
    "\n========== RANK1 COUNTS BY FOLD =========="
)

tmp = evidence[
    rank1
].groupby(
    "fold"
).size()

print(
    tmp.to_string()
)


print(
    "\n========== RANK1 FRACTION BY FOLD =========="
)

fold_total = (
    evidence.groupby(
        "fold"
    )
    .size()
)

fold_rank1 = (
    evidence[
        rank1
    ]
    .groupby(
        "fold"
    )
    .size()
    .reindex(
        fold_total.index,
        fill_value=0,
    )
)

print(
    (
        fold_rank1
        /
        fold_total
    ).to_string()
)


print(
    "\n========== P995 FRAME PARITY =========="
)

dataset_summary = pd.DataFrame(
    dataset_rows
)

print(
    "datasets exact:",
    int(
        (
            dataset_summary[
                "parity_frames"
            ]
            ==
            dataset_summary[
                "total_frames"
            ]
        ).sum()
    ),
    "/",
    len(
        dataset_summary
    ),
)


# ============================================================
# 8. Persist
# ============================================================

evidence.to_pickle(
    OUT
)

dataset_summary.to_pickle(
    DATASET_OUT
)


payload = {
    "stage":
        "12.10E",

    "gt_used":
        False,

    "datasets":
        int(
            evidence[
                "dataset"
            ].nunique()
        ),

    "b12_candidates":
        int(
            len(evidence)
        ),

    "checkpoint_sha256":
        actual_sha,

    "augmentation_contract":
        (
            "joint B12 candidates per frame; "
            "incoming target augmentation and outgoing "
            "source augmentation scored separately"
        ),

    "rank1_both":
        int(
            rank1.sum()
        ),

    "rank1_minprob_025":
        int(
            (
                rank1
                &
                evidence[
                    "bridge_min_prob"
                ].ge(0.25)
            ).sum()
        ),

    "p995_frame_parity_datasets":
        int(
            (
                dataset_summary[
                    "parity_frames"
                ]
                ==
                dataset_summary[
                    "total_frames"
                ]
            ).sum()
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
    DATASET_OUT
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
    "\n========== 12.10E DECISION =========="
)

print(
    "FULL_AUGMENTED_ASSOCIATION_EVIDENCE_EXTRACTED: PASS"
)