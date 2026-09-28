# ============================================================
# SETUP — Stage12 Iterative Reciprocal Utilities
# Purpose:
#   Reusable deployment-valid utilities for second-round
#   reciprocal association recovery after MUTUAL-CONT-V1.
#
# Changes files: YES — this helper module itself
# Runs training/inference: NO at import
# Uses GT: NO
# Keep after running: YES
# ============================================================

import numpy as np
import torch

from replay_stage11_association import (
    DatasetReplay,
    extract_pos_features,
)


def frame_nodes(nodes, t):
    return (
        nodes[
            nodes["t"].astype(int).eq(t)
        ]
        .sort_values("node_id")
        .reset_index(drop=True)
    )


def fast_association_pair(
    replay,
    raw_nodes,
    t,
    model,
    device,
    return_raw=False,
):
    """
    Bit-equivalent association feature path verified in 12.03C-D2.

    Detection coordinates come from persisted P995 nodes.
    Detection TTA is skipped because only association UNet
    features are required.
    """

    src = frame_nodes(raw_nodes, t)
    tgt = frame_nodes(raw_nodes, t + 1)

    src_ids = (
        src["node_id"]
        .astype(int)
        .to_numpy()
    )

    tgt_ids = (
        tgt["node_id"]
        .astype(int)
        .to_numpy()
    )

    if len(src_ids) == 0 or len(tgt_ids) == 0:
        if return_raw:
            return src_ids, tgt_ids, None, None
        return src_ids, tgt_ids, None

    ds = np.asarray(
        replay.downsample,
        dtype=np.float32,
    )

    c_src = np.column_stack([
        np.full(
            len(src),
            t,
            dtype=np.float32,
        ),
        src[["z", "y", "x"]]
        .to_numpy(dtype=np.float32)
        / ds,
    ])

    c_tgt = np.column_stack([
        np.full(
            len(tgt),
            t + 1,
            dtype=np.float32,
        ),
        tgt[["z", "y", "x"]]
        .to_numpy(dtype=np.float32)
        / ds,
    ])

    with torch.no_grad():

        old_tta = replay.cfg.det_tta
        replay.cfg.det_tta = False

        try:
            unet_out, _ = replay._encode(
                [t, t + 1]
            )
        finally:
            replay.cfg.det_tta = old_tta

        p_src = (
            torch.from_numpy(
                c_src[:, 1:]
                .astype(np.float32)
            )
            .unsqueeze(0)
            .to(device)
        )

        p_tgt = (
            torch.from_numpy(
                c_tgt[:, 1:]
                .astype(np.float32)
            )
            .unsqueeze(0)
            .to(device)
        )

        src_rel = c_src.copy()
        tgt_rel = c_tgt.copy()

        src_rel[:, 0] = 0
        tgt_rel[:, 0] = 1

        window_shape = (
            replay.window,
            *replay.image_shape[1:],
        )

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
            len(src_ids),
            dtype=torch.bool,
            device=device,
        )

        mask_tgt = torch.ones(
            1,
            len(tgt_ids),
            dtype=torch.bool,
            device=device,
        )

        feat_src = model._index_features(
            unet_out[:, 0],
            p_src,
            mask_src,
        )

        feat_tgt = model._index_features(
            unet_out[:, 1],
            p_tgt,
            mask_tgt,
        )

        raw = model.predict_edges(
            feat_src,
            feat_tgt,
            p_src * replay.ds_arr_t,
            p_tgt * replay.ds_arr_t,
            pos_src,
            pos_tgt,
            mask_src,
            mask_tgt,
        )[0]

        raw_np = raw.cpu().numpy()

        probs = torch.softmax(
            raw,
            dim=0,
        ).cpu().numpy()

    if return_raw:
        return src_ids, tgt_ids, raw_np, probs

    return src_ids, tgt_ids, probs


def residual_reciprocal_pairs(
    src_ids,
    tgt_ids,
    probs,
    free_sources,
    free_targets,
):
    """
    Round-2 reciprocal matching.

    The rank is recomputed only inside the remaining deployment-
    valid free-source × free-target submatrix.

    Returns:
        (source_id, target_id, original_probability)
    """

    if probs is None:
        return []

    src_idx = np.asarray(
        [
            i
            for i, node_id in enumerate(src_ids)
            if int(node_id) in free_sources
        ],
        dtype=int,
    )

    tgt_idx = np.asarray(
        [
            j
            for j, node_id in enumerate(tgt_ids)
            if int(node_id) in free_targets
        ],
        dtype=int,
    )

    if len(src_idx) == 0 or len(tgt_idx) == 0:
        return []

    sub = probs[
        np.ix_(
            src_idx,
            tgt_idx,
        )
    ]

    row_top = np.argmax(
        sub,
        axis=1,
    )

    col_top = np.argmax(
        sub,
        axis=0,
    )

    pairs = []

    for a, b in enumerate(row_top):

        if int(col_top[b]) != a:
            continue

        i = int(src_idx[a])
        j = int(tgt_idx[b])

        pairs.append(
            (
                int(src_ids[i]),
                int(tgt_ids[j]),
                float(probs[i, j]),
            )
        )

    return pairs