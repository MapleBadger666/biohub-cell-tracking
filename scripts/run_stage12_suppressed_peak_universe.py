# ============================================================
# RUN — Stage 12.10C Full Suppressed-Peak Universe
# Purpose:
#   Extract the COMPLETE GT-blind Baseline256 suppressed-peak
#   universe on folds1–4 6bba datasets.
#
#   Candidate definition:
#
#       production local maximum
#       sigmoid(det_logit) > 0.90
#       AND coordinate absent from persisted P995 detections
#
#   Store detector probability and exact coordinates so all
#   later threshold policies can be derived without rerunning
#   the detector.
#
#   Critical fidelity contract:
#       reconstructed probability > .995 coordinates must
#       exactly equal persisted P995 coordinates PER DATASET.
#
# Changes files: YES — Stage12 candidate artifacts
# Runs training: NO
# Runs inference: YES — 103 development datasets
# Uses GT: NO
# Modifies prediction graphs: NO
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
import torch.nn.functional as F


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

CKPT = (
    PROJECT
    / "models/official_baseline/"
      "stage10_pilot_baseline_256/"
      "split_0/edge_predictor_best.pth"
)

CV_PATH = (
    PROJECT
    / "configs/cv_manifest.csv"
)

OUT_DIR = (
    PROJECT
    / "reports/stage12_residual_audit"
)

UNIVERSE_OUT = (
    OUT_DIR
    / "stage12_folds14_suppressed_peak_universe.pkl"
)

DATASET_OUT = (
    OUT_DIR
    / "stage12_folds14_suppressed_peak_dataset_summary.pkl"
)

SUMMARY_OUT = (
    OUT_DIR
    / "stage12_folds14_suppressed_peak_universe_summary.json"
)


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]


import run_stage11_multiframe_extract as runner

from replay_stage11_association import (
    DatasetReplay,
)

from predict_unet_transformer_mps import (
    _load_frame,
)

from stage11_multiframe_utils import (
    normalize_window,
    encode_windows_with_tta,
)


# ============================================================
# 1. Locked extraction contract
# ============================================================

FLOOR = 0.90
PERSISTED_THRESHOLD = 0.995
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
# 2. Development corpus
# ============================================================

cv = pd.read_csv(
    CV_PATH
)

corp = (
    cv[
        cv["fold"].isin(
            [1, 2, 3, 4]
        )
        &
        cv["dataset"].str.startswith(
            "6bba_"
        )
    ]
    .sort_values(
        ["fold", "dataset"]
    )
    .reset_index(drop=True)
)

assert len(corp) == 103

names = corp[
    "dataset"
].tolist()

fold_map = dict(
    zip(
        corp["dataset"],
        corp["fold"],
    )
)


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

assert tuple(downsample) == (
    1,
    4,
    4,
)

ds_arr = np.asarray(
    downsample,
    dtype=np.float32,
)


# ============================================================
# 4. Helpers
# ============================================================

def coordinate_set(arr):

    return {
        tuple(
            int(x)
            for x in row
        )
        for row in arr
    }


def extract_peaks_with_scores(
    det_logits,
    t,
    pool_k,
):

    # det_logits:
    #     (1, Z, Y, X)
    #
    # EXACT production local-max definition.
    logits = det_logits.unsqueeze(
        0
    )

    pad = tuple(
        k // 2
        for k in pool_k
    )

    pooled = F.max_pool3d(
        logits,
        pool_k,
        stride=1,
        padding=pad,
    )

    probs = torch.sigmoid(
        logits
    )

    is_peak = (
        (logits == pooled)
        &
        (probs > FLOOR)
    )

    idx = torch.nonzero(
        is_peak[0, 0]
    )

    if len(idx) == 0:

        return pd.DataFrame(
            columns=[
                "t",
                "z_ds",
                "y_ds",
                "x_ds",
                "z",
                "y",
                "x",
                "detector_prob",
                "detector_logit",
            ]
        )


    scores = probs[
        0,
        0,
        idx[:, 0],
        idx[:, 1],
        idx[:, 2],
    ]

    raw_scores = logits[
        0,
        0,
        idx[:, 0],
        idx[:, 1],
        idx[:, 2],
    ]


    xyz_ds = (
        idx.detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    xyz = (
        xyz_ds.astype(
            np.float32
        )
        *
        ds_arr
    ).astype(np.int64)


    return pd.DataFrame({
        "t":
            np.full(
                len(idx),
                int(t),
                dtype=np.int64,
            ),

        "z_ds":
            xyz_ds[:, 0],

        "y_ds":
            xyz_ds[:, 1],

        "x_ds":
            xyz_ds[:, 2],

        "z":
            xyz[:, 0],

        "y":
            xyz[:, 1],

        "x":
            xyz[:, 2],

        "detector_prob":
            scores.detach()
            .cpu()
            .numpy()
            .astype(np.float32),

        "detector_logit":
            raw_scores.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
    })


# ============================================================
# 5. Full extraction
# ============================================================

candidate_parts = []
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


    assert abs(
        replay.cfg.det_threshold
        -
        PERSISTED_THRESHOLD
    ) < 1e-12

    assert tuple(
        replay.pool_k
    ) == (
        3,
        3,
        3,
    )


    # --------------------------------------------------------
    # Persisted production P995 node coordinates
    # --------------------------------------------------------

    graph = runner.load_graph(
        P995
        / f"{name}.geff"
    )

    persisted = (
        graph.node_attrs(
            attr_keys=[
                "t",
                "z",
                "y",
                "x",
            ]
        )
        .to_pandas()[
            [
                "t",
                "z",
                "y",
                "x",
            ]
        ]
        .to_numpy()
        .astype(np.int64)
    )

    persisted_set = (
        coordinate_set(
            persisted
        )
    )

    assert (
        len(persisted_set)
        == len(persisted)
    )


    # --------------------------------------------------------
    # Production first-seen semantics, batched:
    #
    # pair t = [t, t+1]
    #
    # frame 0 comes from first element of pair 0.
    # frame k>=1 comes from second element of pair k-1.
    # --------------------------------------------------------

    frame_parts = []

    time_indices = list(
        range(
            replay.T - 1
        )
    )


    with torch.no_grad():

        for start in range(
            0,
            len(time_indices),
            BATCH_SIZE,
        ):

            batch = time_indices[
                start:
                start + BATCH_SIZE
            ]

            windows = []

            for t in batch:

                pair = torch.stack(
                    [
                        _load_frame(
                            replay.zarr,
                            ti,
                            replay.target_shape,
                            replay.downsample,
                        )
                        for ti in (
                            t,
                            t + 1,
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

            _, det = (
                encode_windows_with_tta(
                    model,
                    imgs,
                )
            )


            for b, t in enumerate(
                batch
            ):

                if t == 0:

                    frame_parts.append(
                        extract_peaks_with_scores(
                            det[b, 0],
                            0,
                            replay.pool_k,
                        )
                    )


                frame_parts.append(
                    extract_peaks_with_scores(
                        det[b, 1],
                        t + 1,
                        replay.pool_k,
                    )
                )


            del imgs, det


    all_peaks = pd.concat(
        frame_parts,
        ignore_index=True,
    )


    assert (
        all_peaks[
            [
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


    # --------------------------------------------------------
    # P995 exact parity is a HARD fidelity gate.
    # --------------------------------------------------------

    recon995 = (
        all_peaks[
            all_peaks[
                "detector_prob"
            ].gt(
                PERSISTED_THRESHOLD
            )
        ][
            [
                "t",
                "z",
                "y",
                "x",
            ]
        ]
        .to_numpy()
        .astype(np.int64)
    )

    recon995_set = (
        coordinate_set(
            recon995
        )
    )


    missing = (
        persisted_set
        -
        recon995_set
    )

    extra = (
        recon995_set
        -
        persisted_set
    )


    assert not missing, (
        f"{name}: P995 reconstruction "
        f"missing {len(missing)} coordinates"
    )

    assert not extra, (
        f"{name}: P995 reconstruction "
        f"has {len(extra)} extra coordinates"
    )

    assert (
        len(recon995_set)
        ==
        len(persisted_set)
    )


    # --------------------------------------------------------
    # Suppressed universe = all floor peaks absent from P995.
    #
    # Use exact coordinate membership rather than only
    # probability arithmetic.
    # --------------------------------------------------------

    tuples = list(
        zip(
            all_peaks["t"],
            all_peaks["z"],
            all_peaks["y"],
            all_peaks["x"],
        )
    )

    is_persisted = np.fromiter(
        (
            tuple(
                map(
                    int,
                    q,
                )
            )
            in persisted_set
            for q in tuples
        ),
        dtype=bool,
        count=len(tuples),
    )


    cand = (
        all_peaks[
            ~is_persisted
        ]
        .copy()
        .reset_index(drop=True)
    )


    # No candidate may already be a P995 node.
    assert all(
        tuple(
            map(
                int,
                q,
            )
        )
        not in persisted_set

        for q in zip(
            cand["t"],
            cand["z"],
            cand["y"],
            cand["x"],
        )
    )


    cand.insert(
        0,
        "dataset",
        name,
    )

    cand.insert(
        1,
        "fold",
        int(
            fold_map[name]
        ),
    )


    # Threshold membership is derived later from probability.
    # Store one useful display band only.
    cand["release_band"] = pd.cut(
        cand["detector_prob"],
        bins=[
            FLOOR,
            0.95,
            0.98,
            0.99,
            PERSISTED_THRESHOLD,
        ],
        labels=[
            "P900_P950",
            "P950_P980",
            "P980_P990",
            "P990_P995",
        ],
        include_lowest=False,
        right=True,
    ).astype(str)


    candidate_parts.append(
        cand
    )


    dataset_rows.append({
        "dataset":
            name,

        "fold":
            int(
                fold_map[name]
            ),

        "p995_nodes":
            int(
                len(
                    persisted_set
                )
            ),

        "floor_peaks":
            int(
                len(
                    all_peaks
                )
            ),

        "suppressed_candidates":
            int(
                len(cand)
            ),

        "extra_at_099":
            int(
                cand[
                    "detector_prob"
                ].gt(0.99)
                .sum()
            ),

        "extra_at_098":
            int(
                cand[
                    "detector_prob"
                ].gt(0.98)
                .sum()
            ),

        "extra_at_095":
            int(
                cand[
                    "detector_prob"
                ].gt(0.95)
                .sum()
            ),

        "extra_at_090":
            int(
                len(cand)
            ),

        "parity_missing":
            0,

        "parity_extra":
            0,
    })


    if (
        di == 1
        or di % 10 == 0
        or di == len(names)
    ):

        print(
            f"{di:3d}/{len(names)} | "
            f"{name} | "
            f"P995={len(persisted_set):,} | "
            f"suppressed={len(cand):,} | "
            f"{(time.time()-started)/60:.1f} min"
        )


    if torch.backends.mps.is_available():

        torch.mps.empty_cache()


# ============================================================
# 6. Aggregate universe
# ============================================================

universe = pd.concat(
    candidate_parts,
    ignore_index=True,
)

dataset_summary = pd.DataFrame(
    dataset_rows
)


assert (
    universe[
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


# ============================================================
# 7. GT-blind reports
# ============================================================

print(
    "\n========== FULL SUPPRESSED-PEAK UNIVERSE =========="
)

print(
    "datasets:",
    universe[
        "dataset"
    ].nunique()
)

print(
    "candidates:",
    len(universe)
)

print(
    "median candidates/dataset:",
    float(
        dataset_summary[
            "suppressed_candidates"
        ].median()
    )
)

print(
    "min candidates/dataset:",
    int(
        dataset_summary[
            "suppressed_candidates"
        ].min()
    )
)

print(
    "max candidates/dataset:",
    int(
        dataset_summary[
            "suppressed_candidates"
        ].max()
    )
)


print(
    "\n========== DETECTOR PROBABILITY QUANTILES =========="
)

print(
    universe[
        "detector_prob"
    ]
    .quantile(
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
    .to_string()
)


print(
    "\n========== NESTED THRESHOLD COUNTS =========="
)

for threshold in [
    0.99,
    0.98,
    0.95,
    0.90,
]:

    n = int(
        universe[
            "detector_prob"
        ]
        .gt(threshold)
        .sum()
    )

    print(
        f"p > {threshold:.2f}: "
        f"{n:,}"
    )


print(
    "\n========== COUNTS BY FOLD =========="
)

print(
    universe.groupby(
        "fold"
    )
    .size()
    .to_string()
)


print(
    "\n========== RELEASE BANDS =========="
)

print(
    universe[
        "release_band"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n========== P995 PARITY =========="
)

print(
    "datasets exact:",
    int(
        (
            (
                dataset_summary[
                    "parity_missing"
                ] == 0
            )
            &
            (
                dataset_summary[
                    "parity_extra"
                ] == 0
            )
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

universe.to_pickle(
    UNIVERSE_OUT
)

dataset_summary.to_pickle(
    DATASET_OUT
)


payload = {
    "stage":
        "12.10C",

    "dataset_family":
        "6bba",

    "folds":
        [1, 2, 3, 4],

    "datasets":
        int(
            dataset_summary[
                "dataset"
            ].nunique()
        ),

    "checkpoint_sha256":
        actual_sha,

    "candidate_floor":
        FLOOR,

    "production_threshold":
        PERSISTED_THRESHOLD,

    "pool_kernel_um":
        3.0,

    "pool_kernel":
        [3, 3, 3],

    "downsample":
        list(
            map(
                int,
                downsample,
            )
        ),

    "candidates":
        int(
            len(universe)
        ),

    "p995_parity_datasets":
        int(
            len(
                dataset_summary
            )
        ),

    "gt_used":
        False,
}


UNIVERSE_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_OUT.write_text(
    json.dumps(
        payload,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "\nwritten:",
    UNIVERSE_OUT
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
    "\n========== 12.10C DECISION =========="
)

print(
    "FULL_SUPPRESSED_PEAK_UNIVERSE_EXTRACTED: PASS"
)