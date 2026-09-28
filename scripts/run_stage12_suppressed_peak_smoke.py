# ============================================================
# RUN — Stage 12.10C-S1 P995 Parity + Suppressed-Peak Smoke
# Purpose:
#   Reconstruct production Baseline256 detections on four
#   folds1–4 pilot datasets using the EXACT production path.
#
#   First require exact coordinate parity at det_threshold=.995
#   against persisted P995 GEFF predictions.
#
#   Only after parity succeeds, count additional local maxima at:
#       .99, .98, .95, .90
#
# Changes files: NO
# Runs training: NO
# Runs inference: YES — 4 pilot datasets
# Uses GT: NO
# Modifies graph: NO
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


sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts"),
]


import replay_stage11_association as replay_mod
import run_stage11_multiframe_extract as runner

from replay_stage11_association import DatasetReplay
from predict_unet_transformer_mps import (
    _detect_cells_pooled,
)


# ============================================================
# 1. Fixed smoke thresholds
# ============================================================

THRESHOLDS = [
    0.995,
    0.99,
    0.98,
    0.95,
    0.90,
]


# Exact production threshold contract.
assert abs(
    float(replay_mod.DET_THRESHOLD)
    - 0.995
) < 1e-12


# ============================================================
# 2. One pilot dataset per development fold
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


pilot = (
    corp.groupby(
        "fold",
        as_index=False,
    )
    .first()
    .sort_values("fold")
)

assert len(pilot) == 4


names = pilot[
    "dataset"
].tolist()

fold_map = dict(
    zip(
        pilot["dataset"],
        pilot["fold"],
    )
)


# ============================================================
# 3. Frozen Baseline256 model
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

downsample_arr = np.asarray(
    downsample,
    dtype=np.float32,
)


# ============================================================
# 4. Coordinate helper
#
# Production detector emits downsampled [t,z,y,x].
# predict_video rescales spatial coordinates at the end.
# ============================================================

def to_original_coords(
    arr,
):

    out = arr.astype(
        np.float32,
        copy=True,
    )

    out[:, 1:] *= (
        downsample_arr
    )

    return out.astype(
        np.int64
    )


def coordinate_set(
    arr,
):

    return {
        tuple(
            map(
                int,
                row,
            )
        )
        for row in arr
    }


# ============================================================
# 5. Exact replay
# ============================================================

rows = []

contract_rows = []

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


    # --------------------------------------------------------
    # Production contract must be identical.
    # --------------------------------------------------------

    assert abs(
        float(
            replay.cfg.det_threshold
        )
        - 0.995
    ) < 1e-12


    contract_rows.append({
        "fold":
            int(fold_map[name]),

        "dataset":
            name,

        "det_threshold":
            float(
                replay.cfg.det_threshold
            ),

        "pool_kernel_um":
            float(
                replay.cfg.pool_kernel_um
            ),

        "pool_k":
            str(
                tuple(
                    replay.pool_k
                )
            ),

        "downsample":
            str(
                tuple(
                    replay.downsample
                )
            ),

        "scale":
            str(
                tuple(
                    replay.scale
                )
            ),
    })


    # --------------------------------------------------------
    # Persisted P995 detection coordinates.
    # --------------------------------------------------------

    graph = runner.load_graph(
        P995
        / f"{name}.geff"
    )

    persisted_df = (
        graph.node_attrs(
            attr_keys=[
                "t",
                "z",
                "y",
                "x",
            ]
        )
        .to_pandas()
    )

    persisted_arr = (
        persisted_df[
            ["t", "z", "y", "x"]
        ]
        .to_numpy()
        .astype(np.int64)
    )

    persisted_set = (
        coordinate_set(
            persisted_arr
        )
    )

    # Prediction graph itself should not contain duplicated
    # coordinates.
    assert (
        len(persisted_set)
        == len(persisted_arr)
    )


    # --------------------------------------------------------
    # Reconstruct frame detections using production first-seen
    # semantics:
    #
    #   frame 0  -> first element of [0,1]
    #   frame t  -> second element of [t-1,t], t>=1
    # --------------------------------------------------------

    detected = {
        threshold: []
        for threshold
        in THRESHOLDS
    }


    with torch.no_grad():

        for t in range(
            replay.T
        ):

            if t == 0:

                _, det = (
                    replay._encode(
                        [0, 1]
                    )
                )

                logits = (
                    det[0][0]
                )

            else:

                _, det = (
                    replay._encode(
                        [t - 1, t]
                    )
                )

                logits = (
                    det[1][0]
                )


            for threshold in THRESHOLDS:

                arr_ds = (
                    _detect_cells_pooled(
                        logits,
                        t,
                        threshold,
                        replay.pool_k,
                    )
                )

                arr_original = (
                    to_original_coords(
                        arr_ds
                    )
                )

                detected[
                    threshold
                ].append(
                    arr_original
                )


            del det, logits


    # --------------------------------------------------------
    # Collapse frames.
    # --------------------------------------------------------

    detected_sets = {}


    for threshold in THRESHOLDS:

        arr = np.concatenate(
            detected[threshold],
            axis=0,
        )

        detected_sets[
            threshold
        ] = coordinate_set(
            arr
        )

        assert (
            len(
                detected_sets[
                    threshold
                ]
            )
            == len(arr)
        )


    # --------------------------------------------------------
    # CRITICAL P995 exact parity.
    # --------------------------------------------------------

    recon995 = (
        detected_sets[
            0.995
        ]
    )

    missing = (
        persisted_set
        - recon995
    )

    extra = (
        recon995
        - persisted_set
    )

    exact = (
        len(missing) == 0
        and len(extra) == 0
        and len(recon995)
        == len(persisted_set)
    )


    print(
        f"{name} | "
        f"P995 persisted={len(persisted_set):,} "
        f"reconstructed={len(recon995):,} "
        f"missing={len(missing)} "
        f"extra={len(extra)} "
        f"exact={exact}"
    )


    if not exact:

        print(
            "missing sample:",
            list(
                sorted(missing)
            )[:10],
        )

        print(
            "extra sample:",
            list(
                sorted(extra)
            )[:10],
        )


    assert exact, (
        f"P995 parity failed: {name}"
    )


    # --------------------------------------------------------
    # Nested-threshold contract.
    # Same peak definition; lower threshold can only add peaks.
    # --------------------------------------------------------

    for high, low in zip(
        THRESHOLDS[:-1],
        THRESHOLDS[1:],
    ):

        assert (
            detected_sets[high]
            <=
            detected_sets[low]
        )


    # --------------------------------------------------------
    # Threshold expansion report.
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        current = (
            detected_sets[
                threshold
            ]
        )

        extra_vs_995 = (
            current
            -
            recon995
        )

        rows.append({
            "fold":
                int(
                    fold_map[name]
                ),

            "dataset":
                name,

            "threshold":
                float(
                    threshold
                ),

            "total_peaks":
                int(
                    len(current)
                ),

            "extra_vs_995":
                int(
                    len(
                        extra_vs_995
                    )
                ),

            "extra_ratio_vs_995":
                float(
                    len(
                        extra_vs_995
                    )
                    /
                    max(
                        len(recon995),
                        1,
                    )
                ),
        })


    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ============================================================
# 6. Reports
# ============================================================

contract = pd.DataFrame(
    contract_rows
)

result = pd.DataFrame(
    rows
)


print(
    "\n========== PRODUCTION DETECTION CONTRACT =========="
)

print(
    contract.to_string(
        index=False
    )
)


print(
    "\n========== P995 PARITY =========="
)

print(
    "datasets:",
    len(names)
)

print(
    "exact parity:",
    len(names),
    "/",
    len(names),
)


print(
    "\n========== SUPPRESSED PEAK COUNTS =========="
)

print(
    result.pivot(
        index="dataset",
        columns="threshold",
        values="extra_vs_995",
    )
    .to_string()
)


# ============================================================
# 7. Aggregate expansion
# ============================================================

base_total = int(
    result[
        result[
            "threshold"
        ].eq(0.995)
    ][
        "total_peaks"
    ].sum()
)


agg = (
    result.groupby(
        "threshold"
    )
    .agg(
        total_peaks=(
            "total_peaks",
            "sum",
        ),

        extra_vs_995=(
            "extra_vs_995",
            "sum",
        ),
    )
    .reset_index()
    .sort_values(
        "threshold",
        ascending=False,
    )
)


agg[
    "extra_pct_of_p995"
] = (
    agg[
        "extra_vs_995"
    ]
    /
    base_total
)


print(
    "\n========== AGGREGATE THRESHOLD EXPANSION =========="
)

print(
    agg.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print(
    "\n========== RUNTIME =========="
)

print(
    "elapsed min:",
    f"{(time.time()-started)/60:.1f}"
)


print(
    "\n========== 12.10C-S1 DECISION =========="
)

print(
    "P995_PARITY_AND_SUPPRESSED_PEAK_SMOKE: PASS"
)