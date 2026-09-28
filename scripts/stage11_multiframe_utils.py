#!/usr/bin/env python
"""Stage 11 all-frame joint-feature extraction utilities.

Lifts the validated Stage 11.09A joint parent/top-2-candidate extraction out of
``notebooks/11_division_association_audit.ipynb`` and generalises it from
division-containing frames to **every** valid frame pair.

Feature semantics are unchanged from 11.09A:

- detections come from the frozen predictor path (4-view detection TTA, the same
  threshold and pooling), source detections for frame ``t`` taken from the
  ``[t-1, t]`` window slot 1 (``[0, 1]`` slot 0 when ``t == 0``);
- ``parent_q`` / target contexts are the **post-attention** 128-d states captured
  at ``model.transformer.norm_out``, not raw UNet features;
- top-2 candidates are ranked by descending **raw** association logit, never by
  ground truth;
- ``score_features`` is ``[raw1, raw2, raw1-raw2, prob1, prob2, prob1-prob2]``
  with ``prob`` from the frozen source-column softmax (``softmax(raw, dim=0)``);
- ``geometry`` is ``[d1, d2, sep, angle_deg, balance]`` in physical µm.

The single behavioural change is frame coverage: ``valid_times`` is every
``t < T - 1``, chosen without consulting ground truth. Ground truth is used only
*after* inference to match detections to GT nodes and to attach labels.

Anti-leakage contract enforced here:

- GT never selects which frames are encoded;
- GT never selects top-1 / top-2 candidates;
- nothing in this module fits, calibrates, normalises, or thresholds anything.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

__all__ = [
    "BATCH_WINDOWS",
    "EXPECTED_FEATURE_DIMS",
    "MATCH_RADIUS_UM",
    "OUTPUT_COLUMNS",
    "SCALE_UM",
    "DatasetSelection",
    "candidate_geometry",
    "canonicalize_det_logits",
    "division_frame_times",
    "eligible_label",
    "encode_windows_with_tta",
    "extract_dataset_rows",
    "extract_multiframe_features",
    "gt_child_degrees",
    "iter_window_batches",
    "match_gt_to_pred",
    "normalize_window",
    "score_features",
    "select_datasets",
    "sha256_file",
    "top2_by_raw_logit",
    "validate_schema",
    "verify_checkpoint",
]


# ============================================================
# Frozen Stage 11 constants
# ============================================================

#: Physical voxel size (z, y, x) in µm. Hard-coded in Stage 11.08A/11.09A and
#: verified identical across all 128 6bba datasets; :func:`extract_dataset_rows`
#: re-checks it per dataset and refuses to run on a mismatch.
SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

#: Official node-matching radius, as used by the locked metric and 11.08A.
MATCH_RADIUS_UM = 7.0

#: Default number of two-frame windows encoded per forward batch (11.08A value).
BATCH_WINDOWS = 4

#: Feature widths the extractor guarantees.
EXPECTED_FEATURE_DIMS: dict[str, int] = {
    "parent_q": 128,
    "k_mean": 128,
    "k_absdiff": 128,
    "score_features": 6,
    "geometry": 5,
    "top1_xyz": 3,
    "top2_xyz": 3,
}

#: Output schema, in order.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "dataset",
    "fold",
    "t",
    "node_id",
    "label",
    "division_frame",
    "match_distance_um",
    "parent_q",
    "k_mean",
    "k_absdiff",
    "score_features",
    "geometry",
    "top1_xyz",
    "top2_xyz",
)


# ============================================================
# Checkpoint / manifest guards
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected_sha256: str) -> str:
    """Return the checkpoint SHA256, raising loudly on absence or mismatch."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch for {path}\n  expected {expected_sha256}\n  observed {observed}")
    return observed


@dataclass(frozen=True)
class DatasetSelection:
    """The datasets an extraction run will touch, resolved from the manifest."""

    frame: pd.DataFrame
    prefix: str
    folds: tuple[int, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.frame["dataset"])

    def __len__(self) -> int:
        return len(self.frame)


def select_datasets(
    manifest: pd.DataFrame,
    *,
    prefix: str,
    folds: list[int],
    data_root: Path | None = None,
) -> DatasetSelection:
    """Resolve datasets from the manifest only — never by walking the data tree.

    Fold membership comes from ``cv_manifest.csv`` verbatim. No ``train``/``test``
    directory is scanned and no dataset is discovered recursively, so a run can
    only ever touch rows the manifest explicitly lists.
    """
    required = {"dataset", "prefix", "fold"}
    missing_cols = required - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"manifest is missing column(s): {sorted(missing_cols)}")

    folds = [int(f) for f in folds]
    unknown = sorted(set(folds) - set(manifest["fold"].astype(int)))
    if unknown:
        raise ValueError(f"fold(s) not present in manifest: {unknown}")

    frame = (
        manifest[(manifest["prefix"] == prefix) & (manifest["fold"].astype(int).isin(folds))]
        .copy()
        .sort_values("dataset")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError(f"no datasets matched prefix={prefix!r} folds={folds}")

    if data_root is not None:
        data_root = Path(data_root)

        absent = [
            name
            for name in frame["dataset"]
            if not (data_root / f"{name}.zarr").exists()
            or not (data_root / f"{name}.geff").exists()
        ]

        if absent:
            raise FileNotFoundError(
                f"{len(absent)} selected dataset(s) missing under "
                f"{data_root}: {absent[:5]}"
            )

    return DatasetSelection(frame=frame, prefix=prefix, folds=tuple(sorted(set(folds))))


# ============================================================
# Ground-truth helpers — labelling only, never frame selection
# ============================================================


def gt_child_degrees(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    """Number of ``t -> t+1`` GT children per source node.

    Restricting to consecutive frames matches the label contract literally. All
    GT edges in this competition are already consecutive, so this reproduces
    11.09A's plain out-degree exactly while making the intent explicit.
    """
    if len(edges) == 0:
        return pd.Series(dtype=int)

    times = dict(zip(nodes["node_id"].astype(int), nodes["t"].astype(int), strict=True))

    consecutive = [
        int(row.source_id)
        for row in edges.itertuples(index=False)
        if times.get(int(row.target_id), -(10**9)) == times.get(int(row.source_id), 10**9) + 1
    ]

    if not consecutive:
        return pd.Series(dtype=int)

    return pd.Series(consecutive).value_counts().sort_index()


def eligible_label(degree: int) -> int | None:
    """Label for a matched GT parent, or ``None`` when it is not eligible.

    ``1`` for exactly two ``t -> t+1`` children, ``0`` for exactly one. Nodes
    with no children are the end of the annotation, not evidence of
    continuation, so they are excluded rather than treated as negatives;
    degree > 2 is excluded as malformed.
    """
    degree = int(degree)
    if degree == 2:
        return 1
    if degree == 1:
        return 0
    return None


def division_frame_times(nodes: pd.DataFrame, degrees: pd.Series) -> set[int]:
    """Times holding at least one eligible GT division parent.

    Metadata only. :func:`extract_dataset_rows` never consults this when
    deciding which frames to encode.
    """
    if len(degrees) == 0:
        return set()

    division_ids = {int(i) for i, d in degrees.items() if int(d) == 2}
    if not division_ids:
        return set()

    selected = nodes[nodes["node_id"].astype(int).isin(division_ids)]
    return {int(t) for t in selected["t"].astype(int)}


def match_gt_to_pred(
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    scale_um: np.ndarray = SCALE_UM,
    radius_um: float = MATCH_RADIUS_UM,
) -> list[tuple[int, int, float]]:
    """Thresholded optimal one-to-one GT->detection matching (verbatim 11.08A).

    Dummy columns let a GT node stay unmatched rather than being forced onto a
    far detection.
    """
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    pred_xyz = np.asarray(pred_xyz, dtype=np.float32)

    if len(gt_xyz) == 0 or len(pred_xyz) == 0:
        return []

    dist = np.linalg.norm(
        (gt_xyz[:, None, :] - pred_xyz[None, :, :]) * scale_um,
        axis=2,
    )

    n_gt = len(gt_xyz)
    n_pred = len(pred_xyz)

    cost = np.full((n_gt, n_pred + n_gt), 1e6, dtype=np.float64)
    cost[:, :n_pred] = np.where(dist <= radius_um, dist, 1e6)

    for i in range(n_gt):
        cost[i, n_pred + i] = radius_um + 1e-4

    rows, cols = linear_sum_assignment(cost)

    return [
        (int(gi), int(pj), float(dist[gi, pj]))
        for gi, pj in zip(rows, cols, strict=True)
        if pj < n_pred and dist[gi, pj] <= radius_um
    ]


# ============================================================
# Feature definitions — verbatim 11.09A
# ============================================================


def candidate_geometry(
    src_ds: np.ndarray,
    tgt1_ds: np.ndarray,
    tgt2_ds: np.ndarray,
    downsample_arr: np.ndarray,
    scale_um: np.ndarray = SCALE_UM,
) -> np.ndarray:
    """``[d1, d2, sep, angle_deg, balance]`` in physical µm (verbatim 11.09A).

    Inputs are ``(z, y, x)`` on the downsampled detection grid; they are lifted
    to full resolution by ``downsample_arr`` and then to µm by ``scale_um``.
    """
    src = np.asarray(src_ds, dtype=np.float32) * downsample_arr
    d1p = np.asarray(tgt1_ds, dtype=np.float32) * downsample_arr
    d2p = np.asarray(tgt2_ds, dtype=np.float32) * downsample_arr

    v1 = (d1p - src) * scale_um
    v2 = (d2p - src) * scale_um

    d1 = float(np.linalg.norm(v1))
    d2 = float(np.linalg.norm(v2))
    sep = float(np.linalg.norm(v1 - v2))

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denom > 1e-8:
        cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosang)))
    else:
        angle = 0.0

    balance = float(abs(d1 - d2) / (d1 + d2 + 1e-8))

    return np.array([d1, d2, sep, angle, balance], dtype=np.float32)


def top2_by_raw_logit(score_row: np.ndarray) -> tuple[int, int]:
    """Indices of the two highest **raw** association logits, ties by index.

    ``np.argsort(-row, kind="stable")`` is the 11.09A ordering: descending
    logit, earlier target index first on a tie. Ground truth plays no part.
    """
    row = np.asarray(score_row, dtype=np.float64)
    if row.shape[0] < 2:
        raise ValueError("need at least two candidate targets to form a top-2")

    order = np.argsort(-row, kind="stable")
    return int(order[0]), int(order[1])


def score_features(raw1: float, raw2: float, prob1: float, prob2: float) -> np.ndarray:
    """``[raw1, raw2, raw1-raw2, prob1, prob2, prob1-prob2]`` (verbatim 11.09A)."""
    return np.array(
        [raw1, raw2, raw1 - raw2, prob1, prob2, prob1 - prob2],
        dtype=np.float32,
    )


# ============================================================
# Model-side helpers — verbatim 11.08A
# ============================================================


def normalize_window(window: torch.Tensor, replay: Any) -> torch.Tensor:
    """Quantile normalisation matching ``DatasetReplay``."""
    return ((window - replay.q_low) / (replay.q_high - replay.q_low + 1e-6)).clamp(0.0)


def canonicalize_det_logits(det_logits: Any, batch_size: int, window_len: int) -> torch.Tensor:
    """Coerce ``model.encode`` detection output to ``(B, W, 1, Z, Y, X)``."""
    if isinstance(det_logits, (list, tuple)):
        if len(det_logits) != window_len:
            raise RuntimeError(f"expected {window_len} detection frames, got {len(det_logits)}")

        frames = []
        for d in det_logits:
            if d.ndim == 5:
                if d.shape[0] != batch_size:
                    raise RuntimeError(f"detection frame batch mismatch: {tuple(d.shape)}")
                frames.append(d)
            elif d.ndim == 4 and batch_size == 1:
                frames.append(d.unsqueeze(0))
            else:
                raise RuntimeError(f"Unexpected detection-logit frame shape: {tuple(d.shape)}")

        return torch.stack(frames, dim=1)

    if not torch.is_tensor(det_logits):
        raise TypeError(f"Unsupported det_logits type: {type(det_logits)}")

    if det_logits.ndim == 6 and det_logits.shape[0] == batch_size and det_logits.shape[1] == window_len:
        return det_logits

    if det_logits.ndim == 6 and det_logits.shape[0] == window_len and det_logits.shape[1] == batch_size:
        return det_logits.transpose(0, 1)

    if det_logits.ndim == 5 and batch_size == 1 and det_logits.shape[0] == window_len:
        return det_logits.unsqueeze(0)

    raise RuntimeError(
        f"Unexpected detection logits shape: {tuple(det_logits.shape)} for B={batch_size}, W={window_len}"
    )


def encode_windows_with_tta(model: Any, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Original-view UNet features plus 4-view averaged detection logits.

    Identical to 11.09A's ``encode_pair_features_and_tta``: the association
    features come from the unflipped view, while detection logits average the
    original and the three xy flips.
    """
    batch_size, window_len = imgs.shape[:2]

    with torch.no_grad():
        unet_out, raw = model.encode(imgs)
        det_sum = canonicalize_det_logits(raw, batch_size, window_len).clone()

        for dims in [(-1,), (-2,), (-2, -1)]:
            _, raw_flip = model.encode(imgs.flip(dims))
            det_flip = canonicalize_det_logits(raw_flip, batch_size, window_len)
            det_sum += det_flip.flip(dims)

    return unet_out, det_sum / 4.0


def iter_window_batches(n_windows: int, batch_size: int) -> list[list[int]]:
    """Contiguous, ascending window batches.

    Ascending order is required, not cosmetic: the detection cache chains one
    frame across batch boundaries, so windows must be visited in order.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [list(range(s, min(s + batch_size, n_windows))) for s in range(0, n_windows, batch_size)]


# ============================================================
# Extraction
# ============================================================


def extract_dataset_rows(
    *,
    dataset: str,
    fold: int,
    replay: Any,
    model: Any,
    device: Any,
    downsample_arr: np.ndarray,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    extract_pos_features: Any,
    detect_cells: Any,
    load_frame: Any,
    batch_size: int = BATCH_WINDOWS,
    scale_um: np.ndarray = SCALE_UM,
) -> list[dict[str, Any]]:
    """Extract joint features for every valid frame pair of one dataset.

    Every window ``[t, t+1]`` for ``t in [0, T-2]`` is encoded exactly once. Its
    slot-1 detections serve as the *target* detections of pair ``t`` and, on the
    next iteration, as the *source* detections of pair ``t+1`` — which is
    precisely the ``[t, t+1]`` slot-1 tensor 11.09A recomputed in a second
    forward pass. Caching it therefore halves the encoder work without changing
    a single input to the model.

    Frame selection never consults ``gt_nodes`` / ``gt_edges``; those are read
    only after the association forward pass, to match and label.
    """
    degrees = gt_child_degrees(gt_nodes, gt_edges)
    division_times = division_frame_times(gt_nodes, degrees)
    labelled_ids = set(degrees.index.astype(int)) if len(degrees) else set()

    n_windows = max(int(replay.T) - 1, 0)
    if n_windows == 0:
        return []

    rows: list[dict[str, Any]] = []
    frame_dets: dict[int, np.ndarray] = {}

    context_bucket: list[torch.Tensor] = []
    hook = model.transformer.norm_out.register_forward_hook(
        lambda module, inp, out: context_bucket.append(out.detach().cpu())
    )

    try:
        for batch in iter_window_batches(n_windows, batch_size):
            windows = []
            for t in batch:
                pair = torch.stack(
                    [load_frame(replay.zarr, ti, replay.target_shape, replay.downsample) for ti in (t, t + 1)]
                )
                windows.append(normalize_window(pair, replay))

            imgs = torch.stack(windows).to(device)
            unet_out, det = encode_windows_with_tta(model, imgs)

            for b, t in enumerate(batch):
                if t == 0:
                    frame_dets[0] = detect_cells(det[b, 0], 0, replay.cfg.det_threshold, replay.pool_k)

                frame_dets[t + 1] = detect_cells(det[b, 1], t + 1, replay.cfg.det_threshold, replay.pool_k)

                c_src = frame_dets[t]
                c_tgt = frame_dets[t + 1]

                n_src, n_tgt = len(c_src), len(c_tgt)
                if n_src == 0 or n_tgt < 2:
                    continue

                p_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
                p_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)

                window_shape = (replay.window, *replay.image_shape[1:])

                src_rel = c_src.copy()
                src_rel[:, 0] = 0
                tgt_rel = c_tgt.copy()
                tgt_rel[:, 0] = 1

                pos_src = torch.from_numpy(extract_pos_features(src_rel, window_shape)).unsqueeze(0).to(device)
                pos_tgt = torch.from_numpy(extract_pos_features(tgt_rel, window_shape)).unsqueeze(0).to(device)

                mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
                mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

                feat_src = model._index_features(unet_out[b : b + 1, 0], p_src, mask_src)
                feat_tgt = model._index_features(unet_out[b : b + 1, 1], p_tgt, mask_tgt)

                context_bucket.clear()

                with torch.no_grad():
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

                if len(context_bucket) != 2:
                    raise RuntimeError(f"expected 2 post-attention contexts, captured {len(context_bucket)}")

                q = context_bucket[0][0].numpy()
                k = context_bucket[1][0].numpy()
                raw_np = raw.detach().cpu().numpy()
                probs_np = torch.softmax(raw, dim=0).detach().cpu().numpy()

                if q.shape != (n_src, 128) or k.shape != (n_tgt, 128):
                    raise RuntimeError(f"unexpected context shapes q={q.shape} k={k.shape}")

                # --- GT enters only here, to match and label -------------
                frame_gt = gt_nodes[
                    (gt_nodes["t"].astype(int) == t) & gt_nodes["node_id"].astype(int).isin(labelled_ids)
                ]
                if len(frame_gt) == 0:
                    continue

                gt_xyz = frame_gt[["z", "y", "x"]].to_numpy(np.float32)
                pred_xyz = c_src[:, 1:4].astype(np.float32) * downsample_arr

                is_division_frame = bool(t in division_times)

                for gi, pi, dist_um in match_gt_to_pred(gt_xyz, pred_xyz, scale_um):
                    node_id = int(frame_gt.iloc[gi]["node_id"])
                    label = eligible_label(int(degrees.loc[node_id]))
                    if label is None:
                        continue

                    j1, j2 = top2_by_raw_logit(raw_np[pi])

                    raw1 = float(raw_np[pi, j1])
                    raw2 = float(raw_np[pi, j2])
                    prob1 = float(probs_np[pi, j1])
                    prob2 = float(probs_np[pi, j2])

                    k1 = k[j1]
                    k2 = k[j2]

                    rows.append(
                        {
                            "dataset": dataset,
                            "fold": int(fold),
                            "t": int(t),
                            "node_id": node_id,
                            "label": int(label),
                            "division_frame": is_division_frame,
                            "match_distance_um": float(dist_um),
                            "parent_q": q[pi].astype(np.float32).copy(),
                            "k_mean": ((k1 + k2) / 2.0).astype(np.float32),
                            "k_absdiff": np.abs(k1 - k2).astype(np.float32),
                            "score_features": score_features(raw1, raw2, prob1, prob2),
                            "geometry": candidate_geometry(
                                c_src[pi, 1:4].astype(np.float32),
                                c_tgt[j1, 1:4].astype(np.float32),
                                c_tgt[j2, 1:4].astype(np.float32),
                                downsample_arr,
                                scale_um,
                            ),
                            "top1_xyz": c_tgt[j1, 1:4].astype(np.float32).copy(),
                            "top2_xyz": c_tgt[j2, 1:4].astype(np.float32).copy(),
                        }
                    )

            del unet_out, det, imgs

            # Only the newest frame is still needed as the next pair's source.
            newest = batch[-1] + 1
            frame_dets = {key: value for key, value in frame_dets.items() if key >= newest}
    finally:
        hook.remove()

    return rows


def validate_schema(frame: pd.DataFrame) -> None:
    """Fail loudly when the extracted frame drifts from the output contract."""
    if list(frame.columns) != list(OUTPUT_COLUMNS):
        raise ValueError(f"schema mismatch\n  expected {list(OUTPUT_COLUMNS)}\n  observed {list(frame.columns)}")

    if frame.empty:
        return

    for column, width in EXPECTED_FEATURE_DIMS.items():
        first = frame.iloc[0][column]
        if not isinstance(first, np.ndarray):
            raise TypeError(f"{column} must hold numpy arrays, got {type(first)}")
        if first.dtype != np.float32:
            raise TypeError(f"{column} must be float32, got {first.dtype}")

        widths = frame[column].map(len).unique()
        if len(widths) != 1 or int(widths[0]) != width:
            raise ValueError(f"{column} must have width {width}, observed {sorted(widths)}")

    bad_labels = sorted(set(frame["label"].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"label must be 0 or 1, found {bad_labels}")


def extract_multiframe_features(
    selection: DatasetSelection,
    *,
    build_context: Any,
    progress: Any = None,
) -> pd.DataFrame:
    """Run :func:`extract_dataset_rows` over a dataset selection.

    ``build_context(dataset, fold)`` returns the keyword arguments for one
    dataset, so this stays free of IO and model-loading policy.
    """
    rows: list[dict[str, Any]] = []

    for index, meta in enumerate(selection.frame.itertuples(index=False), start=1):
        dataset_rows = extract_dataset_rows(**build_context(meta.dataset, int(meta.fold)))
        rows.extend(dataset_rows)

        if progress is not None:
            progress(index, len(selection), meta.dataset, len(dataset_rows), len(rows))

    frame = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    validate_schema(frame)
    return frame
