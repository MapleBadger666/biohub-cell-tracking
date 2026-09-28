#!/usr/bin/env python
"""Stage 11 all-frame joint-feature extraction runner.

Extracts the validated Stage 11.09A joint parent/top-2-candidate features for
**every** valid frame pair of the selected datasets, and writes one lossless
pickle plus a compact summary.

Ground truth is used only to match and label rows after inference; it never
selects which frames are encoded or which candidates are scored. Nothing is
fitted, calibrated, or thresholded here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]

for extra in (PROJECT / "src", PROJECT / "scripts", PROJECT / "external/official/scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from replay_stage11_association import (  # noqa: E402 - path bootstrap must precede import
    DatasetReplay,
    _detect_cells_pooled,
    _load_frame,
    extract_pos_features,
    load_model,
)
from stage11_multiframe_utils import (  # noqa: E402 - path bootstrap must precede import
    BATCH_WINDOWS,
    SCALE_UM,
    extract_multiframe_features,
    select_datasets,
    verify_checkpoint,
)

from biohub_cell_tracking.frozen_v9 import load_graph  # noqa: E402

DEFAULT_DATA_ROOT = "data/raw/competition/biohub-cell-tracking-during-development"
DEFAULT_MANIFEST = "configs/cv_manifest.csv"

# 6bba is routed to Baseline256 seed1640 by the current-best hybrid.
DEFAULT_CHECKPOINT = "models/official_baseline/stage10_pilot_baseline_256/split_0/edge_predictor_best.pth"
DEFAULT_CHECKPOINT_SHA = "8bddad68aaabf240d36eab9cda158f12d4969fa1cd6b21277570043ef83ab689"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_folds(text: str) -> list[int]:
    folds = [int(part) for part in text.replace(" ", "").split(",") if part]
    if not folds:
        raise argparse.ArgumentTypeError("--folds must list at least one fold")
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT / DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", type=str, default=DEFAULT_CHECKPOINT_SHA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=parse_folds, default=[0, 1, 2, 3, 4])
    parser.add_argument("--prefix", type=str, default="6bba")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_WINDOWS)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap the dataset count")
    args = parser.parse_args()

    # DATA_ROOT/train explicitly; no recursive train/test resolution anywhere.
    train_root = Path(args.data_root) / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"train root not found: {train_root}")

    if not args.manifest.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    sha = verify_checkpoint(args.checkpoint, args.checkpoint_sha256)

    manifest = pd.read_csv(args.manifest)
    selection = select_datasets(manifest, prefix=args.prefix, folds=args.folds, data_root=train_root)

    if args.limit is not None:
        selection = type(selection)(
            frame=selection.frame.head(args.limit).reset_index(drop=True),
            prefix=selection.prefix,
            folds=selection.folds,
        )

    device = resolve_device(args.device)

    print("Stage 11 all-frame joint-feature extraction")
    print(f"  train root : {train_root}")
    print(f"  manifest   : {args.manifest}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  sha256     : {sha}  VERIFIED")
    print(f"  prefix     : {selection.prefix}")
    print(f"  folds      : {list(selection.folds)}")
    print(f"  datasets   : {len(selection)}")
    print(f"  device     : {device}")
    print(f"  batch size : {args.batch_size} windows")
    print(flush=True)

    model, window_size, downsample = load_model(args.checkpoint, device)
    model.eval()
    downsample_arr = np.asarray(downsample, dtype=np.float32)

    def build_context(dataset: str, fold: int) -> dict:
        replay = DatasetReplay(dataset, train_root, model, window_size, downsample, device)

        observed_scale = np.asarray(replay.scale, dtype=np.float32)
        if not np.allclose(observed_scale, SCALE_UM, rtol=0, atol=1e-6):
            raise ValueError(
                f"{dataset}: voxel scale {tuple(observed_scale)} differs from the Stage 11 "
                f"constant {tuple(SCALE_UM)}; extraction would not be comparable"
            )

        graph = load_graph(train_root / f"{dataset}.geff")

        return {
            "dataset": dataset,
            "fold": fold,
            "replay": replay,
            "model": model,
            "device": device,
            "downsample_arr": downsample_arr,
            "gt_nodes": graph.node_attrs(unpack=True).to_pandas(),
            "gt_edges": graph.edge_attrs(attr_keys=["source_id", "target_id"], unpack=True).to_pandas(),
            "extract_pos_features": extract_pos_features,
            "detect_cells": _detect_cells_pooled,
            "load_frame": _load_frame,
            "batch_size": args.batch_size,
        }

    started = time.time()

    def progress(index: int, total: int, dataset: str, dataset_rows: int, cumulative: int) -> None:
        print(
            f"  [{index:3d}/{total}] {dataset:<16} rows={dataset_rows:<6} "
            f"total={cumulative:<8} {(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    frame = extract_multiframe_features(selection, build_context=build_context, progress=progress)

    elapsed = time.time() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(args.output)

    print()
    print("=" * 60)
    print("Extraction summary")
    print("=" * 60)
    print(f"  datasets  : {len(selection)}")
    print(f"  rows      : {len(frame)}")
    print(f"  positives : {int(frame['label'].sum()) if len(frame) else 0}")
    print(f"  elapsed_s : {elapsed:.1f}")

    if len(frame):
        by_fold = frame.groupby("fold").agg(rows=("label", "size"), positives=("label", "sum"))
        print()
        print("  rows / positives by fold")
        for fold_value, row in by_fold.iterrows():
            print(f"    fold {int(fold_value)} : rows={int(row['rows']):<8} positives={int(row['positives'])}")

    print()
    print(f"  written   : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
