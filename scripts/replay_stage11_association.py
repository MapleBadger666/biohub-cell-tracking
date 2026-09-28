#!/usr/bin/env python
"""Stage 11.04 — targeted association logit replay on judgeable division cases.

Replays the frozen predictor's association head on only the frame pairs that
carry a Stage 11.01 division TP, FP, or all-nodes-present FN, and recovers the
raw logits / column-softmax probabilities that the predictor discards before
GEFF serialisation.

Fidelity first: replayed detections must match the saved GEFF exactly, and the
current rule must reproduce the saved edge set for the pair. Any mismatch is
reported and the affected pair is not interpreted.

No training. No full Fold0 inference. No GEFF is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import zarr  # noqa: E402 - path bootstrap must precede import
from predict_unet_transformer_mps import (  # noqa: E402 - frozen predictor, imported never edited
    PredictConfig,
    _detect_cells_pooled,
    _load_frame,
    load_model,
    pool_kernel_from_um,
)
from tracking_cellmot.io import open_dataset  # noqa: E402 - path bootstrap must precede import
from train_unet_transformer_mps import extract_pos_features  # noqa: E402 - frozen trainer helper

from biohub_cell_tracking.association_replay import (  # noqa: E402 - path bootstrap must precede import
    CANDIDATE_RULES,
    MISSING_EDGE_CLASSES,
    FramePair,
    ReplayTotals,
    build_node_mapping,
    classify_missing_edge,
    column_stats,
    dedupe_frame_pairs,
    edge_visible,
    fork_evidence,
    row_stats,
    rule_candidate_edges,
    rule_fork_counts,
)
from biohub_cell_tracking.division_audit import fork_geometry  # noqa: E402
from biohub_cell_tracking.division_geometry import (  # noqa: E402
    GeometryGate,
    evaluate_geometry_gate,
)
from biohub_cell_tracking.frozen_v9 import load_graph  # noqa: E402

GT_DIR = "data/raw/competition/biohub-cell-tracking-during-development/train"

# Current-best hybrid routing. Checkpoint SHAs are recorded in
# notebooks/10_sparse_label_training.ipynb (cells 10.07 and 10.17).
ROUTING = {
    "44b6": {
        "checkpoint": "models/official_baseline/stage10_pilot_sparse9995_256/split_0/edge_predictor_best.pth",
        "sha256": "ab7a75268f43e2807167913f3267a10055b1f21941af916b34962f544ccb5699",
        "pred_995": "predictions/heqiuyan/stage10_infer_sparse9995_256_det995/split_0",
        "name": "Sparse9995 seed1640",
    },
    "6bba": {
        "checkpoint": "models/official_baseline/stage10_pilot_baseline_256/split_0/edge_predictor_best.pth",
        "sha256": "8bddad68aaabf240d36eab9cda158f12d4969fa1cd6b21277570043ef83ab689",
        "pred_995": "predictions/heqiuyan/stage10_infer_baseline_256_det995/split_0",
        "name": "Baseline256 seed1640",
    },
}

DET_THRESHOLD = 0.995


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(lines: list[str], text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


class DatasetReplay:
    """Frozen-predictor replay for one dataset, reusing the locked implementation."""

    def __init__(self, name: str, gt_dir: Path, model, window_size: int, downsample, device):
        self.name = name
        self.model = model
        self.window = window_size
        self.downsample = tuple(downsample)
        self.device = device

        self.cfg = PredictConfig(det_threshold=DET_THRESHOLD)

        ds = open_dataset(gt_dir / name, normalize=False, load_image=False, downsample=downsample)
        self.zarr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
        self.q_low = float(ds.quantiles["0.001"])
        self.q_high = float(ds.quantiles["0.999"])
        self.T = ds.image_shape[0]
        self.image_shape = (self.T, *ds.image_shape[1:])
        self.target_shape = list(self.image_shape[1:])
        self.scale = tuple(float(s) for s in ds.scale)
        self.pool_k = pool_kernel_from_um(
            self.cfg.pool_kernel_um, tuple(s * d for s, d in zip(ds.scale, downsample, strict=True))
        )
        self.ds_arr_t = torch.from_numpy(np.array(downsample, dtype=np.float32)).to(device)

    def _encode(self, frame_indices: list[int]):
        """UNet encode plus the frozen 4-view detection TTA, verbatim in effect."""
        imgs = torch.stack([_load_frame(self.zarr, t, self.target_shape, self.downsample) for t in frame_indices])
        imgs = ((imgs - self.q_low) / (self.q_high - self.q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(self.device)

        unet_out, det_logits = self.model.encode(imgs)

        if self.cfg.det_tta:
            for dims in [(-1,), (-2,), (-2, -1)]:
                _, det_flip = self.model.encode(imgs.flip(dims))
                for f in range(len(frame_indices)):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
            for f in range(len(frame_indices)):
                det_logits[f] = det_logits[f] / 4

        return unet_out, det_logits

    def replay_pair(self, t: int):
        """Detections for frames t and t+1 plus the association matrices.

        Frame ``t`` is detected in the first window containing it — window
        ``[t-1, t]`` for ``t >= 1`` — exactly as ``predict_video``'s
        ``seen_frames`` dedup arranges. The association for the pair uses the
        ``[t, t+1]`` window.
        """
        with torch.no_grad():
            if t >= 1:
                _, det_prev = self._encode([t - 1, t])
                c_src = _detect_cells_pooled(det_prev[1][0], t, self.cfg.det_threshold, self.pool_k)
                unet_out, det_pair = self._encode([t, t + 1])
            else:
                unet_out, det_pair = self._encode([0, 1])
                c_src = _detect_cells_pooled(det_pair[0][0], 0, self.cfg.det_threshold, self.pool_k)

            c_tgt = _detect_cells_pooled(det_pair[1][0], t + 1, self.cfg.det_threshold, self.pool_k)

            n_src, n_tgt = len(c_src), len(c_tgt)
            if n_src == 0 or n_tgt == 0:
                return c_src, c_tgt, None, None

            p_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(self.device)
            p_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(self.device)

            window_shape = (self.window, *self.image_shape[1:])
            src_rel = c_src.copy()
            src_rel[:, 0] = 0
            tgt_rel = c_tgt.copy()
            tgt_rel[:, 0] = 1

            pos_src = torch.from_numpy(extract_pos_features(src_rel, window_shape)).unsqueeze(0).to(self.device)
            pos_tgt = torch.from_numpy(extract_pos_features(tgt_rel, window_shape)).unsqueeze(0).to(self.device)
            mask_src = torch.ones(1, n_src, dtype=torch.bool, device=self.device)
            mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=self.device)

            feat_src = self.model._index_features(unet_out[:, 0], p_src, mask_src)
            feat_tgt = self.model._index_features(unet_out[:, 1], p_tgt, mask_tgt)

            raw = self.model.predict_edges(
                feat_src,
                feat_tgt,
                p_src * self.ds_arr_t,
                p_tgt * self.ds_arr_t,
                pos_src,
                pos_tgt,
                mask_src,
                mask_tgt,
            )[0]

            probs = torch.softmax(raw, dim=0).cpu().numpy()
            return c_src, c_tgt, raw.cpu().numpy(), probs


def frozen_selection(probs: np.ndarray, cfg: PredictConfig) -> set[tuple[int, int]]:
    """The frozen candidate + greedy-degree sweep, for one frame pair."""
    candidates = sorted(
        [
            (probs[i, j], i, j)
            for i in range(probs.shape[0])
            for j in range(probs.shape[1])
            if probs[i, j] > cfg.threshold
        ],
        reverse=True,
    )

    children: dict[int, int] = {}
    parents: dict[int, int] = {}
    selected: set[tuple[int, int]] = set()

    for _, i, j in candidates:
        if children.get(i, 0) >= cfg.max_children_per_node:
            continue
        if parents.get(j, 0) >= cfg.max_parents_per_node:
            continue
        selected.add((i, j))
        children[i] = children.get(i, 0) + 1
        parents[j] = parents.get(j, 0) + 1

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    gt_dir = project / GT_DIR
    stage1101 = project / "reports/stage11_division_audit"
    out_dir = args.out_dir or (project / "reports/stage11_association_replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    log(lines, "Stage 11.04 — targeted association logit replay")

    # -- checkpoint provenance ---------------------------------------
    for prefix, cfg in ROUTING.items():
        path = project / cfg["checkpoint"]
        if not path.exists():
            log(lines, f"FAIL: missing checkpoint {path}")
            return 2
        observed = sha256_file(path)
        if observed != cfg["sha256"]:
            log(lines, f"FAIL: {prefix} checkpoint SHA mismatch\n  expected {cfg['sha256']}\n  got      {observed}")
            return 2
        log(lines, f"  {prefix} -> {cfg['name']}")
        log(lines, f"      {cfg['checkpoint']}")
        log(lines, f"      sha256 {observed}  VERIFIED")

    # -- replay population -------------------------------------------
    gt_divisions = pd.read_csv(stage1101 / "gt_divisions.csv")
    pred_divisions = pd.read_csv(stage1101 / "pred_divisions.csv")

    fn_cases = gt_divisions[gt_divisions["classification"] == "NODES_PRESENT_EDGES_MISSING"].reset_index(drop=True)
    tp_forks = pred_divisions[pred_divisions["counted_as_tp"]].reset_index(drop=True)
    fp_forks = pred_divisions[pred_divisions["counted_as_fp"]].reset_index(drop=True)

    log(lines)
    log(lines, "Replay population")
    log(lines, f"  A. FN all-nodes-present cases : {len(fn_cases)}")
    log(lines, f"  B. judgeable division FPs     : {len(fp_forks)}")
    log(lines, f"  C. division TPs               : {len(tp_forks)}")

    requests: list[FramePair] = []
    requests += [FramePair(r.dataset, int(r.t)) for r in fn_cases.itertuples(index=False)]
    requests += [FramePair(r.dataset, int(r.t)) for r in tp_forks.itertuples(index=False)]
    requests += [FramePair(r.dataset, int(r.t)) for r in fp_forks.itertuples(index=False)]

    pairs = dedupe_frame_pairs(requests)
    log(lines, f"  unique (dataset, t, t+1) frame pairs to replay : {len(pairs)} (from {len(requests)} requests)")

    prefix_of = {}
    for frame in (fn_cases, tp_forks, fp_forks):
        prefix_of.update(dict(zip(frame["dataset"], frame["prefix"], strict=True)))

    # -- queries per frame pair --------------------------------------
    queries: dict[tuple[str, int], list[dict]] = {}

    def add_query(dataset, t, **payload):
        queries.setdefault((dataset, int(t)), []).append(payload)

    for row in fn_cases.itertuples(index=False):
        for role, daughter in (
            ("daughter_1", row.pred_daughter_1),
            ("daughter_2", row.pred_daughter_2),
        ):
            add_query(
                row.dataset,
                row.t,
                case_type="FN",
                case_id=f"{row.dataset}:{row.gt_parent}",
                role=role,
                source_node=int(row.pred_parent),
                target_node=int(daughter),
            )

    for label, frame in (("TP", tp_forks), ("FP", fp_forks)):
        for row in frame.itertuples(index=False):
            for k, daughter in enumerate(str(row.daughters).split(","), start=1):
                add_query(
                    row.dataset,
                    row.t,
                    case_type=label,
                    case_id=f"{row.dataset}:{row.parent_node}",
                    role=f"daughter_{k}",
                    source_node=int(row.parent_node),
                    target_node=int(daughter),
                )

    # -- model loading ------------------------------------------------
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else torch.device("cpu")
        )
    )
    log(lines)
    log(lines, f"Device: {device}")

    models = {}
    for prefix, cfg in ROUTING.items():
        model, window_size, downsample = load_model(project / cfg["checkpoint"], device)
        models[prefix] = (model, window_size, downsample)

    # -- replay -------------------------------------------------------
    edge_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    totals = ReplayTotals()
    replays: dict[str, DatasetReplay] = {}
    predict_cfg = PredictConfig(det_threshold=DET_THRESHOLD)

    log(lines)
    log(lines, "Replaying frame pairs")
    started = time.time()

    for index, pair in enumerate(pairs, start=1):
        prefix = prefix_of[pair.dataset]
        model, window_size, downsample = models[prefix]

        if pair.dataset not in replays:
            replays[pair.dataset] = DatasetReplay(pair.dataset, gt_dir, model, window_size, downsample, device)
        replay = replays[pair.dataset]

        geff = load_graph(project / ROUTING[prefix]["pred_995"] / f"{pair.dataset}.geff")
        nodes = geff.node_attrs(unpack=True).to_pandas()
        edges = geff.edge_attrs(attr_keys=["edge_prob"], unpack=True).to_pandas()

        c_src, c_tgt, raw, probs = replay.replay_pair(pair.t)

        src_frame = nodes[nodes["t"] == pair.t].sort_values("node_id")
        tgt_frame = nodes[nodes["t"] == pair.t_next].sort_values("node_id")

        map_src = build_node_mapping(pair.dataset, pair.t, c_src, src_frame, downsample)
        map_tgt = build_node_mapping(pair.dataset, pair.t_next, c_tgt, tgt_frame, downsample)

        saved_edges = {
            (int(r.source_id), int(r.target_id))
            for r in edges.itertuples(index=False)
            if int(r.source_id) in set(map_src.node_ids) and int(r.target_id) in set(map_tgt.node_ids)
        }

        edges_match = False
        replay_edge_count = 0
        if probs is not None and map_src.is_valid and map_tgt.is_valid:
            selection = frozen_selection(probs, predict_cfg)
            replayed = {(map_src.node_ids[i], map_tgt.node_ids[j]) for i, j in selection}
            replay_edge_count = len(replayed)
            edges_match = replayed == saved_edges

        fidelity_rows.append(
            {
                "dataset": pair.dataset,
                "prefix": prefix,
                "t": pair.t,
                "replay_src_nodes": map_src.replay_count,
                "saved_src_nodes": map_src.saved_count,
                "src_coords_match": map_src.coordinates_match,
                "replay_tgt_nodes": map_tgt.replay_count,
                "saved_tgt_nodes": map_tgt.saved_count,
                "tgt_coords_match": map_tgt.coordinates_match,
                "saved_edges": len(saved_edges),
                "replay_edges": replay_edge_count,
                "current_rule_reproduces_geff": edges_match,
                "faithful": bool(map_src.is_valid and map_tgt.is_valid and edges_match),
            }
        )

        if not (map_src.is_valid and map_tgt.is_valid and edges_match):
            log(lines, f"  MISMATCH {pair.dataset} t={pair.t} — not interpreted")
            continue

        for rule in CANDIDATE_RULES:
            totals.add(rule, rule_fork_counts(rule_candidate_edges(rule, probs, raw)))

        current_children = {}
        for i, _ in frozen_selection(probs, predict_cfg):
            current_children[i] = current_children.get(i, 0) + 1

        for query in queries.get(pair.key, []):
            i = map_src.index_of(query["source_node"])
            j = map_tgt.index_of(query["target_node"])
            if i is None or j is None:
                log(
                    lines,
                    f"  UNMAPPED {pair.dataset} t={pair.t} "
                    f"{query['source_node']}->{query['target_node']} — not interpreted",
                )
                continue

            col = column_stats(probs, raw, i, j)
            row = row_stats(raw, i, j)

            record = {
                "dataset": pair.dataset,
                "prefix": prefix,
                "case_type": query["case_type"],
                "case_id": query["case_id"],
                "role": query["role"],
                "t": pair.t,
                "source_node": query["source_node"],
                "target_node": query["target_node"],
                "edge_in_geff": (query["source_node"], query["target_node"]) in saved_edges,
                "raw_logit": col.raw_logit,
                "column_softmax_prob": col.column_softmax_prob,
                "column_rank": col.column_rank,
                "column_size": col.column_size,
                "column_top1_source": map_src.node_ids[col.column_top1_source],
                "column_top1_prob": col.column_top1_prob,
                "column_top2_source": (
                    map_src.node_ids[col.column_top2_source] if col.column_top2_source is not None else None
                ),
                "column_top2_prob": col.column_top2_prob,
                "column_margin_to_top1": col.column_margin_to_top1,
                "row_rank_by_raw_logit": row.row_rank_by_raw_logit,
                "row_size": row.row_size,
                "parent_top1_target": map_tgt.node_ids[row.parent_top1_target],
                "parent_top2_target": (
                    map_tgt.node_ids[row.parent_top2_target] if row.parent_top2_target is not None else None
                ),
                "parent_top3_target": (
                    map_tgt.node_ids[row.parent_top3_target] if row.parent_top3_target is not None else None
                ),
                "parent_current_children": int(current_children.get(i, 0)),
            }

            for rule in CANDIDATE_RULES:
                record[f"visible_{rule.name}"] = edge_visible(
                    rule,
                    column_rank=col.column_rank,
                    column_prob=col.column_softmax_prob,
                    row_rank=row.row_rank_by_raw_logit,
                )

            edge_rows.append(record)

        if index == 1 or index % 10 == 0 or index == len(pairs):
            log(lines, f"  replayed {index:02d}/{len(pairs)}  ({time.time() - started:.0f}s)")

    fidelity = pd.DataFrame(fidelity_rows)
    edge_table = pd.DataFrame(edge_rows)

    # -- fidelity gate -------------------------------------------------
    log(lines)
    log(lines, "Replay fidelity")
    log(lines, f"  frame pairs replayed        : {len(fidelity)}")
    log(
        lines,
        f"  detection coords exact      : {int((fidelity['src_coords_match'] & fidelity['tgt_coords_match']).sum())}",
    )
    log(lines, f"  current rule reproduces GEFF: {int(fidelity['current_rule_reproduces_geff'].sum())}")
    log(lines, f"  fully faithful pairs        : {int(fidelity['faithful'].sum())}")

    if not bool(fidelity["faithful"].all()):
        log(lines, "  STOP: replay is not faithful on every pair; ranks below are not interpretable.")
        fidelity.to_csv(out_dir / "replay_fidelity.csv", index=False)
        (out_dir / "replay_run.log").write_text("\n".join(lines) + "\n")
        return 1

    # -- missing-edge classification ------------------------------------
    fn_edges = edge_table[edge_table["case_type"] == "FN"].copy()
    missing = fn_edges[~fn_edges["edge_in_geff"]].copy()

    missing["classification"] = [
        classify_missing_edge(
            column_rank=int(r.column_rank),
            column_softmax_prob=float(r.column_softmax_prob),
            column_margin_to_top1=float(r.column_margin_to_top1),
            parent_at_child_cap=int(r.parent_current_children) >= 2,
        )
        for r in missing.itertuples(index=False)
    ]

    log(lines)
    log(lines, "Missing required-edge evidence")
    log(lines, f"  required edges examined : {len(fn_edges)}")
    log(lines, f"  present in GEFF         : {int(fn_edges['edge_in_geff'].sum())}")
    log(lines, f"  missing                 : {len(missing)}")

    log(lines)
    log(lines, "  column-rank distribution of the missing edges")
    for rank_label, count in missing["column_rank"].value_counts().sort_index().items():
        log(lines, f"    rank {int(rank_label):>3} : {int(count)}")
    log(
        lines,
        f"    rank==1 {int((missing['column_rank'] == 1).sum())}, "
        f"rank==2 {int((missing['column_rank'] == 2).sum())}, "
        f"rank<=3 {int((missing['column_rank'] <= 3).sum())}, "
        f"rank>3 {int((missing['column_rank'] > 3).sum())}",
    )

    log(lines)
    log(lines, "  column-softmax probability of the missing edges")
    probs_missing = missing["column_softmax_prob"]
    log(
        lines,
        f"    min {probs_missing.min():.4f}  q25 {probs_missing.quantile(0.25):.4f}  "
        f"median {probs_missing.median():.4f}  q75 {probs_missing.quantile(0.75):.4f}  "
        f"max {probs_missing.max():.4f}",
    )
    log(
        lines,
        f"    above 0.40 {int((probs_missing > 0.40).sum())}, "
        f"above 0.30 {int((probs_missing > 0.30).sum())}, "
        f"above 0.20 {int((probs_missing > 0.20).sum())}",
    )

    log(lines)
    log(lines, "  parent-centric (row) rank of the missing edges")
    log(
        lines,
        f"    row rank<=2 {int((missing['row_rank_by_raw_logit'] <= 2).sum())}, "
        f"<=3 {int((missing['row_rank_by_raw_logit'] <= 3).sum())}, "
        f">3 {int((missing['row_rank_by_raw_logit'] > 3).sum())}",
    )

    log(lines)
    log(lines, "  classification counts")
    class_counts = {c: int((missing["classification"] == c).sum()) for c in MISSING_EDGE_CLASSES}
    for name, count in class_counts.items():
        log(lines, f"    {name:<34} {count:>3}")

    log(lines)
    log(lines, "  classification by prefix")
    by_prefix = missing.groupby(["prefix", "classification"]).size().unstack(fill_value=0)
    log(lines, by_prefix.to_string())

    # -- fork-level evidence --------------------------------------------
    fork_rows = []
    for case_id, block in edge_table.groupby("case_id"):
        if len(block) != 2:
            continue
        a, b = block.iloc[0].to_dict(), block.iloc[1].to_dict()
        fork_rows.append(
            {
                "case_id": case_id,
                "dataset": a["dataset"],
                "prefix": a["prefix"],
                "case_type": a["case_type"],
                "t": a["t"],
                "parent": a["source_node"],
                "edges_in_geff": int(a["edge_in_geff"]) + int(b["edge_in_geff"]),
                **fork_evidence(a, b),
            }
        )
    forks = pd.DataFrame(fork_rows)

    log(lines)
    log(lines, "Fork-level neural evidence by case type (descriptive; TP n=2)")
    summary_cols = [
        "min_prob",
        "max_prob",
        "prob_product",
        "prob_gap",
        "min_raw_logit",
        "sum_raw_logit",
        "max_column_rank",
        "max_row_rank",
    ]
    fork_summary = forks.groupby("case_type")[summary_cols].agg(["count", "min", "median", "max"])
    log(lines, fork_summary.to_string())

    log(lines)
    log(lines, "Fork rank structure by case type")
    rank_struct = forks.groupby("case_type")[
        ["both_column_rank_le_2", "both_row_rank_le_2", "both_row_rank_le_3"]
    ].sum()
    rank_struct["n"] = forks.groupby("case_type").size()
    log(lines, rank_struct.to_string())

    # -- candidate rule audit --------------------------------------------
    fn_case_index = fn_edges.groupby("case_id")
    geometry_by_case = {}
    for row in fn_cases.itertuples(index=False):
        replay = replays[row.dataset]
        geff = load_graph(project / ROUTING[row.prefix]["pred_995"] / f"{row.dataset}.geff")
        nodes = geff.node_attrs(unpack=True).to_pandas().set_index("node_id")
        try:
            p = nodes.loc[int(row.pred_parent), ["z", "y", "x"]].to_numpy(dtype=float)
            c1 = nodes.loc[int(row.pred_daughter_1), ["z", "y", "x"]].to_numpy(dtype=float)
            c2 = nodes.loc[int(row.pred_daughter_2), ["z", "y", "x"]].to_numpy(dtype=float)
        except KeyError:
            continue
        geometry_by_case[f"{row.dataset}:{row.gt_parent}"] = fork_geometry(p, c1, c2, replay.scale)

    frozen_gate = GeometryGate.frozen()
    rule_rows = []
    for rule in CANDIDATE_RULES:
        col = f"visible_{rule.name}"

        recovered_edges = int((~fn_edges["edge_in_geff"] & fn_edges[col]).sum())

        complete_cases = []
        for case_id, block in fn_case_index:
            if bool(block[col].all()):
                complete_cases.append(case_id)

        geometry_frame = pd.DataFrame(
            [
                {
                    "case_id": case_id,
                    "daughter_separation_um": geometry_by_case[case_id].daughter_separation_um,
                    "division_angle_deg": geometry_by_case[case_id].division_angle_deg,
                    "max_parent_daughter_um": geometry_by_case[case_id].max_parent_daughter_um,
                    "distance_balance": geometry_by_case[case_id].distance_balance,
                }
                for case_id in complete_cases
                if case_id in geometry_by_case
            ]
        )
        g2_reachable = (
            int(evaluate_geometry_gate(geometry_frame, frozen_gate)["passes_geometry"].sum())
            if len(geometry_frame)
            else 0
        )

        tp_block = edge_table[edge_table["case_type"] == "TP"]
        tp_retained = int(tp_block.groupby("case_id")[col].all().sum())

        fp_block = edge_table[edge_table["case_type"] == "FP"]
        fp_exposed = int(fp_block.groupby("case_id")[col].all().sum())

        exposure = totals.per_rule[rule.name]
        rule_rows.append(
            {
                "rule": rule.name,
                "kind": rule.kind,
                "value": rule.value,
                "division_only": rule.division_only,
                "recovered_fn_edges": recovered_edges,
                "recovered_fn_cases_both_edges": len(complete_cases),
                "recovered_fn_g2_reachable": g2_reachable,
                "tp_forks_retained": tp_retained,
                "fp_forks_exposed": fp_exposed,
                "candidate_edges_replayed_pairs": int(exposure.get("candidate_edges", 0)),
                "candidate_forks_replayed_pairs": int(exposure.get("candidate_forks", 0)),
                "frame_pairs": int(exposure.get("frame_pairs", 0)),
            }
        )

    rules_table = pd.DataFrame(rule_rows)
    current = rules_table[rules_table["rule"] == "A_current_col_gt_0.50"].iloc[0]
    rules_table["fork_inflation_vs_current"] = rules_table["candidate_forks_replayed_pairs"] / max(
        int(current["candidate_forks_replayed_pairs"]), 1
    )

    log(lines)
    log(lines, "Candidate-rule audit (RULE C/D are division-only visibility, not graph edges)")
    log(
        lines,
        rules_table[
            [
                "rule",
                "division_only",
                "recovered_fn_edges",
                "recovered_fn_cases_both_edges",
                "recovered_fn_g2_reachable",
                "tp_forks_retained",
                "fp_forks_exposed",
                "candidate_forks_replayed_pairs",
                "fork_inflation_vs_current",
            ]
        ].to_string(index=False),
    )

    # -- artifacts --------------------------------------------------------
    edge_table.to_csv(out_dir / "replay_edges.csv", index=False)
    forks.to_csv(out_dir / "replay_division_cases.csv", index=False)
    missing.to_csv(out_dir / "missing_edge_classification.csv", index=False)
    rules_table.to_csv(out_dir / "candidate_rule_summary.csv", index=False)
    fidelity.to_csv(out_dir / "replay_fidelity.csv", index=False)

    payload = {
        "checkpoints": {p: {k: v for k, v in c.items() if k != "pred_995"} for p, c in ROUTING.items()},
        "frame_pairs_replayed": len(pairs),
        "replay_faithful": bool(fidelity["faithful"].all()),
        "required_edges": len(fn_edges),
        "missing_edges": len(missing),
        "missing_rank_1": int((missing["column_rank"] == 1).sum()),
        "missing_rank_2": int((missing["column_rank"] == 2).sum()),
        "missing_rank_le_3": int((missing["column_rank"] <= 3).sum()),
        "missing_rank_gt_3": int((missing["column_rank"] > 3).sum()),
        "missing_row_rank_le_2": int((missing["row_rank_by_raw_logit"] <= 2).sum()),
        "missing_row_rank_le_3": int((missing["row_rank_by_raw_logit"] <= 3).sum()),
        "classification_counts": class_counts,
        "classification_by_prefix": by_prefix.to_dict(),
        "missing_prob_quantiles": {
            "min": float(probs_missing.min()),
            "q25": float(probs_missing.quantile(0.25)),
            "median": float(probs_missing.median()),
            "q75": float(probs_missing.quantile(0.75)),
            "max": float(probs_missing.max()),
        },
        "candidate_rules": rules_table.to_dict("records"),
        "fork_rank_structure": rank_struct.to_dict(),
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(payload, indent=2, default=float))
    (out_dir / "replay_run.log").write_text("\n".join(lines) + "\n")

    print(f"\nArtifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
