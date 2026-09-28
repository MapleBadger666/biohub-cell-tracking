"""Stage 11.02 — association candidate-generation source/dataflow audit.

Read-only diagnostics for the *prediction* side of the pipeline: why a required
parent -> daughter association is absent from a saved prediction GEFF.

Stage 11.01 established that 19 of 28 Fold0 division false negatives have all
three biological nodes present as matched predicted nodes, yet Frozen G2 never
receives both parent -> daughter edges. This module reads the same saved GEFFs
and determines, per case, which stage of the frozen predictor dropped the edge.

The decisive structural fact about the frozen predictor
(``scripts/predict_unet_transformer_mps.py``) is that association probabilities
are a **column-wise** softmax::

    probs = torch.softmax(raw, dim=0)      # raw is (n_src, n_tgt)

so ``sum_i probs[i, j] == 1`` for every target ``j``. Combined with the frozen
``threshold = 0.5`` candidate gate, **at most one source can exceed the
threshold in any target column**. Every target therefore receives at most one
candidate edge, and ``max_parents_per_node = 1`` is unreachable. The only
greedy constraint that can still bind is ``max_children_per_node = 2``.

That yields an exhaustive, disk-determinable decision procedure for a missing
edge ``parent -> daughter``:

- the daughter has an incoming edge from another source
  => that source exceeded 0.5 in the column, so the parent did not; the parent
  was **not** the column argmax (:data:`DAUGHTER_CLAIMED_BY_COMPETITOR`);
- the daughter has no incoming edge and the parent's out-degree is below the
  child cap => the child cap cannot have blocked anything, so no source cleared
  0.5 in that column (:data:`DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP`);
- the daughter has no incoming edge and the parent is at the child cap => the
  parent may have won the column and been skipped by the cap, or the column may
  have been below threshold; the GEFF cannot separate these
  (:data:`DAUGHTER_UNCLAIMED_PARENT_AT_CAP`).

Nothing here trains, runs inference, loads a checkpoint, or mutates any graph,
GEFF, or Stage 11.01 artifact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .division_audit import fork_geometry
from .frozen_v9 import FROZEN_V9_CONFIG

__all__ = [
    "ASSOCIATION_DATAFLOW",
    "BLOCKER_ASSOCIATION_ONLY",
    "BLOCKER_G2_GEOMETRY",
    "CASE_STRUCTURES",
    "DAUGHTER_CLAIMED_BY_COMPETITOR",
    "DAUGHTER_UNCLAIMED_PARENT_AT_CAP",
    "DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP",
    "EDGE_PRESENT",
    "EDGE_STATUSES",
    "FROZEN_PREDICT_CONFIG",
    "RECOVERY_BLOCKERS",
    "ROOT_CAUSE_HYPOTHESES",
    "EdgeRecord",
    "RawGraphView",
    "build_case_table",
    "case_structure",
    "classify_missing_edge",
    "daughter_incoming_edges",
    "graph_view",
    "gt_fork_geometry",
    "parent_outgoing_edges",
    "restored_fork_geometry",
    "summarize_cases",
]


# ============================================================
# Frozen predictor constants
#
# Mirrored from scripts/predict_unet_transformer_mps.py, which is
# source-locked. Recorded here so the audit states the policy it
# reasons about; nothing in this module can change the predictor.
# ============================================================

#: Effective ``PredictConfig`` for the Stage10 prediction sets. The CLI exposes
#: no ``--threshold``/``--edge-activation`` flag and the runs passed neither
#: ``--use-ilp`` nor any edge override, so the dataclass defaults are what ran.
FROZEN_PREDICT_CONFIG: dict[str, Any] = {
    "edge_activation": "softmax",
    "softmax_dim": 0,
    "softmax_axis": "source (t) axis — normalises each target column over all parents",
    "threshold": 0.5,
    "max_parents_per_node": 1,
    "max_children_per_node": 2,
    "use_ilp": False,
    "source": "scripts/predict_unet_transformer_mps.py:87-112,474-520",
}


#: The association dataflow, stage by stage, with source anchors.
ASSOCIATION_DATAFLOW: dict[str, Any] = {
    "stages": [
        {
            "stage": "image window",
            "detail": "W consecutive frames, quantile-normalised (0.1%-99.9%), strided downsample (1, 4, 4).",
            "source": "scripts/predict_unet_transformer_mps.py:386-392",
            "tensor": "(1, W, Z, Y, X)",
        },
        {
            "stage": "UNet encode",
            "detail": "TemporalUNet3D produces per-frame feature maps and detection logits.",
            "source": "scripts/predict_unet_transformer_mps.py:394",
            "tensor": "unet_out (1, W, C, Z', Y', X'); det_logits list of W x (1, 1, Z', Y', X')",
        },
        {
            "stage": "detection peaks",
            "detail": (
                "flip-xy TTA averaged over 4 views, then 3D max-pool local maxima with "
                "sigmoid(logit) > det_threshold. This is the ONLY node gate."
            ),
            "source": "scripts/predict_unet_transformer_mps.py:273-314,398-420",
            "tensor": "coords (N_t, 4) per frame",
        },
        {
            "stage": "pooled node features",
            "detail": (
                "Integer index of the UNet feature map at each detection, concatenated with "
                "sinusoidal position features."
            ),
            "source": "scripts/train_unet_transformer_mps.py:477-510,536-550",
            "tensor": "unet_feat_src (1, n_src, C), unet_feat_tgt (1, n_tgt, C)",
        },
        {
            "stage": "transformer association logits",
            "detail": (
                "SimpleNodeTransformer: 4 bidirectional cross-attention blocks then a dense pairwise MLP "
                "over every (source, target) pair. No spatial gate, no kNN, no top-k — relative position "
                "enters only as a 3-vector feature (coords difference / 100)."
            ),
            "source": "external/official/src/tracking_cellmot/models/simple_node_transformer.py:97-210",
            "tensor": "edge_logits (1, n_src, n_tgt) — directional, source-major",
        },
        {
            "stage": "activation",
            "detail": (
                "softmax over dim=0 (the source axis): each TARGET COLUMN is normalised over all parents. "
                "Column sums to 1. Rows are unconstrained, which is what permits divisions."
            ),
            "source": "scripts/predict_unet_transformer_mps.py:474-477",
            "tensor": "probs (n_src, n_tgt), column-stochastic",
        },
        {
            "stage": "candidate gate (threshold)",
            "detail": (
                "keep (i, j) iff probs[i, j] > 0.5. Because each column sums to 1, at most ONE source per "
                "target can pass. Non-selected scores are discarded here and never serialised."
            ),
            "source": "scripts/predict_unet_transformer_mps.py:479-487",
            "tensor": "candidate list sorted by probability, descending",
        },
        {
            "stage": "greedy degree assignment",
            "detail": (
                "descending-probability sweep with max_children_per_node=2 and max_parents_per_node=1. "
                "The parent cap is the only one that can bind: the threshold gate already guarantees at "
                "most one candidate per target column, so parents_count[j] is never 1 when column j is "
                "reached."
            ),
            "source": "scripts/predict_unet_transformer_mps.py:489-520",
            "tensor": "edges list of (src, tgt, prob, dist)",
        },
        {
            "stage": "graph edge creation",
            "detail": (
                "edge_prob and edge_dist written per accepted edge; edge_dist is Euclidean in downsampled voxels."
            ),
            "source": "scripts/predict_unet_transformer_mps.py:133-170,508-518",
            "tensor": "tracksdata InMemoryGraph",
        },
        {
            "stage": "GEFF serialisation",
            "detail": "save_graph writes the graph; ILP is off, so no further edge rewriting occurs.",
            "source": "scripts/predict_unet_transformer_mps.py:585-597",
            "tensor": "*.geff on disk",
        },
    ],
    "diagram": (
        "image window -> UNet encode -> detection peaks (sigmoid > det_threshold)\n"
        "  -> pooled node features -> dense transformer logits (n_src x n_tgt)\n"
        "  -> column-wise softmax (dim=0)\n"
        "  -> [threshold > 0.5]            <-- at most one source per target column\n"
        "  -> [greedy degree cap: children <= 2, parents <= 1]\n"
        "  -> graph edge creation (edge_prob, edge_dist)\n"
        "  -> GEFF"
    ),
    "answers": {
        "1_parent_features": "unet_feat_src (1, n_src, C) from model._index_features on frame t detections.",
        "2_candidate_features": "unet_feat_tgt (1, n_tgt, C) from model._index_features on frame t+1 detections.",
        "3_logits_shape": (
            "(1, n_src, n_tgt); raw = edge_logits[0] is (n_src, n_tgt), entry [i, j] = source i -> target j."
        ),
        "4_activation": "softmax over dim=0 — column-wise over sources. Not sigmoid, not row-wise.",
        "5_symmetry": (
            "Directional. The pairwise MLP takes ordered (source, target) features plus a signed "
            "relative-position vector."
        ),
        "6_spatial_gating": "None. Every source-target pair in the frame pair is scored densely.",
        "7_top_k": "None anywhere in the predictor.",
        "8_threshold": "probs > 0.5, frozen dataclass default; no CLI override exists.",
        "9_mutual_nn": "None.",
        "10_assignment": "None. Greedy descending-probability sweep, not Hungarian; ILP path exists but was off.",
        "11_one_to_one": (
            "Enforced implicitly on the target side by column-softmax + 0.5, and explicitly by "
            "max_parents_per_node=1 (redundant given the former)."
        ),
        "12_max_out_degree": "max_children_per_node = 2, applied in the greedy sweep.",
        "13_division_mechanism": (
            "Column-wise softmax leaves rows unconstrained, so one parent may exceed 0.5 in two different "
            "target columns; the child cap of 2 then admits both."
        ),
        "14_second_best_survives": (
            "No. A second-best score for a target column is dropped at the threshold gate and never reaches "
            "graph construction."
        ),
        "15_edge_attrs_written": "scripts/predict_unet_transformer_mps.py:508-518 (collect) and 133-170 (build_graph).",
        "16_scores_discarded": (
            "Yes, permanently. Only accepted edges carry a probability into the GEFF; the full "
            "(n_src, n_tgt) matrix is freed per frame pair and never saved."
        ),
    },
}


#: Candidate root causes this audit distinguishes.
ROOT_CAUSE_HYPOTHESES: tuple[str, ...] = (
    "MODEL_SCORE_LIMITED",
    "CANDIDATE_TRUNCATION",
    "MATCHING_SUPPRESSION",
    "EDGE_THRESHOLD_SUPPRESSION",
    "DEGREE_SUPPRESSION",
    "OTHER",
)


# ============================================================
# Edge status vocabulary
# ============================================================

EDGE_PRESENT = "EDGE_PRESENT"
DAUGHTER_CLAIMED_BY_COMPETITOR = "DAUGHTER_CLAIMED_BY_COMPETITOR"
DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP = "DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP"
DAUGHTER_UNCLAIMED_PARENT_AT_CAP = "DAUGHTER_UNCLAIMED_PARENT_AT_CAP"

EDGE_STATUSES: tuple[str, ...] = (
    EDGE_PRESENT,
    DAUGHTER_CLAIMED_BY_COMPETITOR,
    DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP,
    DAUGHTER_UNCLAIMED_PARENT_AT_CAP,
)

#: What each edge status implies about the frozen predictor, and whether the
#: saved GEFF is sufficient to settle it.
EDGE_STATUS_MEANING: dict[str, dict[str, Any]] = {
    EDGE_PRESENT: {
        "root_cause": None,
        "determinable_from_geff": True,
        "implication": "The required association survived to the GEFF.",
    },
    DAUGHTER_CLAIMED_BY_COMPETITOR: {
        "root_cause": "MODEL_SCORE_LIMITED",
        "determinable_from_geff": True,
        "implication": (
            "Another source exceeded 0.5 in this target column, so the true parent was not the column "
            "argmax. Column rank of the true parent is >= 2."
        ),
    },
    DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP: {
        "root_cause": "EDGE_THRESHOLD_SUPPRESSION",
        "determinable_from_geff": True,
        "implication": (
            "No source cleared 0.5 in this target column and the child cap could not have blocked "
            "anything. The column mass was split. Whether the true parent was rank 1 needs raw scores."
        ),
    },
    DAUGHTER_UNCLAIMED_PARENT_AT_CAP: {
        "root_cause": None,
        "determinable_from_geff": False,
        "implication": (
            "The daughter has no incoming edge and the parent already emitted 2 children, so threshold "
            "suppression and degree suppression are indistinguishable on disk."
        ),
    },
}


CASE_ONE_EDGE_PRESENT = "ONE_EDGE_PRESENT"
CASE_ZERO_REQUIRED_EDGES = "ZERO_REQUIRED_EDGES"
CASE_BOTH_EDGES_PRESENT = "BOTH_EDGES_PRESENT"

#: What still blocks a division TP for a case, assuming the missing association
#: were restored. G2 adjudicates *predicted* coordinates, so the predicted fork
#: geometry is what decides; the GT fork geometry says whether the frozen
#: thresholds are appropriate for real divisions at all.
BLOCKER_ASSOCIATION_ONLY = "ASSOCIATION_ONLY"
BLOCKER_G2_GEOMETRY = "G2_GEOMETRY"

RECOVERY_BLOCKERS: tuple[str, ...] = (BLOCKER_ASSOCIATION_ONLY, BLOCKER_G2_GEOMETRY)

CASE_STRUCTURES: tuple[str, ...] = (
    CASE_BOTH_EDGES_PRESENT,
    CASE_ONE_EDGE_PRESENT,
    CASE_ZERO_REQUIRED_EDGES,
)


# ============================================================
# Read-only graph view
# ============================================================


@dataclass(frozen=True)
class EdgeRecord:
    """One directed predicted association as stored in the GEFF."""

    source: int
    target: int
    edge_prob: float
    edge_dist: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_prob": self.edge_prob,
            "edge_dist": self.edge_dist,
        }


@dataclass(frozen=True)
class RawGraphView:
    """Immutable snapshot of one prediction GEFF.

    Built by copying node and edge tables out of the graph. The source graph is
    read through its public accessors only and is never modified.
    """

    dataset: str
    nodes: pd.DataFrame
    edges: tuple[EdgeRecord, ...]
    out_edges: dict[int, tuple[EdgeRecord, ...]]
    in_edges: dict[int, tuple[EdgeRecord, ...]]
    scale: tuple[float, float, float]

    def has_node(self, node: int) -> bool:
        return int(node) in self.nodes.index

    def time_of(self, node: int) -> int | None:
        if not self.has_node(node):
            return None
        return int(self.nodes.loc[int(node), "t"])

    def raw_position(self, node: int) -> np.ndarray | None:
        """Unscaled (z, y, x) voxel position of *node*, or None when absent."""
        if not self.has_node(node):
            return None
        row = self.nodes.loc[int(node)]
        return np.asarray([row["z"], row["y"], row["x"]], dtype=float)

    def position_um(self, node: int) -> np.ndarray | None:
        """Physical (z, y, x) position of *node*, or None when absent."""
        if not self.has_node(node):
            return None
        row = self.nodes.loc[int(node)]
        return np.asarray([row["z"], row["y"], row["x"]], dtype=float) * np.asarray(self.scale, dtype=float)

    def distance_um(self, a: int, b: int) -> float:
        pa = self.position_um(a)
        pb = self.position_um(b)
        if pa is None or pb is None:
            return float("nan")
        return float(np.linalg.norm(pa - pb))

    def out_degree(self, node: int) -> int:
        return len(self.out_edges.get(int(node), ()))

    def in_degree(self, node: int) -> int:
        return len(self.in_edges.get(int(node), ()))

    def edge_between(self, source: int, target: int) -> EdgeRecord | None:
        for edge in self.out_edges.get(int(source), ()):
            if edge.target == int(target):
                return edge
        return None


def graph_view(graph, dataset: str, scale) -> RawGraphView:
    """Snapshot *graph* into a :class:`RawGraphView` without mutating it."""
    nodes = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")

    # GT graphs carry no association attributes; request only what exists so the
    # same view type can describe a prediction GEFF and a GT GEFF.
    available = set(graph.edge_attr_keys())
    wanted = [key for key in ("edge_prob", "edge_dist") if key in available]

    edge_frame = graph.edge_attrs(attr_keys=wanted, unpack=True).to_pandas()

    edges = tuple(
        EdgeRecord(
            source=int(row.source_id),
            target=int(row.target_id),
            edge_prob=float(getattr(row, "edge_prob", float("nan"))),
            edge_dist=float(getattr(row, "edge_dist", float("nan"))),
        )
        for row in edge_frame.itertuples(index=False)
    )

    out_map: dict[int, list[EdgeRecord]] = defaultdict(list)
    in_map: dict[int, list[EdgeRecord]] = defaultdict(list)

    for edge in edges:
        out_map[edge.source].append(edge)
        in_map[edge.target].append(edge)

    def freeze(mapping: dict[int, list[EdgeRecord]]) -> dict[int, tuple[EdgeRecord, ...]]:
        return {
            node: tuple(sorted(items, key=lambda e: (-e.edge_prob, e.target, e.source)))
            for node, items in mapping.items()
        }

    return RawGraphView(
        dataset=dataset,
        nodes=nodes,
        edges=edges,
        out_edges=freeze(out_map),
        in_edges=freeze(in_map),
        scale=tuple(float(s) for s in scale),
    )


def parent_outgoing_edges(view: RawGraphView, parent: int) -> tuple[EdgeRecord, ...]:
    """Outgoing associations of *parent*, highest probability first."""
    return view.out_edges.get(int(parent), ())


def daughter_incoming_edges(view: RawGraphView, daughter: int) -> tuple[EdgeRecord, ...]:
    """Incoming associations of *daughter*, highest probability first."""
    return view.in_edges.get(int(daughter), ())


def competing_outgoing_edges(view: RawGraphView, parent: int, required_targets: set[int]) -> tuple[EdgeRecord, ...]:
    """Outgoing edges of *parent* that do not lead to a required daughter."""
    return tuple(e for e in parent_outgoing_edges(view, parent) if e.target not in required_targets)


def competing_incoming_edges(view: RawGraphView, daughter: int, parent: int) -> tuple[EdgeRecord, ...]:
    """Incoming edges of *daughter* that do not come from *parent*."""
    return tuple(e for e in daughter_incoming_edges(view, daughter) if e.source != int(parent))


# ============================================================
# Classification
# ============================================================


def classify_missing_edge(
    view: RawGraphView,
    parent: int,
    daughter: int,
    *,
    max_children: int = FROZEN_PREDICT_CONFIG["max_children_per_node"],
) -> str:
    """Status of the required ``parent -> daughter`` association.

    Deterministic given the GEFF: the column-softmax + threshold structure of
    the frozen predictor makes the daughter's in-degree and the parent's
    out-degree jointly sufficient to name the stage, except when the parent sits
    exactly at the child cap.
    """
    if view.edge_between(parent, daughter) is not None:
        return EDGE_PRESENT

    if competing_incoming_edges(view, daughter, parent):
        return DAUGHTER_CLAIMED_BY_COMPETITOR

    if view.out_degree(parent) >= max_children:
        return DAUGHTER_UNCLAIMED_PARENT_AT_CAP

    return DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP


def restored_fork_geometry(
    view: RawGraphView,
    parent: int,
    daughter1: int,
    daughter2: int,
    config=FROZEN_V9_CONFIG,
) -> dict[str, Any]:
    """Frozen G2 geometry of the fork that restoring the missing edge would create.

    Every G2 geometry criterion (daughter separation, angle, max parent-daughter
    distance, distance balance) is symmetric in the two daughters, so the answer
    does not depend on which edge is the missing one or on the probability
    ordering G2 would apply. Only the ``p2`` confidence test is unavailable here,
    because the missing edge carries no saved probability.

    Reuses :func:`biohub_cell_tracking.division_audit.fork_geometry`, which
    mirrors the frozen ``apply_frozen_g2`` arithmetic.
    """
    p = view.raw_position(parent)
    c1 = view.raw_position(daughter1)
    c2 = view.raw_position(daughter2)

    if p is None or c1 is None or c2 is None:
        return {
            "restored_daughter_separation_um": float("nan"),
            "restored_division_angle_deg": float("nan"),
            "restored_max_parent_daughter_um": float("nan"),
            "restored_distance_balance": float("nan"),
            "restored_passes_separation": False,
            "restored_passes_angle": False,
            "restored_passes_distance": False,
            "restored_passes_balance": False,
            "restored_passes_g2_geometry": False,
        }

    geometry = fork_geometry(p, c1, c2, view.scale)

    passes_separation = bool(geometry.daughter_separation_um >= config.division_daughter_sep_min_um)
    passes_angle = bool(
        np.isfinite(geometry.division_angle_deg) and geometry.division_angle_deg >= config.division_angle_min_deg
    )
    passes_distance = bool(geometry.max_parent_daughter_um <= config.division_max_parent_daughter_um)
    passes_balance = bool(geometry.distance_balance <= config.division_distance_balance_max)

    return {
        "restored_daughter_separation_um": geometry.daughter_separation_um,
        "restored_division_angle_deg": geometry.division_angle_deg,
        "restored_max_parent_daughter_um": geometry.max_parent_daughter_um,
        "restored_distance_balance": geometry.distance_balance,
        "restored_passes_separation": passes_separation,
        "restored_passes_angle": passes_angle,
        "restored_passes_distance": passes_distance,
        "restored_passes_balance": passes_balance,
        "restored_passes_g2_geometry": bool(passes_separation and passes_angle and passes_distance and passes_balance),
    }


def gt_fork_geometry(
    gt_view: RawGraphView,
    gt_parent: int,
    gt_daughter1: int,
    gt_daughter2: int,
    config=FROZEN_V9_CONFIG,
) -> dict[str, Any]:
    """Frozen G2 geometry of the *ground-truth* division itself.

    Answers a question the prediction alone cannot: is this division compatible
    with the frozen G2 geometry policy at all? When the GT fork already violates
    a threshold, no amount of association recovery can turn it into a G2
    division — the policy, not the predictor, is the binding constraint.
    """
    geometry_keys = {
        "gt_daughter_separation_um": "daughter_separation_um",
        "gt_division_angle_deg": "division_angle_deg",
        "gt_max_parent_daughter_um": "max_parent_daughter_um",
        "gt_distance_balance": "distance_balance",
    }

    p = gt_view.raw_position(gt_parent)
    c1 = gt_view.raw_position(gt_daughter1)
    c2 = gt_view.raw_position(gt_daughter2)

    if p is None or c1 is None or c2 is None:
        row: dict[str, Any] = dict.fromkeys(geometry_keys, float("nan"))
        row.update(
            {
                "gt_passes_separation": False,
                "gt_passes_angle": False,
                "gt_passes_distance": False,
                "gt_passes_balance": False,
                "gt_passes_g2_geometry": False,
            }
        )
        return row

    geometry = fork_geometry(p, c1, c2, gt_view.scale)

    passes_separation = bool(geometry.daughter_separation_um >= config.division_daughter_sep_min_um)
    passes_angle = bool(
        np.isfinite(geometry.division_angle_deg) and geometry.division_angle_deg >= config.division_angle_min_deg
    )
    passes_distance = bool(geometry.max_parent_daughter_um <= config.division_max_parent_daughter_um)
    passes_balance = bool(geometry.distance_balance <= config.division_distance_balance_max)

    row = {out: getattr(geometry, attr) for out, attr in geometry_keys.items()}
    row.update(
        {
            "gt_passes_separation": passes_separation,
            "gt_passes_angle": passes_angle,
            "gt_passes_distance": passes_distance,
            "gt_passes_balance": passes_balance,
            "gt_passes_g2_geometry": bool(passes_separation and passes_angle and passes_distance and passes_balance),
        }
    )
    return row


def case_structure(status1: str, status2: str) -> str:
    """Case-level shape from the two required-edge statuses."""
    present = int(status1 == EDGE_PRESENT) + int(status2 == EDGE_PRESENT)

    if present == 2:
        return CASE_BOTH_EDGES_PRESENT
    if present == 1:
        return CASE_ONE_EDGE_PRESENT
    return CASE_ZERO_REQUIRED_EDGES


def _root_cause(status: str) -> str | None:
    return EDGE_STATUS_MEANING[status]["root_cause"]


def _determinable(status: str) -> bool:
    return bool(EDGE_STATUS_MEANING[status]["determinable_from_geff"])


# ============================================================
# Case table
# ============================================================


def build_case_row(
    view: RawGraphView,
    *,
    dataset: str,
    prefix: str,
    gt_parent: Any,
    gt_daughter_1: Any,
    gt_daughter_2: Any,
    parent: int,
    daughter1: int,
    daughter2: int,
    gt_view: RawGraphView | None = None,
    max_children: int = FROZEN_PREDICT_CONFIG["max_children_per_node"],
) -> dict[str, Any]:
    """One audited GT division: what the saved GEFF says about both edges."""
    parent = int(parent)
    daughter1 = int(daughter1)
    daughter2 = int(daughter2)
    required = {daughter1, daughter2}

    status1 = classify_missing_edge(view, parent, daughter1, max_children=max_children)
    status2 = classify_missing_edge(view, parent, daughter2, max_children=max_children)

    edge1 = view.edge_between(parent, daughter1)
    edge2 = view.edge_between(parent, daughter2)

    competing_out = competing_outgoing_edges(view, parent, required)
    competing_in_1 = competing_incoming_edges(view, daughter1, parent)
    competing_in_2 = competing_incoming_edges(view, daughter2, parent)

    parent_t = view.time_of(parent)
    d1_t = view.time_of(daughter1)
    d2_t = view.time_of(daughter2)

    missing_statuses = [s for s in (status1, status2) if s != EDGE_PRESENT]
    determinable = all(_determinable(s) for s in missing_statuses)
    root_causes = sorted({c for c in (_root_cause(s) for s in missing_statuses) if c})

    row: dict[str, Any] = {
        "dataset": dataset,
        "prefix": prefix,
        "gt_parent": gt_parent,
        "gt_daughter_1": gt_daughter_1,
        "gt_daughter_2": gt_daughter_2,
        "parent": parent,
        "daughter_1": daughter1,
        "daughter_2": daughter2,
        "parent_t": parent_t,
        "daughter_1_t": d1_t,
        "daughter_2_t": d2_t,
        "daughter_1_frame_gap": (None if (parent_t is None or d1_t is None) else d1_t - parent_t),
        "daughter_2_frame_gap": (None if (parent_t is None or d2_t is None) else d2_t - parent_t),
        "raw_parent_out_degree": view.out_degree(parent),
        "daughter_1_in_degree": view.in_degree(daughter1),
        "daughter_2_in_degree": view.in_degree(daughter2),
        "edge_1_status": status1,
        "edge_2_status": status2,
        "case_structure": case_structure(status1, status2),
        "edge_1_prob": None if edge1 is None else edge1.edge_prob,
        "edge_2_prob": None if edge2 is None else edge2.edge_prob,
        "edge_1_dist_voxels": None if edge1 is None else edge1.edge_dist,
        "edge_2_dist_voxels": None if edge2 is None else edge2.edge_dist,
        "daughter_1_distance_um": view.distance_um(parent, daughter1),
        "daughter_2_distance_um": view.distance_um(parent, daughter2),
        "competing_out_count": len(competing_out),
        "competing_out_targets": ",".join(str(e.target) for e in competing_out),
        "competing_out_probs": ",".join(f"{e.edge_prob:.6f}" for e in competing_out),
        "competing_out_max_prob": (max((e.edge_prob for e in competing_out), default=float("nan"))),
        "competing_in_1_count": len(competing_in_1),
        "competing_in_1_sources": ",".join(str(e.source) for e in competing_in_1),
        "competing_in_1_probs": ",".join(f"{e.edge_prob:.6f}" for e in competing_in_1),
        "competing_in_2_count": len(competing_in_2),
        "competing_in_2_sources": ",".join(str(e.source) for e in competing_in_2),
        "competing_in_2_probs": ",".join(f"{e.edge_prob:.6f}" for e in competing_in_2),
        "competitor_distance_um": _competitor_distance(view, competing_in_1, competing_in_2, parent),
        "competitor_1_to_daughter_um": _competitor_to_daughter(view, competing_in_1, daughter1),
        "competitor_2_to_daughter_um": _competitor_to_daughter(view, competing_in_2, daughter2),
        "missing_daughter_max_distance_um": _missing_daughter_distance(
            view, parent, daughter1, daughter2, status1, status2
        ),
        **restored_fork_geometry(view, parent, daughter1, daughter2),
        **(
            gt_fork_geometry(gt_view, int(gt_parent), int(gt_daughter_1), int(gt_daughter_2))
            if gt_view is not None
            else {}
        ),
        "matching_suppression_indicated": any(s == DAUGHTER_CLAIMED_BY_COMPETITOR for s in (status1, status2)),
        "degree_cap_possible": any(s == DAUGHTER_UNCLAIMED_PARENT_AT_CAP for s in (status1, status2)),
        "root_cause_determinable_from_geff": determinable,
        "root_causes": ",".join(root_causes),
    }

    row["recovery_blocker"] = BLOCKER_ASSOCIATION_ONLY if row["restored_passes_g2_geometry"] else BLOCKER_G2_GEOMETRY

    if "gt_passes_g2_geometry" in row:
        row["association_only_and_gt_g2_compatible"] = bool(
            row["restored_passes_g2_geometry"] and row["gt_passes_g2_geometry"]
        )

    return row


def _missing_daughter_distance(
    view: RawGraphView,
    parent: int,
    daughter1: int,
    daughter2: int,
    status1: str,
    status2: str,
) -> float:
    """Largest parent->daughter distance among the edges that are missing."""
    distances = [
        view.distance_um(parent, daughter)
        for daughter, status in ((daughter1, status1), (daughter2, status2))
        if status != EDGE_PRESENT
    ]
    return max(distances) if distances else float("nan")


def _competitor_to_daughter(
    view: RawGraphView,
    competing_in: tuple[EdgeRecord, ...],
    daughter: int,
) -> float:
    """Distance from the competitor that claimed *daughter* to that daughter."""
    if not competing_in:
        return float("nan")
    return min(view.distance_um(edge.source, daughter) for edge in competing_in)


def _competitor_distance(
    view: RawGraphView,
    competing_in_1: tuple[EdgeRecord, ...],
    competing_in_2: tuple[EdgeRecord, ...],
    parent: int,
) -> float:
    """Distance from the true parent to the competitor that claimed a daughter."""
    competitors = [e.source for e in (*competing_in_1, *competing_in_2)]
    if not competitors:
        return float("nan")
    return min(view.distance_um(parent, source) for source in competitors)


def build_case_table(
    cases: pd.DataFrame,
    views: dict[str, RawGraphView],
    *,
    gt_views: dict[str, RawGraphView] | None = None,
    max_children: int = FROZEN_PREDICT_CONFIG["max_children_per_node"],
) -> pd.DataFrame:
    """Audit every row of *cases* against its dataset's :class:`RawGraphView`.

    *cases* must carry the Stage 11.01 columns ``dataset``, ``prefix``,
    ``gt_parent``, ``gt_daughter_1``, ``gt_daughter_2``, ``pred_parent``,
    ``pred_daughter_1`` and ``pred_daughter_2``. Rows are emitted in the input
    order, so the table is deterministic.
    """
    rows = [
        build_case_row(
            views[row.dataset],
            dataset=row.dataset,
            prefix=row.prefix,
            gt_parent=row.gt_parent,
            gt_daughter_1=row.gt_daughter_1,
            gt_daughter_2=row.gt_daughter_2,
            parent=int(row.pred_parent),
            daughter1=int(row.pred_daughter_1),
            daughter2=int(row.pred_daughter_2),
            gt_view=None if gt_views is None else gt_views.get(row.dataset),
            max_children=max_children,
        )
        for row in cases.itertuples(index=False)
    ]

    return pd.DataFrame(rows)


def summarize_cases(case_table: pd.DataFrame) -> dict[str, Any]:
    """Aggregate the case table into the Stage 11.02 headline counts."""
    statuses = pd.concat([case_table["edge_1_status"], case_table["edge_2_status"]], ignore_index=True)

    missing = statuses[statuses != EDGE_PRESENT]

    determinable = int(case_table["root_cause_determinable_from_geff"].sum())

    return {
        "n_cases": len(case_table),
        "case_structure_counts": {
            structure: int((case_table["case_structure"] == structure).sum()) for structure in CASE_STRUCTURES
        },
        "edge_status_counts": {status: int((statuses == status).sum()) for status in EDGE_STATUSES},
        "missing_edges": len(missing),
        "cases_determinable_from_geff": determinable,
        "cases_requiring_raw_scores": len(case_table) - determinable,
        "cases_with_competitor_claim": int(case_table["matching_suppression_indicated"].sum()),
        "cases_with_possible_degree_cap": int(case_table["degree_cap_possible"].sum()),
        "root_cause_counts": {
            cause: int(case_table["root_causes"].str.contains(cause, regex=False).sum())
            for cause in ROOT_CAUSE_HYPOTHESES
            if cause != "OTHER"
        },
        "restored_fork_passes_g2_geometry": int(case_table["restored_passes_g2_geometry"].sum()),
        "restored_fork_fails_g2_geometry": int((~case_table["restored_passes_g2_geometry"]).sum()),
        "restored_geometry_failure_counts": {
            flag: int((~case_table[f"restored_passes_{flag}"]).sum())
            for flag in ("separation", "angle", "distance", "balance")
        },
        "gt_fork_passes_g2_geometry": (
            int(case_table["gt_passes_g2_geometry"].sum()) if "gt_passes_g2_geometry" in case_table else None
        ),
        "gt_geometry_failure_counts": (
            {
                flag: int((~case_table[f"gt_passes_{flag}"]).sum())
                for flag in ("separation", "angle", "distance", "balance")
            }
            if "gt_passes_g2_geometry" in case_table
            else None
        ),
        "recovery_blocker_counts": {
            blocker: int((case_table["recovery_blocker"] == blocker).sum()) for blocker in RECOVERY_BLOCKERS
        },
        "association_only_and_gt_g2_compatible": (
            int(case_table["association_only_and_gt_g2_compatible"].sum())
            if "association_only_and_gt_g2_compatible" in case_table
            else None
        ),
        "recoverable_cases": int(
            (case_table["restored_passes_g2_geometry"] & case_table["root_cause_determinable_from_geff"]).sum()
        ),
        "prefix_counts": case_table["prefix"].value_counts().to_dict(),
    }


def load_views(
    datasets: list[str],
    pred_dirs: dict[str, Path],
    scales: dict[str, tuple[float, float, float]],
    loader,
) -> dict[str, RawGraphView]:
    """Load one :class:`RawGraphView` per dataset from its prediction GEFF.

    *loader* is injected (normally ``frozen_v9.load_graph``) so this stays free
    of IO policy and remains testable without touching disk.
    """
    return {name: graph_view(loader(Path(pred_dirs[name]) / f"{name}.geff"), name, scales[name]) for name in datasets}
