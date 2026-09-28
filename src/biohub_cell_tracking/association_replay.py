"""Stage 11.04 — targeted association logit replay and fork evidence audit.

Recovers the raw association evidence the frozen predictor discards before GEFF
serialisation, for the small set of judgeable division cases only.

The frozen predictor scores every ``(source, target)`` pair densely, applies
``softmax(dim=0)`` — a *column-wise* normalisation over parents — keeps pairs
above ``0.5``, then sweeps greedily under degree caps. Everything below the
threshold is freed per frame pair and never written to disk. This module replays
the exact same computation on a handful of frame pairs and keeps compact
per-edge evidence instead of the full matrices.

Two rank notions are used, and they are not interchangeable:

``column rank``
    rank of a source within a target's column, by column-softmax probability.
    This is the quantity the frozen threshold actually adjudicates, because the
    softmax normalises down columns.

``row rank``
    rank of a target within a source's row, by **raw logit**. The softmax is not
    monotonic along a row, so row ordering must come from the logits. Diagnostic
    only — no normalisation is changed anywhere.

Nothing here trains, writes a GEFF, or mutates a graph. Model-dependent replay
lives behind :class:`ReplayContext`; every scoring, ranking, classification and
rule function below it is pure and testable without a model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "CANDIDATE_RULES",
    "COMPETITOR_MARGIN_MAX",
    "DEGREE_CONFLICT",
    "HIGH_RANK_COMPETITOR_SUPPRESSED",
    "HIGH_RANK_THRESHOLD_SUPPRESSED",
    "LOW_RANK_MODEL_FAILURE",
    "MISSING_EDGE_CLASSES",
    "OTHER",
    "CandidateRule",
    "ColumnStats",
    "EdgeQuery",
    "FramePair",
    "RowStats",
    "classify_missing_edge",
    "column_stats",
    "dedupe_frame_pairs",
    "fork_evidence",
    "row_stats",
    "rule_candidate_edges",
    "rule_fork_counts",
]


# ============================================================
# Frame pairs
# ============================================================


@dataclass(frozen=True, order=True)
class FramePair:
    """One ``(dataset, t, t+1)`` association problem to replay."""

    dataset: str
    t: int

    @property
    def t_next(self) -> int:
        return self.t + 1

    @property
    def key(self) -> tuple[str, int]:
        return (self.dataset, self.t)


def dedupe_frame_pairs(pairs: list[FramePair]) -> tuple[FramePair, ...]:
    """Unique frame pairs in deterministic ``(dataset, t)`` order.

    Several division cases routinely share a frame pair; the model must run once
    per pair, not once per case.
    """
    return tuple(sorted(set(pairs)))


@dataclass(frozen=True)
class EdgeQuery:
    """One ``parent -> daughter`` association whose evidence is wanted."""

    dataset: str
    prefix: str
    case_type: str
    case_id: str
    t: int
    source_node: int
    target_node: int
    role: str = ""


# ============================================================
# Rank statistics — pure
# ============================================================


@dataclass(frozen=True)
class ColumnStats:
    """Where a source sits inside a target's column of the softmax matrix."""

    column_rank: int
    column_softmax_prob: float
    raw_logit: float
    column_top1_source: int
    column_top1_prob: float
    column_top2_source: int | None
    column_top2_prob: float
    column_margin_to_top1: float
    column_size: int

    @property
    def is_column_winner(self) -> bool:
        return self.column_rank == 1


def _ordered_indices(values: np.ndarray) -> np.ndarray:
    """Indices of *values* sorted descending, ties broken by ascending index.

    Deterministic by construction, so a tie can never reorder between runs.
    """
    return np.lexsort((np.arange(len(values)), -np.asarray(values, dtype=float)))


def column_stats(
    probs: np.ndarray,
    raw: np.ndarray,
    source_index: int,
    target_index: int,
) -> ColumnStats:
    """Rank and margin of ``source_index`` within column ``target_index``."""
    column = np.asarray(probs, dtype=float)[:, target_index]
    order = _ordered_indices(column)

    rank_of = {int(idx): position + 1 for position, idx in enumerate(order)}

    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else None

    return ColumnStats(
        column_rank=int(rank_of[int(source_index)]),
        column_softmax_prob=float(column[source_index]),
        raw_logit=float(np.asarray(raw, dtype=float)[source_index, target_index]),
        column_top1_source=top1,
        column_top1_prob=float(column[top1]),
        column_top2_source=top2,
        column_top2_prob=float(column[top2]) if top2 is not None else float("nan"),
        column_margin_to_top1=float(column[top1] - column[source_index]),
        column_size=len(column),
    )


@dataclass(frozen=True)
class RowStats:
    """Where a target sits inside a source's row of the raw-logit matrix."""

    row_rank_by_raw_logit: int
    raw_logit: float
    parent_top1_target: int
    parent_top2_target: int | None
    parent_top3_target: int | None
    row_size: int


def row_stats(raw: np.ndarray, source_index: int, target_index: int) -> RowStats:
    """Rank and top targets of ``source_index``, by raw logit.

    Uses logits rather than probabilities because ``softmax(dim=0)`` normalises
    columns, so probabilities are not comparable along a row.
    """
    row = np.asarray(raw, dtype=float)[source_index, :]
    order = _ordered_indices(row)

    rank_of = {int(idx): position + 1 for position, idx in enumerate(order)}
    tops = [int(idx) for idx in order[:3]]
    while len(tops) < 3:
        tops.append(None)

    return RowStats(
        row_rank_by_raw_logit=int(rank_of[int(target_index)]),
        raw_logit=float(row[target_index]),
        parent_top1_target=tops[0],
        parent_top2_target=tops[1],
        parent_top3_target=tops[2],
        row_size=len(row),
    )


# ============================================================
# Missing-edge classification — pure
# ============================================================

HIGH_RANK_THRESHOLD_SUPPRESSED = "HIGH_RANK_THRESHOLD_SUPPRESSED"
HIGH_RANK_COMPETITOR_SUPPRESSED = "HIGH_RANK_COMPETITOR_SUPPRESSED"
LOW_RANK_MODEL_FAILURE = "LOW_RANK_MODEL_FAILURE"
DEGREE_CONFLICT = "DEGREE_CONFLICT"
OTHER = "OTHER"

MISSING_EDGE_CLASSES: tuple[str, ...] = (
    HIGH_RANK_THRESHOLD_SUPPRESSED,
    HIGH_RANK_COMPETITOR_SUPPRESSED,
    LOW_RANK_MODEL_FAILURE,
    DEGREE_CONFLICT,
    OTHER,
)

#: Predeclared "small margin" for calling a rank-2 edge competitor-suppressed
#: rather than a model failure. Fixed before looking at the replayed scores.
COMPETITOR_MARGIN_MAX: float = 0.20


def classify_missing_edge(
    *,
    column_rank: int,
    column_softmax_prob: float,
    column_margin_to_top1: float,
    threshold: float = 0.50,
    parent_at_child_cap: bool = False,
    margin_max: float = COMPETITOR_MARGIN_MAX,
) -> str:
    """Why the frozen predictor did not emit this required association.

    Ordered so the earliest binding mechanism wins. An edge that clears the
    threshold as its column's winner *would* have been emitted, so its absence
    can only be the child-degree cap.
    """
    if column_rank == 1 and column_softmax_prob > threshold:
        return DEGREE_CONFLICT if parent_at_child_cap else OTHER

    if column_rank == 1:
        return HIGH_RANK_THRESHOLD_SUPPRESSED

    if column_rank == 2 and column_margin_to_top1 <= margin_max:
        return HIGH_RANK_COMPETITOR_SUPPRESSED

    return LOW_RANK_MODEL_FAILURE


# ============================================================
# Fork-level evidence — pure
# ============================================================


def fork_evidence(edge_a: dict[str, Any], edge_b: dict[str, Any]) -> dict[str, Any]:
    """Fork-level neural summary of a parent's two daughter associations.

    Descriptive only. These are the features a division discriminator could use
    if one were ever built; nothing here decides anything.
    """
    p_a = float(edge_a["column_softmax_prob"])
    p_b = float(edge_b["column_softmax_prob"])
    l_a = float(edge_a["raw_logit"])
    l_b = float(edge_b["raw_logit"])

    return {
        "min_prob": min(p_a, p_b),
        "max_prob": max(p_a, p_b),
        "prob_product": p_a * p_b,
        "prob_gap": abs(p_a - p_b),
        "min_raw_logit": min(l_a, l_b),
        "max_raw_logit": max(l_a, l_b),
        "sum_raw_logit": l_a + l_b,
        "max_column_rank": max(int(edge_a["column_rank"]), int(edge_b["column_rank"])),
        "min_column_rank": min(int(edge_a["column_rank"]), int(edge_b["column_rank"])),
        "max_row_rank": max(int(edge_a["row_rank_by_raw_logit"]), int(edge_b["row_rank_by_raw_logit"])),
        "max_competitor_margin": max(float(edge_a["column_margin_to_top1"]), float(edge_b["column_margin_to_top1"])),
        "both_column_rank_le_2": bool(int(edge_a["column_rank"]) <= 2 and int(edge_b["column_rank"]) <= 2),
        "both_row_rank_le_2": bool(
            int(edge_a["row_rank_by_raw_logit"]) <= 2 and int(edge_b["row_rank_by_raw_logit"]) <= 2
        ),
        "both_row_rank_le_3": bool(
            int(edge_a["row_rank_by_raw_logit"]) <= 3 and int(edge_b["row_rank_by_raw_logit"]) <= 3
        ),
    }


# ============================================================
# Candidate-generation rules — pure
# ============================================================


@dataclass(frozen=True)
class CandidateRule:
    """One predeclared candidate-generation rule.

    ``division_only`` marks rules whose extra candidates must never enter the
    ordinary association graph: exposing a low-ranked hypothesis to a division
    proposal layer is a different act from adding a normal edge, and conflating
    them would create merges.
    """

    name: str
    kind: str  # "column_threshold" | "column_topk" | "row_topk"
    value: float
    division_only: bool

    @property
    def label(self) -> str:
        return self.name


#: The predeclared rule set. No dense threshold search is performed.
CANDIDATE_RULES: tuple[CandidateRule, ...] = (
    CandidateRule("A_current_col_gt_0.50", "column_threshold", 0.50, False),
    CandidateRule("B_col_gt_0.40", "column_threshold", 0.40, False),
    CandidateRule("B_col_gt_0.30", "column_threshold", 0.30, False),
    CandidateRule("B_col_gt_0.20", "column_threshold", 0.20, False),
    CandidateRule("C_col_rank_le_2", "column_topk", 2, True),
    CandidateRule("D_row_rank_le_2", "row_topk", 2, True),
    CandidateRule("D_row_rank_le_3", "row_topk", 3, True),
)


def edge_visible(rule: CandidateRule, *, column_rank: int, column_prob: float, row_rank: int) -> bool:
    """Whether *rule* exposes one edge as a candidate."""
    if rule.kind == "column_threshold":
        return bool(column_prob > rule.value)
    if rule.kind == "column_topk":
        return bool(column_rank <= rule.value)
    if rule.kind == "row_topk":
        return bool(row_rank <= rule.value)
    raise ValueError(f"unknown rule kind: {rule.kind}")


def rule_candidate_edges(rule: CandidateRule, probs: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Boolean ``(n_src, n_tgt)`` mask of the edges *rule* exposes."""
    probs = np.asarray(probs, dtype=float)
    raw = np.asarray(raw, dtype=float)

    if rule.kind == "column_threshold":
        return probs > rule.value

    if rule.kind == "column_topk":
        mask = np.zeros(probs.shape, dtype=bool)
        k = int(rule.value)
        for j in range(probs.shape[1]):
            order = _ordered_indices(probs[:, j])[:k]
            mask[order, j] = True
        return mask

    if rule.kind == "row_topk":
        mask = np.zeros(raw.shape, dtype=bool)
        k = int(rule.value)
        for i in range(raw.shape[0]):
            order = _ordered_indices(raw[i, :])[:k]
            mask[i, order] = True
        return mask

    raise ValueError(f"unknown rule kind: {rule.kind}")


def rule_fork_counts(mask: np.ndarray) -> dict[str, int]:
    """Candidate-edge and candidate-fork counts implied by an exposure mask."""
    mask = np.asarray(mask, dtype=bool)
    per_source = mask.sum(axis=1)

    return {
        "candidate_edges": int(mask.sum()),
        "candidate_forks": int((per_source >= 2).sum()),
        "sources_with_any": int((per_source >= 1).sum()),
        "max_out_candidates": int(per_source.max()) if per_source.size else 0,
    }


# ============================================================
# Node mapping between replay and the saved GEFF
# ============================================================


@dataclass
class NodeMapping:
    """Replay detection index <-> saved GEFF node id, for one frame.

    The predictor assigns node ids in detection order per frame, so the mapping
    is positional — but it is only trusted after coordinates are checked
    element-wise. Any disagreement makes the mapping invalid and the frame pair
    uninterpretable; a nearby detection is never substituted.
    """

    dataset: str
    t: int
    node_ids: tuple[int, ...]
    replay_count: int
    saved_count: int
    coordinates_match: bool

    @property
    def is_valid(self) -> bool:
        return bool(self.coordinates_match and self.replay_count == self.saved_count)

    def index_of(self, node_id: int) -> int | None:
        try:
            return self.node_ids.index(int(node_id))
        except ValueError:
            return None


def build_node_mapping(
    dataset: str,
    t: int,
    replay_coords: np.ndarray,
    saved_frame: Any,
    downsample: tuple[float, float, float],
) -> NodeMapping:
    """Map replay detections onto saved GEFF nodes for one frame, fail-closed.

    *replay_coords* are ``(N, 4)`` ``[t, z, y, x]`` in downsampled space, exactly
    as ``predict_video`` holds them before its final rescale; *saved_frame* is
    the GEFF's nodes for this frame ordered by node id.
    """
    rescaled = np.asarray(replay_coords, dtype=np.float32).copy()
    rescaled[:, 1:] *= np.asarray(downsample, dtype=np.float32)
    rescaled = rescaled.astype(np.int16)

    saved = saved_frame[["t", "z", "y", "x"]].to_numpy().astype(np.int16)

    matches = rescaled.shape == saved.shape and bool(np.array_equal(rescaled, saved))

    return NodeMapping(
        dataset=dataset,
        t=int(t),
        node_ids=tuple(int(n) for n in saved_frame["node_id"]),
        replay_count=len(rescaled),
        saved_count=len(saved),
        coordinates_match=matches,
    )


# ============================================================
# Aggregation
# ============================================================


@dataclass
class ReplayTotals:
    """Rule-level exposure totals accumulated across replayed frame pairs."""

    per_rule: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def add(self, rule: CandidateRule, counts: dict[str, int]) -> None:
        bucket = self.per_rule[rule.name]
        for key, value in counts.items():
            bucket[key] = bucket.get(key, 0) + int(value)
        bucket["frame_pairs"] = bucket.get("frame_pairs", 0) + 1

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {"rule": name, **{k: int(v) for k, v in sorted(counts.items())}}
            for name, counts in sorted(self.per_rule.items())
        ]
