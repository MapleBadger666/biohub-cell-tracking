"""Stage 11.01 — division error decomposition audit.

Read-only diagnostics for the Frozen V9 pipeline. This module answers a single
research question: for every ground-truth division in a split, **what is the
earliest stage at which the pipeline lost it?**

It introduces no new algorithm and no second matching system:

- graph processing is delegated to :mod:`biohub_cell_tracking.frozen_v9`; the
  frozen order (G2 -> K9 -> single-edge recovery -> strict two-edge recovery ->
  cleanup) is preserved exactly, this module only snapshots between stages;
- node/division matching is delegated to the source-locked official metric
  (``tracking_cellmot.division_metrics``): ``extract_divisions``,
  ``match_divisions`` and ``score_divisions`` are the same calls the locked
  evaluator makes, so a row classified ``DIVISION_TP`` here is a division TP
  there by construction;
- the G2 geometry measured here (:func:`fork_geometry`) mirrors the frozen
  ``apply_frozen_g2`` computation; :func:`assert_candidates_match_g2` checks
  that measurement against the real post-G2 graph on every dataset.

Nothing here trains, runs inference, writes a prediction GEFF, or mutates GT.

The central distinction the audit exists to draw:

``DETECTION_LIMITED``
    a required biological node is not represented by a matched prediction.

``LINKING_LIMITED``
    parent and both daughters are already matched; graph construction or the
    frozen division policy is what failed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tracksdata as td
from tracking_cellmot.division_metrics import (
    _match_full,
    _matched_node_attrs,
    extract_divisions,
    match_divisions,
    score_divisions,
)
from tracking_cellmot.io import open_dataset

from .frozen_v9 import (
    FROZEN_V9_CONFIG,
    FrozenV9Application,
    FrozenV9Config,
    FrozenV9Policies,
    add_available_repairs,
    add_two_edge_bridges,
    apply_frozen_component_cleanup,
    apply_frozen_g2,
    apply_low_conf_component_pruning,
    load_graph,
    node_coordinate_keys,
)

__all__ = [
    "ASSOCIATION_GAP_KINDS",
    "CLASSIFICATIONS",
    "DETECTION_LIMITED",
    "FP_STRUCTURE_CLASSES",
    "LINKING_LIMITED",
    "DatasetDivisionAudit",
    "DivisionAudit",
    "DivisionCandidate",
    "ForkGeometry",
    "StageSnapshot",
    "StagedFrozenV9",
    "assert_candidates_match_g2",
    "audit_dataset",
    "build_division_candidates",
    "classify_gt_division",
    "fn_bottleneck_summary",
    "fork_geometry",
    "geometry_distributions",
    "run_staged_pipeline",
    "snapshot_graph",
    "staged_frozen_v9",
    "summarize_classifications",
    "summarize_fp_structure",
    "summarize_gt_by_prefix",
    "summarize_pred_fork_scope",
]


# ============================================================
# Classification vocabulary
# ============================================================

DIVISION_TP = "DIVISION_TP"
PARENT_MISSING = "PARENT_MISSING"
DAUGHTERS_MISSING_2 = "DAUGHTERS_MISSING_2"
DAUGHTER_MISSING_1 = "DAUGHTER_MISSING_1"
NODES_PRESENT_EDGES_MISSING = "NODES_PRESENT_EDGES_MISSING"
CANDIDATE_REJECT_P2 = "CANDIDATE_REJECT_P2"
CANDIDATE_REJECT_GEOMETRY = "CANDIDATE_REJECT_GEOMETRY"
CANDIDATE_ACCEPTED_LATER_LOST = "CANDIDATE_ACCEPTED_LATER_LOST"
OTHER = "OTHER"

#: Every classification a GT division row can take, in report order.
CLASSIFICATIONS: tuple[str, ...] = (
    DIVISION_TP,
    PARENT_MISSING,
    DAUGHTERS_MISSING_2,
    DAUGHTER_MISSING_1,
    NODES_PRESENT_EDGES_MISSING,
    CANDIDATE_REJECT_P2,
    CANDIDATE_REJECT_GEOMETRY,
    CANDIDATE_ACCEPTED_LATER_LOST,
    OTHER,
)

#: Failures where a required biological node is simply not represented.
DETECTION_LIMITED: frozenset[str] = frozenset({PARENT_MISSING, DAUGHTERS_MISSING_2, DAUGHTER_MISSING_1})

#: Failures where all three nodes exist and the graph/policy is what failed.
LINKING_LIMITED: frozenset[str] = frozenset(
    {
        NODES_PRESENT_EDGES_MISSING,
        CANDIDATE_REJECT_P2,
        CANDIDATE_REJECT_GEOMETRY,
        CANDIDATE_ACCEPTED_LATER_LOST,
        OTHER,
    }
)

FP_STRUCTURALLY_IMPLAUSIBLE = "STRUCTURALLY_IMPLAUSIBLE"
FP_PLAUSIBLE_UNLABELED = "PLAUSIBLY_UNLABELED_SPARSE_GT"
FP_GEOMETRICALLY_PLAUSIBLE = "GEOMETRICALLY_PLAUSIBLE_CONTRADICTS_GT"

FP_STRUCTURE_CLASSES: tuple[str, ...] = (
    FP_STRUCTURALLY_IMPLAUSIBLE,
    FP_PLAUSIBLE_UNLABELED,
    FP_GEOMETRICALLY_PLAUSIBLE,
)


# ============================================================
# Frozen G2 geometry, measured rather than enforced
# ============================================================


@dataclass(frozen=True)
class ForkGeometry:
    """Physical geometry of one predicted parent -> (child1, child2) fork.

    Computed exactly the way ``frozen_v9.apply_frozen_g2`` computes it: raw zyx
    coordinates multiplied by the dataset's physical scale, ``child1`` being the
    higher-probability daughter.
    """

    d1_um: float
    d2_um: float
    daughter_separation_um: float
    division_angle_deg: float
    max_parent_daughter_um: float
    distance_balance: float


def fork_geometry(parent_zyx, child1_zyx, child2_zyx, scale) -> ForkGeometry:
    """Measure the frozen G2 geometry of one fork.

    Mirrors the arithmetic inside ``frozen_v9.apply_frozen_g2`` — including the
    degenerate ``d1 == 0`` / ``d2 == 0`` case producing a NaN angle and the
    ``1e-12`` guard in the balance denominator — but applies no threshold.
    """
    scale = np.asarray(scale, dtype=float)

    p = np.asarray(parent_zyx, dtype=float) * scale
    c1 = np.asarray(child1_zyx, dtype=float) * scale
    c2 = np.asarray(child2_zyx, dtype=float) * scale

    v1 = c1 - p
    v2 = c2 - p

    d1 = float(np.linalg.norm(v1))
    d2 = float(np.linalg.norm(v2))

    daughter_sep = float(np.linalg.norm(c1 - c2))

    if d1 > 0 and d2 > 0:
        cosine = float(np.clip(float(np.dot(v1, v2) / (d1 * d2)), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
    else:
        angle = float("nan")

    return ForkGeometry(
        d1_um=d1,
        d2_um=d2,
        daughter_separation_um=daughter_sep,
        division_angle_deg=angle,
        max_parent_daughter_um=float(max(d1, d2)),
        distance_balance=float(abs(d1 - d2) / max(d1 + d2, 1e-12)),
    )


@dataclass(frozen=True)
class DivisionCandidate:
    """One out-degree-2 predicted fork, scored against the frozen G2 policy.

    ``target_high`` is the daughter G2 always keeps; ``target_low`` carries the
    second-edge probability ``p2`` that the policy actually adjudicates.
    """

    source_id: int
    target_high: int
    target_low: int
    p1: float
    p2: float
    geometry: ForkGeometry
    passes_p2: bool
    passes_separation: bool
    passes_angle: bool
    passes_distance: bool
    passes_balance: bool

    @property
    def targets(self) -> frozenset[int]:
        return frozenset({self.target_high, self.target_low})

    @property
    def passes_geometry(self) -> bool:
        """All four frozen geometry criteria pass (confidence excluded)."""
        return bool(self.passes_separation and self.passes_angle and self.passes_distance and self.passes_balance)

    @property
    def accepted(self) -> bool:
        """G2 keeps the second edge: confidence *and* geometry both pass."""
        return bool(self.passes_p2 and self.passes_geometry)

    @property
    def reject_reason_kind(self) -> str:
        """Why G2 dropped the second edge, or ``"accepted"``."""
        if self.accepted:
            return "accepted"
        if self.passes_p2:
            return "geometry_only"
        if self.passes_geometry:
            return "p2_only"
        return "p2_and_geometry"

    @property
    def failed_geometry_flags(self) -> tuple[str, ...]:
        failed = []
        if not self.passes_separation:
            failed.append("separation")
        if not self.passes_angle:
            failed.append("angle")
        if not self.passes_distance:
            failed.append("max_distance")
        if not self.passes_balance:
            failed.append("balance")
        return tuple(failed)


def build_division_candidates(
    graph,
    scale,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
) -> dict[int, DivisionCandidate]:
    """Score every out-degree-2 fork of *graph* against the frozen G2 policy.

    Fork selection and daughter ordering follow ``apply_frozen_g2`` exactly:
    only sources with *exactly* two outgoing edges are considered, and the two
    edges are ordered by ``(edge_prob desc, edge_id asc)``. Sources with three
    or more children are not adjudicated by G2 and are therefore absent here.
    """
    scale = np.asarray(scale, dtype=float)

    nodes = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")
    edges = graph.edge_attrs(attr_keys=["edge_prob"], unpack=True).to_pandas()

    candidates: dict[int, DivisionCandidate] = {}

    for source_id, group in edges.groupby("source_id"):
        if len(group) != 2:
            continue

        group = group.sort_values(["edge_prob", "edge_id"], ascending=[False, True])

        e1 = group.iloc[0]
        e2 = group.iloc[1]

        source_id = int(source_id)
        target1 = int(e1["target_id"])
        target2 = int(e2["target_id"])

        parent = nodes.loc[source_id]
        child1 = nodes.loc[target1]
        child2 = nodes.loc[target2]

        geometry = fork_geometry(
            (parent["z"], parent["y"], parent["x"]),
            (child1["z"], child1["y"], child1["x"]),
            (child2["z"], child2["y"], child2["x"]),
            scale,
        )

        p2 = float(e2["edge_prob"])

        candidates[source_id] = DivisionCandidate(
            source_id=source_id,
            target_high=target1,
            target_low=target2,
            p1=float(e1["edge_prob"]),
            p2=p2,
            geometry=geometry,
            passes_p2=bool(p2 >= config.division_p2_min),
            passes_separation=bool(geometry.daughter_separation_um >= config.division_daughter_sep_min_um),
            passes_angle=bool(
                np.isfinite(geometry.division_angle_deg)
                and geometry.division_angle_deg >= config.division_angle_min_deg
            ),
            passes_distance=bool(geometry.max_parent_daughter_um <= config.division_max_parent_daughter_um),
            passes_balance=bool(geometry.distance_balance <= config.division_distance_balance_max),
        )

    return candidates


# ============================================================
# Staged Frozen V9 execution
# ============================================================


@dataclass(frozen=True)
class StageSnapshot:
    """Node and directed-edge sets of the graph at one pipeline stage."""

    nodes: frozenset[int]
    edges: frozenset[tuple[int, int]]

    def has_fork(self, source: int, target_a: int, target_b: int) -> bool:
        return (source, target_a) in self.edges and (source, target_b) in self.edges


def snapshot_graph(graph) -> StageSnapshot:
    return StageSnapshot(
        nodes=frozenset(int(n) for n in graph.node_ids()),
        edges=frozenset((int(s), int(t)) for s, t in graph.edge_list()),
    )


@dataclass
class StagedFrozenV9:
    """One Frozen V9 run with a snapshot recorded after every stage.

    The graph transitions are produced by the frozen functions themselves, in
    the frozen order. Only the snapshots are new.
    """

    dataset: str
    graph: Any
    scale: np.ndarray
    application: FrozenV9Application
    candidates: dict[int, DivisionCandidate]
    raw: StageSnapshot
    after_g2: StageSnapshot
    after_k9: StageSnapshot
    after_single: StageSnapshot
    after_two_edge: StageSnapshot
    final: StageSnapshot

    def raw_successors(self) -> dict[int, set[int]]:
        succ: dict[int, set[int]] = defaultdict(set)
        for source, target in self.raw.edges:
            succ[source].add(target)
        return succ


def run_staged_pipeline(
    graph,
    high_conf_keys,
    scale,
    single_sub: pd.DataFrame,
    two_sub: pd.DataFrame,
    *,
    dataset: str = "",
    config: FrozenV9Config = FROZEN_V9_CONFIG,
    strict_single_policy: bool = True,
) -> StagedFrozenV9:
    """Run the frozen order on an already-loaded graph, snapshotting each stage.

    Equivalent to ``frozen_v9.prepare_precleanup_graph`` followed by
    ``frozen_v9.apply_frozen_v9`` on the same inputs: the same frozen functions
    are called with the same arguments in the same order. *graph* is mutated in
    place, exactly as the frozen pipeline mutates its own working copy.
    """
    scale = np.asarray(scale, dtype=float)

    raw = snapshot_graph(graph)
    candidates = build_division_candidates(graph, scale, config)

    apply_frozen_g2(graph, scale, config)
    after_g2 = snapshot_graph(graph)

    apply_low_conf_component_pruning(
        graph,
        high_conf_keys,
        config.confidence_component_limit,
    )
    after_k9 = snapshot_graph(graph)

    single_stats = add_available_repairs(graph, single_sub, config)

    if strict_single_policy and int(single_stats["added"]) != len(single_sub):
        raise AssertionError(f"{dataset}: single-edge policy unexpectedly not fully addable.\n{single_stats}")

    after_single = snapshot_graph(graph)

    two_stats = add_two_edge_bridges(graph, two_sub, config)
    after_two_edge = snapshot_graph(graph)

    cleanup_stats = apply_frozen_component_cleanup(graph, scale, config)
    final = snapshot_graph(graph)

    application = FrozenV9Application(
        dataset=dataset,
        single_added=int(single_stats["added"]),
        two_requested=int(two_stats["requested_bridges"]),
        two_added=int(two_stats["added_bridges"]),
        two_edges_added=int(two_stats["edges_added"]),
        two_missing_endpoint=int(two_stats["missing_endpoint"]),
        two_degree_conflict=int(two_stats["degree_conflict"]),
        two_existing_edge_conflict=int(two_stats["existing_edge_conflict"]),
        cleanup_removed=int(cleanup_stats["nodes_removed"]),
        cleanup_singleton_components=int(cleanup_stats["singleton_components"]),
        cleanup_long_two_node_components=int(cleanup_stats["long_two_node_components"]),
    )

    return StagedFrozenV9(
        dataset=dataset,
        graph=graph,
        scale=scale,
        application=application,
        candidates=candidates,
        raw=raw,
        after_g2=after_g2,
        after_k9=after_k9,
        after_single=after_single,
        after_two_edge=after_two_edge,
        final=final,
    )


def staged_frozen_v9(
    dataset_name: str,
    pred_995_dir: Path,
    pred_9975_dir: Path,
    policies: FrozenV9Policies,
    train_dir: Path,
    *,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
    strict_single_policy: bool = True,
) -> StagedFrozenV9:
    """Load one dataset's predictions and run the staged frozen pipeline.

    IO wrapper around :func:`run_staged_pipeline`. The prediction GEFFs on disk
    are read and never written.
    """
    pred_995_dir = Path(pred_995_dir)
    pred_9975_dir = Path(pred_9975_dir)
    train_dir = Path(train_dir)

    graph = load_graph(pred_995_dir / f"{dataset_name}.geff")
    high_graph = load_graph(pred_9975_dir / f"{dataset_name}.geff")

    scale = np.asarray(
        open_dataset(train_dir / dataset_name, load_image=False).scale,
        dtype=float,
    )

    single_sub = policies.single_policy[policies.single_policy["dataset"] == dataset_name].copy()
    two_sub = policies.strict_two_policy[policies.strict_two_policy["dataset"] == dataset_name].copy()

    return run_staged_pipeline(
        graph,
        node_coordinate_keys(high_graph),
        scale,
        single_sub,
        two_sub,
        dataset=dataset_name,
        config=config,
        strict_single_policy=strict_single_policy,
    )


def assert_candidates_match_g2(staged: StagedFrozenV9) -> None:
    """Check the measured candidate table reproduces the real G2 decision.

    For every out-degree-2 fork, ``candidate.accepted`` must equal "the second
    edge is still present after ``apply_frozen_g2``". This is the guard that
    keeps :func:`build_division_candidates` from drifting away from the frozen
    implementation it mirrors.
    """
    mismatches = []

    for candidate in staged.candidates.values():
        kept = (candidate.source_id, candidate.target_low) in staged.after_g2.edges
        if kept != candidate.accepted:
            mismatches.append((candidate.source_id, candidate.accepted, kept))

    if mismatches:
        raise AssertionError(
            f"{staged.dataset}: {len(mismatches)} candidate/G2 disagreement(s); "
            f"first: source={mismatches[0][0]} accepted={mismatches[0][1]} kept={mismatches[0][2]}"
        )


# ============================================================
# GT division classification
# ============================================================


def classify_gt_division(
    *,
    final_correct: bool,
    parent_matched: bool,
    daughter1_matched: bool,
    daughter2_matched: bool,
    candidate: DivisionCandidate | None,
    fork_present_final: bool,
) -> str:
    """Classify one GT division into the earliest stage that lost it.

    The order is deliberate: a stage can only be blamed once every earlier
    stage succeeded. A candidate failing *both* confidence and geometry is
    reported as ``CANDIDATE_REJECT_GEOMETRY`` so that the
    ``CANDIDATE_REJECT_P2`` bucket means "confidence was the only obstacle";
    ``DivisionCandidate.reject_reason_kind`` keeps the mixed case visible.
    """
    if final_correct:
        return DIVISION_TP

    if not parent_matched:
        return PARENT_MISSING

    if not daughter1_matched and not daughter2_matched:
        return DAUGHTERS_MISSING_2

    if not (daughter1_matched and daughter2_matched):
        return DAUGHTER_MISSING_1

    if candidate is None:
        return NODES_PRESENT_EDGES_MISSING

    if candidate.accepted:
        return CANDIDATE_ACCEPTED_LATER_LOST if not fork_present_final else OTHER

    if not candidate.passes_geometry:
        return CANDIDATE_REJECT_GEOMETRY

    if not candidate.passes_p2:
        return CANDIDATE_REJECT_P2

    return OTHER


def failure_mode(classification: str) -> str:
    """``DETECTION_LIMITED`` / ``LINKING_LIMITED`` / ``TP`` for one row."""
    if classification == DIVISION_TP:
        return "TP"
    if classification in DETECTION_LIMITED:
        return "DETECTION_LIMITED"
    return "LINKING_LIMITED"


def _pred_to_gt_map(matched_graph) -> dict[int, int]:
    attrs = _matched_node_attrs(matched_graph)
    if attrs.is_empty():
        return {}
    return {
        int(pred): int(gt)
        for pred, gt in zip(
            attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list(),
            strict=True,
        )
    }


def _invert_matches(pred_to_gt: dict[int, int]) -> dict[int, int]:
    """GT node -> lowest predicted node id matched to it (deterministic)."""
    gt_to_pred: dict[int, int] = {}
    for pred, gt in sorted(pred_to_gt.items()):
        gt_to_pred.setdefault(gt, pred)
    return gt_to_pred


def _find_candidate(
    staged: StagedFrozenV9,
    by_pair: dict[frozenset[int], list[DivisionCandidate]],
    pred_parent: int | None,
    pred_d1: int | None,
    pred_d2: int | None,
) -> tuple[DivisionCandidate | None, str]:
    """Locate the raw G2 candidate that should have produced this division.

    Preferred anchor is the predicted node matched to the GT parent. A fork one
    frame off — the official topology check tolerates it — is recovered by the
    daughter-pair fallback.
    """
    if pred_d1 is None or pred_d2 is None or pred_d1 == pred_d2:
        return None, "none"

    pair = frozenset({pred_d1, pred_d2})

    if pred_parent is not None:
        candidate = staged.candidates.get(pred_parent)
        if candidate is not None and candidate.targets == pair:
            return candidate, "parent"

    matches = by_pair.get(pair, [])
    if matches:
        return min(matches, key=lambda c: c.source_id), "daughter_pair"

    return None, "none"


# ============================================================
# Per-dataset audit
# ============================================================


@dataclass
class DatasetDivisionAudit:
    """Audit rows for one dataset."""

    dataset: str
    prefix: str
    gt_rows: list[dict[str, Any]]
    pred_rows: list[dict[str, Any]]
    division_tp: int
    division_fp: int
    division_fn: int


def _node_frame(graph) -> pd.DataFrame:
    return graph.node_attrs(unpack=True).to_pandas().set_index("node_id")


def audit_dataset(
    staged: StagedFrozenV9,
    gt_graph,
    *,
    prefix: str,
    config: FrozenV9Config = FROZEN_V9_CONFIG,
) -> DatasetDivisionAudit:
    """Decompose every GT division and every predicted fork for one dataset.

    Matching comes entirely from the official division metric: ``score_divisions``
    supplies TP/FP verdicts, ``match_divisions`` supplies the per-division node
    correspondence, and ``_match_full`` supplies the whole-graph correspondence
    used for the false-positive audit.
    """
    graph = staged.graph
    scale = tuple(staged.scale)
    max_distance = config.metric_max_distance
    dataset = staged.dataset

    scores = score_divisions(graph, gt_graph, scale, max_distance)
    matched_per_division = match_divisions(graph, gt_graph, scale, max_distance)
    gt_divisions = extract_divisions(gt_graph)

    full_pred_to_gt = _pred_to_gt_map(_match_full(graph, gt_graph, scale, max_distance))

    nodes = _node_frame(graph)
    gt_nodes = _node_frame(gt_graph)

    raw_succ = staged.raw_successors()

    by_pair: dict[frozenset[int], list[DivisionCandidate]] = defaultdict(list)
    for candidate in staged.candidates.values():
        by_pair[candidate.targets].append(candidate)

    final_succ: dict[int, list[int]] = defaultdict(list)
    final_pred: dict[int, list[int]] = defaultdict(list)
    for source, target in staged.final.edges:
        final_succ[source].append(target)
        final_pred[target].append(source)

    edge_prob = {
        (int(row.source_id), int(row.target_id)): float(row.edge_prob)
        for row in graph.edge_attrs(attr_keys=["edge_prob"], unpack=True).to_pandas().itertuples(index=False)
    }

    gt_rows = [
        _gt_division_row(
            staged=staged,
            dataset=dataset,
            prefix=prefix,
            div_node=div_node,
            gt_graph=gt_graph,
            gt_div=gt_divisions[div_node],
            matched_pred=matched_per_division[div_node],
            score=int(scores.scores.get(div_node, 0)),
            by_pair=by_pair,
            gt_nodes=gt_nodes,
            raw_succ=raw_succ,
        )
        for div_node in sorted(gt_divisions)
    ]

    pred_forks = sorted(node for node, targets in final_succ.items() if len(targets) >= 2)

    pred_rows = [
        _pred_division_row(
            staged=staged,
            dataset=dataset,
            prefix=prefix,
            fork=fork,
            nodes=nodes,
            gt_graph=gt_graph,
            final_succ=final_succ,
            final_pred=final_pred,
            edge_prob=edge_prob,
            pred_to_gt=full_pred_to_gt,
            raw_succ=raw_succ,
            config=config,
            is_tp=fork in scores.tp_forks,
            is_fp=fork in scores.fp_forks,
        )
        for fork in pred_forks
    ]

    tp = sum(scores.scores.values())

    return DatasetDivisionAudit(
        dataset=dataset,
        prefix=prefix,
        gt_rows=gt_rows,
        pred_rows=pred_rows,
        division_tp=tp,
        division_fp=len(scores.fp_forks),
        division_fn=len(scores.scores) - tp,
    )


ASSOCIATION_GAP_KINDS: tuple[str, ...] = (
    "both_proposed",
    "one_of_two_proposed",
    "neither_proposed",
    "no_outgoing_edge",
)


def _association_gap_kind(raw_out_degree: int, daughters_proposed: int) -> str:
    """How much of the required parent->daughter pair the raw graph proposed.

    Distinguishes "the association model never offered the second edge" from
    "it offered both and the frozen policy dropped one", which is the actionable
    split inside ``NODES_PRESENT_EDGES_MISSING``.
    """
    if daughters_proposed >= 2:
        return "both_proposed"
    if raw_out_degree == 0:
        return "no_outgoing_edge"
    if daughters_proposed == 1:
        return "one_of_two_proposed"
    return "neither_proposed"


def _gt_division_row(
    *,
    staged: StagedFrozenV9,
    dataset: str,
    prefix: str,
    div_node: int,
    gt_graph,
    gt_div,
    matched_pred,
    score: int,
    by_pair: dict[frozenset[int], list[DivisionCandidate]],
    gt_nodes: pd.DataFrame,
    raw_succ: dict[int, set[int]],
) -> dict[str, Any]:
    gt_children = sorted(int(c) for c in gt_graph.successors(div_node))

    pred_to_gt = _pred_to_gt_map(matched_pred)
    gt_to_pred = _invert_matches(pred_to_gt)

    gt_d1 = gt_children[0] if len(gt_children) > 0 else None
    gt_d2 = gt_children[1] if len(gt_children) > 1 else None

    pred_parent = gt_to_pred.get(int(div_node))
    pred_d1 = gt_to_pred.get(gt_d1) if gt_d1 is not None else None
    pred_d2 = gt_to_pred.get(gt_d2) if gt_d2 is not None else None

    def lineage_matched(child: int | None) -> bool:
        if child is None:
            return False
        lineage = {child, *(int(g) for g in gt_div.successors(child))}
        return any(gt in lineage for gt in pred_to_gt.values())

    candidate, candidate_source = _find_candidate(staged, by_pair, pred_parent, pred_d1, pred_d2)

    fork_present_final = candidate is not None and staged.final.has_fork(
        candidate.source_id, candidate.target_high, candidate.target_low
    )

    classification = classify_gt_division(
        final_correct=bool(score == 1),
        parent_matched=pred_parent is not None,
        daughter1_matched=pred_d1 is not None,
        daughter2_matched=pred_d2 is not None,
        candidate=candidate,
        fork_present_final=fork_present_final,
    )

    raw_parent_children = raw_succ.get(pred_parent, set()) if pred_parent is not None else set()

    row: dict[str, Any] = {
        "dataset": dataset,
        "prefix": prefix,
        "gt_parent": int(div_node),
        "gt_daughter_1": gt_d1,
        "gt_daughter_2": gt_d2,
        "t": int(gt_nodes.loc[int(div_node), "t"]) if int(div_node) in gt_nodes.index else None,
        "gt_daughter_count": len(gt_children),
        "classification": classification,
        "failure_mode": failure_mode(classification),
        "parent_matched": pred_parent is not None,
        "daughter1_matched": pred_d1 is not None,
        "daughter2_matched": pred_d2 is not None,
        "daughter1_lineage_matched": lineage_matched(gt_d1),
        "daughter2_lineage_matched": lineage_matched(gt_d2),
        "pred_parent": pred_parent,
        "pred_daughter_1": pred_d1,
        "pred_daughter_2": pred_d2,
        "candidate_found": candidate is not None,
        "candidate_anchor": candidate_source,
        "candidate_source": candidate.source_id if candidate is not None else None,
        "raw_parent_out_degree": len(raw_parent_children),
        "raw_parent_daughters_proposed": sum(
            1 for daughter in (pred_d1, pred_d2) if daughter is not None and daughter in raw_parent_children
        ),
        "raw_parent_has_both_daughters": bool(
            pred_d1 is not None and pred_d2 is not None and {pred_d1, pred_d2} <= raw_parent_children
        ),
        "association_gap_kind": _association_gap_kind(
            len(raw_parent_children),
            sum(1 for daughter in (pred_d1, pred_d2) if daughter is not None and daughter in raw_parent_children),
        ),
        "final_division_correct": bool(score == 1),
        "final_fork_present": fork_present_final,
    }

    if candidate is None:
        row.update(
            {
                "p1": None,
                "p2": None,
                "daughter_separation_um": None,
                "division_angle_deg": None,
                "max_parent_daughter_um": None,
                "distance_balance": None,
                "passes_p2": None,
                "passes_separation": None,
                "passes_angle": None,
                "passes_distance": None,
                "passes_balance": None,
                "reject_reason_kind": None,
                "failed_geometry_flags": None,
                "accepted_by_g2": None,
                "survives_k9": None,
                "survives_recovery": None,
                "survives_cleanup": None,
            }
        )
        return row

    fork = (candidate.source_id, candidate.target_high, candidate.target_low)

    accepted = candidate.accepted
    survives_k9 = accepted and staged.after_k9.has_fork(*fork)
    survives_recovery = survives_k9 and staged.after_two_edge.has_fork(*fork)
    survives_cleanup = survives_recovery and staged.final.has_fork(*fork)

    row.update(
        {
            "p1": candidate.p1,
            "p2": candidate.p2,
            "daughter_separation_um": candidate.geometry.daughter_separation_um,
            "division_angle_deg": candidate.geometry.division_angle_deg,
            "max_parent_daughter_um": candidate.geometry.max_parent_daughter_um,
            "distance_balance": candidate.geometry.distance_balance,
            "passes_p2": candidate.passes_p2,
            "passes_separation": candidate.passes_separation,
            "passes_angle": candidate.passes_angle,
            "passes_distance": candidate.passes_distance,
            "passes_balance": candidate.passes_balance,
            "reject_reason_kind": candidate.reject_reason_kind,
            "failed_geometry_flags": ",".join(candidate.failed_geometry_flags),
            "accepted_by_g2": accepted,
            "survives_k9": survives_k9,
            "survives_recovery": survives_recovery,
            "survives_cleanup": survives_cleanup,
        }
    )

    return row


def _pred_division_row(
    *,
    staged: StagedFrozenV9,
    dataset: str,
    prefix: str,
    fork: int,
    nodes: pd.DataFrame,
    gt_graph,
    final_succ: dict[int, list[int]],
    final_pred: dict[int, list[int]],
    edge_prob: dict[tuple[int, int], float],
    pred_to_gt: dict[int, int],
    raw_succ: dict[int, set[int]],
    config: FrozenV9Config,
    is_tp: bool,
    is_fp: bool,
) -> dict[str, Any]:
    daughters = sorted(final_succ[fork])
    daughter_count = len(daughters)

    ranked = sorted(daughters, key=lambda d: (-edge_prob.get((fork, d), float("nan")), d))
    top_two = ranked[:2]

    parent = nodes.loc[fork]
    child1 = nodes.loc[top_two[0]]
    child2 = nodes.loc[top_two[1]]

    geometry = fork_geometry(
        (parent["z"], parent["y"], parent["x"]),
        (child1["z"], child1["y"], child1["x"]),
        (child2["z"], child2["y"], child2["x"]),
        staged.scale,
    )

    probs = [edge_prob.get((fork, d), float("nan")) for d in ranked]
    p1 = probs[0] if probs else float("nan")
    p2 = probs[1] if len(probs) > 1 else float("nan")

    passes_separation = geometry.daughter_separation_um >= config.division_daughter_sep_min_um
    passes_angle = bool(
        np.isfinite(geometry.division_angle_deg) and geometry.division_angle_deg >= config.division_angle_min_deg
    )
    passes_distance = geometry.max_parent_daughter_um <= config.division_max_parent_daughter_um
    passes_balance = geometry.distance_balance <= config.division_distance_balance_max
    passes_geometry = bool(passes_separation and passes_angle and passes_distance and passes_balance)

    merged_branch = any(len(final_pred.get(d, [])) > 1 for d in daughters)

    gt_parent = pred_to_gt.get(fork)
    gt_parent_out_degree = int(gt_graph.out_degree(gt_parent)) if gt_parent is not None else None
    daughters_matched = [d for d in daughters if d in pred_to_gt]

    from_g2 = fork in staged.candidates and staged.candidates[fork].accepted
    from_recovery = any((fork, d) not in staged.raw.edges for d in daughters)

    # GT sparsity is evidence about the PARENT's fate. When the parent is
    # matched to a GT node with at least one annotated child, GT states what
    # that cell did, so a predicted fork there contradicts GT rather than
    # filling an annotation gap — even if the extra branch leads to an object
    # GT never tracked.
    gt_evidence_absent = gt_parent is None or gt_parent_out_degree == 0

    sparse_gt_plausible = bool(passes_geometry and daughter_count == 2 and not merged_branch and gt_evidence_absent)

    if daughter_count > 2 or merged_branch or not passes_geometry:
        structure_class = FP_STRUCTURALLY_IMPLAUSIBLE
    elif sparse_gt_plausible:
        structure_class = FP_PLAUSIBLE_UNLABELED
    else:
        structure_class = FP_GEOMETRICALLY_PLAUSIBLE

    return {
        "dataset": dataset,
        "prefix": prefix,
        "parent_node": int(fork),
        "t": int(parent["t"]),
        "daughter_count": daughter_count,
        "daughters": ",".join(str(d) for d in daughters),
        "counted_as_tp": bool(is_tp),
        "counted_as_fp": bool(is_fp),
        "evaluated_by_metric": bool(is_tp or is_fp),
        "p1": float(p1),
        "p2": float(p2),
        "edge_probs": ",".join(f"{p:.6f}" for p in probs),
        "daughter_separation_um": geometry.daughter_separation_um,
        "division_angle_deg": geometry.division_angle_deg,
        "max_parent_daughter_um": geometry.max_parent_daughter_um,
        "distance_balance": geometry.distance_balance,
        "passes_separation": bool(passes_separation),
        "passes_angle": passes_angle,
        "passes_distance": bool(passes_distance),
        "passes_balance": bool(passes_balance),
        "passes_geometry": passes_geometry,
        "parent_matches_gt": gt_parent is not None,
        "gt_parent": gt_parent,
        "gt_parent_out_degree": gt_parent_out_degree,
        "gt_parent_is_division": bool(gt_parent_out_degree is not None and gt_parent_out_degree >= 2),
        "gt_parent_is_non_division_parent": bool(gt_parent_out_degree == 1),
        "gt_parent_annotation_ends": bool(gt_parent_out_degree == 0),
        "daughters_matched_count": len(daughters_matched),
        "all_nodes_match_gt": bool(gt_parent is not None and len(daughters_matched) == daughter_count),
        "unmatched_daughter_branches": daughter_count - len(daughters_matched),
        "gt_contradicts_division": bool(gt_parent_out_degree == 1),
        "gt_evidence_absent": bool(gt_evidence_absent),
        "merged_branch": merged_branch,
        "origin_g2_accepted": bool(from_g2),
        "origin_includes_recovered_edge": bool(from_recovery),
        "raw_parent_out_degree": len(raw_succ.get(fork, set())),
        "sparse_gt_plausible": sparse_gt_plausible,
        "fp_structure_class": structure_class,
    }


# ============================================================
# Aggregation
# ============================================================


@dataclass
class DivisionAudit:
    """Full-split audit result."""

    gt_divisions: pd.DataFrame
    pred_divisions: pd.DataFrame
    per_dataset: pd.DataFrame
    metric_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def division_tp(self) -> int:
        return int(self.per_dataset["division_tp"].sum())

    @property
    def division_fp(self) -> int:
        return int(self.per_dataset["division_fp"].sum())

    @property
    def division_fn(self) -> int:
        return int(self.per_dataset["division_fn"].sum())


def summarize_classifications(gt_divisions: pd.DataFrame) -> pd.DataFrame:
    """Counts and percentages per classification, in the canonical order.

    Percentages are of all GT divisions (``pct_of_gt``) and of the false
    negatives only (``pct_of_fn``), so the FN decomposition reads directly.
    """
    total = len(gt_divisions)
    n_fn = int((gt_divisions["classification"] != DIVISION_TP).sum()) if total else 0

    counts = gt_divisions["classification"].value_counts() if total else pd.Series(dtype=int)

    rows = []
    for classification in CLASSIFICATIONS:
        n = int(counts.get(classification, 0))
        is_fn = classification != DIVISION_TP
        rows.append(
            {
                "classification": classification,
                "failure_mode": failure_mode(classification),
                "n": n,
                "pct_of_gt": (100.0 * n / total) if total else float("nan"),
                "pct_of_fn": (100.0 * n / n_fn) if (is_fn and n_fn) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def summarize_gt_by_prefix(gt_divisions: pd.DataFrame) -> pd.DataFrame:
    """Per-prefix classification decomposition, one block per prefix."""
    frames = []

    for prefix in sorted(gt_divisions["prefix"].unique()):
        subset = gt_divisions[gt_divisions["prefix"] == prefix]
        block = summarize_classifications(subset)
        block.insert(0, "prefix", prefix)
        frames.append(block)

    if not frames:
        return pd.DataFrame(columns=["prefix", "classification", "failure_mode", "n", "pct_of_gt", "pct_of_fn"])

    return pd.concat(frames, ignore_index=True)


def fn_bottleneck_summary(gt_divisions: pd.DataFrame) -> dict[str, Any]:
    """Headline false-negative decomposition, as counts plus percentages of FN.

    Requires the ``reject_reason_kind`` column so the "only by confidence" and
    "only by geometry" buckets mean what they say.
    """
    fn = gt_divisions[gt_divisions["classification"] != DIVISION_TP]
    n_fn = len(fn)

    def share(mask: pd.Series) -> float:
        return (100.0 * int(mask.sum()) / n_fn) if n_fn else float("nan")

    is_parent_missing = fn["classification"] == PARENT_MISSING
    is_daughter_missing = fn["classification"].isin({DAUGHTERS_MISSING_2, DAUGHTER_MISSING_1})
    all_present = fn["classification"].isin(LINKING_LIMITED)
    accepted_lost = fn["classification"] == CANDIDATE_ACCEPTED_LATER_LOST

    # "Only" means exactly that: read the obstacle off reject_reason_kind rather
    # than off the classification label, because CANDIDATE_REJECT_GEOMETRY also
    # absorbs candidates that fail confidence *and* geometry.
    reasons = fn["reject_reason_kind"]
    p2_only = reasons == "p2_only"
    geometry_only = reasons == "geometry_only"
    both_failed = reasons == "p2_and_geometry"
    no_candidate = fn["classification"] == NODES_PRESENT_EDGES_MISSING

    counts = {
        "n_fn": n_fn,
        "parent_missing": int(is_parent_missing.sum()),
        "daughter_missing": int(is_daughter_missing.sum()),
        "all_nodes_present": int(all_present.sum()),
        "no_candidate_generated": int(no_candidate.sum()),
        "rejected_confidence_only": int(p2_only.sum()),
        "rejected_geometry_only": int(geometry_only.sum()),
        "rejected_confidence_and_geometry": int(both_failed.sum()),
        "accepted_then_lost": int(accepted_lost.sum()),
        "detection_limited": int((fn["failure_mode"] == "DETECTION_LIMITED").sum()),
        "linking_limited": int((fn["failure_mode"] == "LINKING_LIMITED").sum()),
    }

    pct = {
        "pct_parent_missing": share(is_parent_missing),
        "pct_daughter_missing": share(is_daughter_missing),
        "pct_all_nodes_present": share(all_present),
        "pct_no_candidate_generated": share(no_candidate),
        "pct_rejected_confidence_only": share(p2_only),
        "pct_rejected_geometry_only": share(geometry_only),
        "pct_rejected_confidence_and_geometry": share(both_failed),
        "pct_accepted_then_lost": share(accepted_lost),
        "pct_detection_limited": share(fn["failure_mode"] == "DETECTION_LIMITED"),
        "pct_linking_limited": share(fn["failure_mode"] == "LINKING_LIMITED"),
    }

    largest = ""
    if n_fn:
        largest = str(fn["classification"].value_counts().idxmax())

    return {**counts, **pct, "largest_group": largest}


GEOMETRY_COLUMNS: tuple[str, ...] = (
    "p2",
    "daughter_separation_um",
    "division_angle_deg",
    "max_parent_daughter_um",
    "distance_balance",
)


def geometry_distributions(
    gt_divisions: pd.DataFrame,
    *,
    group_column: str = "classification",
) -> pd.DataFrame:
    """Per-group distribution of the frozen G2 measurements.

    Only rows that actually have a candidate contribute; rows without one have
    no geometry to describe.
    """
    subset = gt_divisions[gt_divisions["candidate_found"]]

    rows = []
    for group, block in subset.groupby(group_column, dropna=False):
        row: dict[str, Any] = {group_column: group, "n": len(block)}
        for column in GEOMETRY_COLUMNS:
            values = pd.to_numeric(block[column], errors="coerce").dropna()
            row[f"{column}_min"] = float(values.min()) if len(values) else float("nan")
            row[f"{column}_median"] = float(values.median()) if len(values) else float("nan")
            row[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{column}_max"] = float(values.max()) if len(values) else float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_column, ignore_index=True)


def summarize_pred_fork_scope(pred_divisions: pd.DataFrame) -> pd.DataFrame:
    """Split every predicted fork by what the locked metric did with it.

    ``not_evaluated`` forks are the ones sparse GT shields: the metric only
    charges a fork as FP when its parent matches a GT node with at least one
    annotated child.
    """
    total = len(pred_divisions)

    def block(label: str, mask: pd.Series) -> dict[str, Any]:
        subset = pred_divisions[mask]
        return {
            "scope": label,
            "n": len(subset),
            "pct": (100.0 * len(subset) / total) if total else float("nan"),
            "parent_matches_gt": int(subset["parent_matches_gt"].sum()),
            "gt_evidence_absent": int(subset["gt_evidence_absent"].sum()),
            "gt_contradicts_division": int(subset["gt_contradicts_division"].sum()),
            "passes_geometry": int(subset["passes_geometry"].sum()),
        }

    return pd.DataFrame(
        [
            block("division_tp", pred_divisions["counted_as_tp"]),
            block("division_fp", pred_divisions["counted_as_fp"]),
            block("not_evaluated", ~pred_divisions["evaluated_by_metric"]),
        ]
    )


def summarize_fp_structure(pred_divisions: pd.DataFrame) -> pd.DataFrame:
    """Structural breakdown of the predicted forks the metric counts as FP."""
    fp = pred_divisions[pred_divisions["counted_as_fp"]]
    total = len(fp)

    rows = []
    for structure_class in FP_STRUCTURE_CLASSES:
        n = int((fp["fp_structure_class"] == structure_class).sum()) if total else 0
        rows.append(
            {
                "fp_structure_class": structure_class,
                "n": n,
                "pct_of_fp": (100.0 * n / total) if total else float("nan"),
            }
        )

    return pd.DataFrame(rows)
