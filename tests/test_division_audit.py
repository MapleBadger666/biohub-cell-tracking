"""Focused tests for src/biohub_cell_tracking/division_audit.py.

Small deterministic in-memory graphs only. No dataset loads, no training, no
inference, no GT evaluation. The full Fold0 audit lives in
scripts/audit_stage11_divisions.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import tracksdata as td

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub_cell_tracking.division_audit import (
    ASSOCIATION_GAP_KINDS,
    CANDIDATE_ACCEPTED_LATER_LOST,
    CANDIDATE_REJECT_GEOMETRY,
    CANDIDATE_REJECT_P2,
    CLASSIFICATIONS,
    DAUGHTER_MISSING_1,
    DAUGHTERS_MISSING_2,
    DETECTION_LIMITED,
    DIVISION_TP,
    FP_STRUCTURE_CLASSES,
    LINKING_LIMITED,
    NODES_PRESENT_EDGES_MISSING,
    OTHER,
    PARENT_MISSING,
    StageSnapshot,
    _association_gap_kind,
    assert_candidates_match_g2,
    audit_dataset,
    build_division_candidates,
    classify_gt_division,
    failure_mode,
    fn_bottleneck_summary,
    fork_geometry,
    geometry_distributions,
    run_staged_pipeline,
    snapshot_graph,
    summarize_classifications,
    summarize_fp_structure,
    summarize_gt_by_prefix,
    summarize_pred_fork_scope,
)
from biohub_cell_tracking.frozen_v9 import (
    FROZEN_V9_CONFIG,
    add_available_repairs,
    add_two_edge_bridges,
    apply_frozen_component_cleanup,
    apply_frozen_g2,
    apply_low_conf_component_pruning,
    node_coordinate_keys,
)

UNIT_SCALE = (1.0, 1.0, 1.0)


# ============================================================
# Graph fixtures
# ============================================================


def new_graph():
    """Empty graph with the node/edge attribute schema the pipeline expects."""
    graph = td.graph.IndexedRXGraph()

    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)

    for key in ("edge_prob", "edge_dist"):
        graph.add_edge_attr_key(key, pl.Float64, 0.0)

    return graph


def add_node(graph, t: int, z: float = 0.0, y: float = 0.0, x: float = 0.0) -> int:
    return graph.add_node({"t": int(t), "z": float(z), "y": float(y), "x": float(x)})


def add_edge(graph, source: int, target: int, prob: float = 0.9) -> int:
    return graph.add_edge(source, target, {"edge_prob": float(prob), "edge_dist": 1.0})


def empty_single_policy() -> pd.DataFrame:
    return pd.DataFrame(columns=["dataset", "prefix", "source_id", "target_id"])


def empty_two_policy() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset",
            "prefix",
            "a_id",
            "b_id",
            "c_id",
            "cv_residual_um",
            "max_step_um",
            "step_balance",
            "turn_angle_deg",
            "ambiguity_margin_um",
            "pair_count",
            "conflict_free",
        ]
    )


def division_graph(
    *,
    p2: float = 0.95,
    d1: float = 4.0,
    d2: float = 4.0,
    angle_deg: float = 180.0,
):
    """A parent forking into two daughters with fully controlled geometry.

    ``d1`` / ``d2`` are the parent->daughter distances and ``angle_deg`` the
    angle between them, so daughter separation follows as
    ``sqrt(d1**2 + d2**2 - 2*d1*d2*cos(angle))``. Defaults satisfy every frozen
    G2 criterion.
    """
    theta = np.radians(angle_deg)

    graph = new_graph()

    parent = add_node(graph, t=0, y=0.0, x=0.0)
    child1 = add_node(graph, t=1, y=d1, x=0.0)
    child2 = add_node(graph, t=1, y=d2 * np.cos(theta), x=d2 * np.sin(theta))

    add_edge(graph, parent, child1, prob=0.99)
    add_edge(graph, parent, child2, prob=p2)

    return graph, parent, child1, child2


# Geometry presets, each failing exactly one frozen G2 criterion.
FAILS_SEPARATION = {"d1": 2.9, "d2": 2.9}
FAILS_ANGLE = {"d1": 6.0, "d2": 6.0, "angle_deg": 90.0}
FAILS_MAX_DISTANCE = {"d1": 9.0, "d2": 9.0}
FAILS_BALANCE = {"d1": 7.0, "d2": 2.0}


# ============================================================
# fork_geometry
# ============================================================


def test_fork_geometry_measures_a_clean_symmetric_division() -> None:
    geometry = fork_geometry((0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, -4.0, 0.0), UNIT_SCALE)

    assert geometry.d1_um == pytest.approx(4.0)
    assert geometry.d2_um == pytest.approx(4.0)
    assert geometry.daughter_separation_um == pytest.approx(8.0)
    assert geometry.division_angle_deg == pytest.approx(180.0)
    assert geometry.max_parent_daughter_um == pytest.approx(4.0)
    assert geometry.distance_balance == pytest.approx(0.0)


def test_fork_geometry_respects_anisotropic_scale() -> None:
    isotropic = fork_geometry((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    anisotropic = fork_geometry((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (4.0, 1.0, 1.0))

    assert isotropic.daughter_separation_um == pytest.approx(2.0)
    assert anisotropic.daughter_separation_um == pytest.approx(8.0)


def test_fork_geometry_angle_is_nan_for_a_zero_length_step() -> None:
    geometry = fork_geometry((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, -4.0, 0.0), UNIT_SCALE)

    assert np.isnan(geometry.division_angle_deg)
    assert geometry.distance_balance == pytest.approx(1.0)


# ============================================================
# build_division_candidates
# ============================================================


def test_candidate_orders_daughters_by_edge_probability() -> None:
    graph, parent, child1, child2 = division_graph(p2=0.85)

    candidates = build_division_candidates(graph, UNIT_SCALE)

    assert set(candidates) == {parent}
    candidate = candidates[parent]

    assert candidate.target_high == child1
    assert candidate.target_low == child2
    assert candidate.p1 == pytest.approx(0.99)
    assert candidate.p2 == pytest.approx(0.85)
    assert candidate.targets == frozenset({child1, child2})


def test_candidate_accepts_a_clean_division() -> None:
    graph, parent, _, _ = division_graph()

    candidate = build_division_candidates(graph, UNIT_SCALE)[parent]

    assert candidate.accepted
    assert candidate.passes_geometry
    assert candidate.reject_reason_kind == "accepted"
    assert candidate.failed_geometry_flags == ()


@pytest.mark.parametrize(
    ("kwargs", "flag", "expected_flags"),
    [
        ({"p2": 0.5}, "passes_p2", ()),
        (FAILS_SEPARATION, "passes_separation", ("separation",)),
        (FAILS_ANGLE, "passes_angle", ("angle",)),
        (FAILS_MAX_DISTANCE, "passes_distance", ("max_distance",)),
        (FAILS_BALANCE, "passes_balance", ("balance",)),
    ],
)
def test_candidate_flags_isolate_single_criteria(kwargs, flag, expected_flags) -> None:
    graph, parent, _, _ = division_graph(**kwargs)

    candidate = build_division_candidates(graph, UNIT_SCALE)[parent]

    assert getattr(candidate, flag) is False
    assert candidate.accepted is False
    assert candidate.failed_geometry_flags == expected_flags


def test_candidate_reject_reason_kind_separates_mixed_failures() -> None:
    graph, parent, _, _ = division_graph(p2=0.5, **FAILS_SEPARATION)

    candidate = build_division_candidates(graph, UNIT_SCALE)[parent]

    assert candidate.reject_reason_kind == "p2_and_geometry"


def test_candidate_table_skips_forks_g2_does_not_adjudicate() -> None:
    graph, parent, _, _ = division_graph()
    third = add_node(graph, t=1, y=0.0, x=4.0)
    add_edge(graph, parent, third, prob=0.7)

    # Out-degree 3: apply_frozen_g2 skips it, so the candidate table must too.
    assert build_division_candidates(graph, UNIT_SCALE) == {}


def test_candidates_agree_with_apply_frozen_g2_on_accept_and_reject() -> None:
    for kwargs in ({}, {"p2": 0.5}, FAILS_SEPARATION, FAILS_ANGLE, FAILS_MAX_DISTANCE, FAILS_BALANCE):
        graph, parent, _, _ = division_graph(**kwargs)

        candidate = build_division_candidates(graph, UNIT_SCALE)[parent]
        apply_frozen_g2(graph, np.asarray(UNIT_SCALE, dtype=float))

        second_edge_kept = (parent, candidate.target_low) in {(int(s), int(t)) for s, t in graph.edge_list()}

        assert second_edge_kept == candidate.accepted


# ============================================================
# Staged pipeline
# ============================================================


def staged_from(graph, *, high_keys=None, single=None, two=None):
    return run_staged_pipeline(
        graph,
        node_coordinate_keys(graph) if high_keys is None else high_keys,
        np.asarray(UNIT_SCALE, dtype=float),
        empty_single_policy() if single is None else single,
        empty_two_policy() if two is None else two,
        dataset="synthetic",
    )


def test_staged_pipeline_matches_the_frozen_function_sequence() -> None:
    graph_a, parent, child1, child2 = division_graph()
    # A short trailing track keeps the division component above cleanup size.
    tail = add_node(graph_a, t=2, y=4.0)
    add_edge(graph_a, child1, tail)

    graph_b, _, child1_b, _ = division_graph()
    tail_b = add_node(graph_b, t=2, y=4.0)
    add_edge(graph_b, child1_b, tail_b)

    staged = staged_from(graph_a)

    scale = np.asarray(UNIT_SCALE, dtype=float)
    high_keys = node_coordinate_keys(graph_b)
    apply_frozen_g2(graph_b, scale, FROZEN_V9_CONFIG)
    apply_low_conf_component_pruning(graph_b, high_keys, FROZEN_V9_CONFIG.confidence_component_limit)
    add_available_repairs(graph_b, empty_single_policy(), FROZEN_V9_CONFIG)
    add_two_edge_bridges(graph_b, empty_two_policy(), FROZEN_V9_CONFIG)
    apply_frozen_component_cleanup(graph_b, scale, FROZEN_V9_CONFIG)

    assert staged.final == snapshot_graph(graph_b)
    assert staged.final.has_fork(parent, child1, child2)


def test_staged_snapshots_record_the_g2_rejection() -> None:
    graph, parent, child1, child2 = division_graph(p2=0.5)
    tail = add_node(graph, t=2, y=4.0)
    add_edge(graph, child1, tail)

    staged = staged_from(graph)

    assert staged.raw.has_fork(parent, child1, child2)
    assert not staged.after_g2.has_fork(parent, child1, child2)
    assert not staged.final.has_fork(parent, child1, child2)


def test_assert_candidates_match_g2_passes_on_a_staged_run() -> None:
    graph, _, child1, _ = division_graph()
    tail = add_node(graph, t=2, y=4.0)
    add_edge(graph, child1, tail)

    assert_candidates_match_g2(staged_from(graph))


def test_assert_candidates_match_g2_detects_a_disagreement() -> None:
    graph, _, child1, _ = division_graph()
    tail = add_node(graph, t=2, y=4.0)
    add_edge(graph, child1, tail)

    staged = staged_from(graph)
    staged.after_g2 = StageSnapshot(nodes=staged.after_g2.nodes, edges=frozenset())

    with pytest.raises(AssertionError, match="candidate/G2 disagreement"):
        assert_candidates_match_g2(staged)


def test_staged_pipeline_reports_frozen_application_stats() -> None:
    graph, _, child1, _ = division_graph()
    tail = add_node(graph, t=2, y=4.0)
    add_edge(graph, child1, tail)

    staged = staged_from(graph)

    assert staged.application.dataset == "synthetic"
    assert staged.application.single_added == 0
    assert staged.application.two_requested == 0


def test_staged_pipeline_raises_when_the_single_policy_is_not_addable() -> None:
    graph, *_ = division_graph()

    single = pd.DataFrame([{"dataset": "synthetic", "prefix": "synthetic", "source_id": 999, "target_id": 1000}])

    with pytest.raises(AssertionError, match="not fully addable"):
        staged_from(graph, single=single)


# ============================================================
# classify_gt_division
# ============================================================


def classify(**overrides) -> str:
    kwargs = {
        "final_correct": False,
        "parent_matched": True,
        "daughter1_matched": True,
        "daughter2_matched": True,
        "candidate": None,
        "fork_present_final": False,
    }
    kwargs.update(overrides)
    return classify_gt_division(**kwargs)


def candidate_for(**kwargs):
    graph, parent, _, _ = division_graph(**kwargs)
    return build_division_candidates(graph, UNIT_SCALE)[parent]


def test_perfect_division_is_a_true_positive() -> None:
    assert classify(final_correct=True) == DIVISION_TP


def test_true_positive_wins_over_every_other_signal() -> None:
    assert classify(final_correct=True, parent_matched=False, daughter1_matched=False) == DIVISION_TP


def test_missing_parent() -> None:
    assert classify(parent_matched=False) == PARENT_MISSING


def test_missing_parent_outranks_missing_daughters() -> None:
    assert classify(parent_matched=False, daughter1_matched=False, daughter2_matched=False) == PARENT_MISSING


def test_both_daughters_missing() -> None:
    assert classify(daughter1_matched=False, daughter2_matched=False) == DAUGHTERS_MISSING_2


@pytest.mark.parametrize("missing", ["daughter1_matched", "daughter2_matched"])
def test_one_daughter_missing(missing) -> None:
    assert classify(**{missing: False}) == DAUGHTER_MISSING_1


def test_all_nodes_present_but_association_missing() -> None:
    assert classify(candidate=None) == NODES_PRESENT_EDGES_MISSING


def test_candidate_failing_p2_only() -> None:
    candidate = candidate_for(p2=0.5)

    assert candidate.reject_reason_kind == "p2_only"
    assert classify(candidate=candidate) == CANDIDATE_REJECT_P2


def test_candidate_failing_separation_only() -> None:
    candidate = candidate_for(**FAILS_SEPARATION)

    assert candidate.failed_geometry_flags == ("separation",)
    assert classify(candidate=candidate) == CANDIDATE_REJECT_GEOMETRY


def test_candidate_failing_angle_only() -> None:
    candidate = candidate_for(**FAILS_ANGLE)

    assert candidate.failed_geometry_flags == ("angle",)
    assert candidate.passes_p2
    assert candidate.passes_separation
    assert classify(candidate=candidate) == CANDIDATE_REJECT_GEOMETRY


def test_candidate_failing_max_distance_only() -> None:
    candidate = candidate_for(**FAILS_MAX_DISTANCE)

    assert candidate.failed_geometry_flags == ("max_distance",)
    assert classify(candidate=candidate) == CANDIDATE_REJECT_GEOMETRY


def test_candidate_failing_balance_only() -> None:
    candidate = candidate_for(**FAILS_BALANCE)

    assert candidate.failed_geometry_flags == ("balance",)
    assert candidate.passes_p2
    assert candidate.passes_separation
    assert candidate.passes_distance
    assert classify(candidate=candidate) == CANDIDATE_REJECT_GEOMETRY


def test_geometry_failure_outranks_confidence_failure() -> None:
    candidate = candidate_for(p2=0.5, **FAILS_SEPARATION)

    assert classify(candidate=candidate) == CANDIDATE_REJECT_GEOMETRY
    assert candidate.reject_reason_kind == "p2_and_geometry"


def test_accepted_candidate_destroyed_later() -> None:
    candidate = candidate_for()

    assert candidate.accepted
    assert classify(candidate=candidate, fork_present_final=False) == CANDIDATE_ACCEPTED_LATER_LOST


def test_accepted_candidate_surviving_every_stage_but_unscored_is_other() -> None:
    candidate = candidate_for()

    assert classify(candidate=candidate, fork_present_final=True) == OTHER


def test_classification_is_deterministic() -> None:
    candidate = candidate_for(p2=0.5)

    results = {classify(candidate=candidate) for _ in range(25)}

    assert results == {CANDIDATE_REJECT_P2}


def test_every_classification_maps_to_a_failure_mode() -> None:
    assert failure_mode(DIVISION_TP) == "TP"

    for classification in CLASSIFICATIONS:
        if classification == DIVISION_TP:
            continue
        expected = "DETECTION_LIMITED" if classification in DETECTION_LIMITED else "LINKING_LIMITED"
        assert failure_mode(classification) == expected

    assert DETECTION_LIMITED | LINKING_LIMITED == set(CLASSIFICATIONS) - {DIVISION_TP}
    assert DETECTION_LIMITED.isdisjoint(LINKING_LIMITED)


# ============================================================
# Aggregation
# ============================================================


REJECT_REASON_FOR = {
    DIVISION_TP: "accepted",
    CANDIDATE_REJECT_P2: "p2_only",
    CANDIDATE_REJECT_GEOMETRY: "geometry_only",
    CANDIDATE_ACCEPTED_LATER_LOST: "accepted",
}


def gt_frame(rows: list[tuple[str, str]], *, reject_reasons: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": f"{prefix}_{i:02d}",
                "prefix": prefix,
                "classification": classification,
                "failure_mode": failure_mode(classification),
                "candidate_found": False,
                "reject_reason_kind": (
                    reject_reasons[i] if reject_reasons is not None else REJECT_REASON_FOR.get(classification)
                ),
            }
            for i, (prefix, classification) in enumerate(rows)
        ]
    )


def test_summarize_classifications_covers_every_label_in_order() -> None:
    summary = summarize_classifications(gt_frame([("44b6", DIVISION_TP), ("44b6", PARENT_MISSING)]))

    assert list(summary["classification"]) == list(CLASSIFICATIONS)
    assert summary.set_index("classification").loc[DIVISION_TP, "n"] == 1
    assert summary.set_index("classification").loc[PARENT_MISSING, "n"] == 1
    assert summary.set_index("classification").loc[PARENT_MISSING, "pct_of_gt"] == pytest.approx(50.0)
    assert summary.set_index("classification").loc[PARENT_MISSING, "pct_of_fn"] == pytest.approx(100.0)


def test_prefix_aggregation_splits_counts_by_prefix() -> None:
    gt = gt_frame(
        [
            ("44b6", PARENT_MISSING),
            ("44b6", PARENT_MISSING),
            ("6bba", DIVISION_TP),
            ("6bba", CANDIDATE_REJECT_P2),
        ]
    )

    summary = summarize_gt_by_prefix(gt).set_index(["prefix", "classification"])

    assert summary.loc[("44b6", PARENT_MISSING), "n"] == 2
    assert summary.loc[("44b6", DIVISION_TP), "n"] == 0
    assert summary.loc[("6bba", DIVISION_TP), "n"] == 1
    assert summary.loc[("6bba", CANDIDATE_REJECT_P2), "n"] == 1
    assert summary.loc[("6bba", CANDIDATE_REJECT_P2), "pct_of_fn"] == pytest.approx(100.0)


def test_prefix_aggregation_totals_match_the_overall_summary() -> None:
    gt = gt_frame(
        [
            ("44b6", PARENT_MISSING),
            ("6bba", PARENT_MISSING),
            ("6bba", DIVISION_TP),
        ]
    )

    overall = summarize_classifications(gt).set_index("classification")["n"]
    by_prefix = summarize_gt_by_prefix(gt).groupby("classification")["n"].sum()

    assert overall.to_dict() == by_prefix.reindex(overall.index).to_dict()


def test_fn_bottleneck_summary_separates_detection_from_linking() -> None:
    gt = gt_frame(
        [
            ("6bba", DIVISION_TP),
            ("6bba", PARENT_MISSING),
            ("6bba", DAUGHTER_MISSING_1),
            ("6bba", DAUGHTERS_MISSING_2),
            ("6bba", NODES_PRESENT_EDGES_MISSING),
            ("6bba", CANDIDATE_REJECT_P2),
            ("6bba", CANDIDATE_REJECT_GEOMETRY),
            ("6bba", CANDIDATE_ACCEPTED_LATER_LOST),
        ]
    )

    summary = fn_bottleneck_summary(gt)

    assert summary["n_fn"] == 7
    assert summary["parent_missing"] == 1
    assert summary["daughter_missing"] == 2
    assert summary["all_nodes_present"] == 4
    assert summary["no_candidate_generated"] == 1
    assert summary["rejected_confidence_only"] == 1
    assert summary["rejected_geometry_only"] == 1
    assert summary["rejected_confidence_and_geometry"] == 0
    assert summary["accepted_then_lost"] == 1
    assert summary["detection_limited"] == 3
    assert summary["linking_limited"] == 4
    assert summary["pct_detection_limited"] == pytest.approx(300.0 / 7)


def test_fn_bottleneck_counts_mixed_rejections_separately_from_geometry_only() -> None:
    gt = gt_frame(
        [
            ("6bba", CANDIDATE_REJECT_GEOMETRY),
            ("6bba", CANDIDATE_REJECT_GEOMETRY),
        ],
        reject_reasons=["geometry_only", "p2_and_geometry"],
    )

    summary = fn_bottleneck_summary(gt)

    assert summary["rejected_geometry_only"] == 1
    assert summary["rejected_confidence_and_geometry"] == 1
    assert summary["rejected_confidence_only"] == 0
    assert summary["all_nodes_present"] == 2


def test_fn_bottleneck_summary_handles_a_split_with_no_false_negatives() -> None:
    summary = fn_bottleneck_summary(gt_frame([("6bba", DIVISION_TP)]))

    assert summary["n_fn"] == 0
    assert summary["largest_group"] == ""
    assert np.isnan(summary["pct_detection_limited"])


def test_geometry_distributions_only_describe_candidate_rows() -> None:
    gt = pd.DataFrame(
        [
            {
                "classification": CANDIDATE_REJECT_P2,
                "candidate_found": True,
                "p2": 0.4,
                "daughter_separation_um": 8.0,
                "division_angle_deg": 170.0,
                "max_parent_daughter_um": 4.0,
                "distance_balance": 0.1,
            },
            {
                "classification": CANDIDATE_REJECT_P2,
                "candidate_found": True,
                "p2": 0.6,
                "daughter_separation_um": 10.0,
                "division_angle_deg": 150.0,
                "max_parent_daughter_um": 6.0,
                "distance_balance": 0.3,
            },
            {
                "classification": PARENT_MISSING,
                "candidate_found": False,
                "p2": None,
                "daughter_separation_um": None,
                "division_angle_deg": None,
                "max_parent_daughter_um": None,
                "distance_balance": None,
            },
        ]
    )

    dist = geometry_distributions(gt).set_index("classification")

    assert list(dist.index) == [CANDIDATE_REJECT_P2]
    assert dist.loc[CANDIDATE_REJECT_P2, "n"] == 2
    assert dist.loc[CANDIDATE_REJECT_P2, "p2_min"] == pytest.approx(0.4)
    assert dist.loc[CANDIDATE_REJECT_P2, "p2_max"] == pytest.approx(0.6)
    assert dist.loc[CANDIDATE_REJECT_P2, "division_angle_deg_median"] == pytest.approx(160.0)


def test_summarize_fp_structure_counts_only_metric_false_positives() -> None:
    pred = pd.DataFrame(
        [
            {"counted_as_fp": True, "fp_structure_class": FP_STRUCTURE_CLASSES[0]},
            {"counted_as_fp": True, "fp_structure_class": FP_STRUCTURE_CLASSES[1]},
            {"counted_as_fp": True, "fp_structure_class": FP_STRUCTURE_CLASSES[1]},
            {"counted_as_fp": False, "fp_structure_class": FP_STRUCTURE_CLASSES[2]},
        ]
    )

    summary = summarize_fp_structure(pred).set_index("fp_structure_class")

    assert summary.loc[FP_STRUCTURE_CLASSES[0], "n"] == 1
    assert summary.loc[FP_STRUCTURE_CLASSES[1], "n"] == 2
    assert summary.loc[FP_STRUCTURE_CLASSES[2], "n"] == 0
    assert summary["n"].sum() == 3
    assert summary.loc[FP_STRUCTURE_CLASSES[1], "pct_of_fp"] == pytest.approx(200.0 / 3)


# ============================================================
# audit_dataset — end to end against the official matcher
# ============================================================

FAR_AWAY = 100.0


def lineage_graph(
    graph,
    *,
    p2: float = 0.95,
    divider_y: float = 0.0,
    branch2_y: float = -4.0,
    link_second_daughter: bool = True,
):
    """``root -> divider -> (daughter1, daughter2) -> grandchildren``.

    The shape :func:`tracking_cellmot.division_metrics.extract_divisions`
    expects. Offsets let a test push one node — or one whole branch — out of
    the 7 um matching radius.
    """
    root = add_node(graph, t=0, y=0.0)
    divider = add_node(graph, t=1, y=divider_y)
    daughter1 = add_node(graph, t=2, y=4.0)
    daughter2 = add_node(graph, t=2, y=branch2_y)
    grandchild1 = add_node(graph, t=3, y=5.0)
    grandchild2 = add_node(graph, t=3, y=branch2_y - 1.0)

    add_edge(graph, root, divider, prob=0.99)
    add_edge(graph, divider, daughter1, prob=0.99)
    add_edge(graph, daughter1, grandchild1, prob=0.99)
    add_edge(graph, daughter2, grandchild2, prob=0.99)

    if link_second_daughter:
        add_edge(graph, divider, daughter2, prob=p2)

    return {
        "root": root,
        "divider": divider,
        "daughter1": daughter1,
        "daughter2": daughter2,
    }


def gt_lineage():
    graph = new_graph()
    lineage_graph(graph)
    return graph


def audit_of(pred_graph):
    staged = staged_from(pred_graph)
    assert_candidates_match_g2(staged)
    return staged, audit_dataset(staged, gt_lineage(), prefix="synthetic")


def only_gt_row(audit) -> dict:
    rows = audit.gt_rows
    assert len(rows) == 1
    return rows[0]


def test_audit_scores_a_perfect_division_as_a_true_positive() -> None:
    pred = new_graph()
    lineage_graph(pred)

    _, audit = audit_of(pred)
    row = only_gt_row(audit)

    assert (audit.division_tp, audit.division_fp, audit.division_fn) == (1, 0, 0)
    assert row["classification"] == DIVISION_TP
    assert row["failure_mode"] == "TP"
    assert row["final_division_correct"]


def test_audit_reports_an_accepted_candidate_surviving_every_stage() -> None:
    pred = new_graph()
    lineage_graph(pred)

    _, audit = audit_of(pred)
    row = only_gt_row(audit)

    assert row["accepted_by_g2"]
    assert row["survives_k9"]
    assert row["survives_recovery"]
    assert row["survives_cleanup"]
    assert row["final_fork_present"]
    assert row["candidate_anchor"] == "parent"


def test_audit_classifies_a_p2_rejection_from_the_real_graph() -> None:
    pred = new_graph()
    lineage_graph(pred, p2=0.5)

    _, audit = audit_of(pred)
    row = only_gt_row(audit)

    assert audit.division_fn == 1
    assert row["classification"] == CANDIDATE_REJECT_P2
    assert row["failure_mode"] == "LINKING_LIMITED"
    assert row["candidate_found"]
    assert row["passes_p2"] is False
    assert row["accepted_by_g2"] is False
    assert row["survives_k9"] is False


def test_audit_classifies_an_unmatched_parent() -> None:
    pred = new_graph()
    lineage_graph(pred, divider_y=FAR_AWAY)

    _, audit = audit_of(pred)
    row = only_gt_row(audit)

    assert row["classification"] == PARENT_MISSING
    assert row["failure_mode"] == "DETECTION_LIMITED"
    assert row["parent_matched"] is False


def test_audit_classifies_one_unmatched_daughter_branch() -> None:
    pred = new_graph()
    lineage_graph(pred, branch2_y=-FAR_AWAY)

    _, audit = audit_of(pred)
    row = only_gt_row(audit)

    assert row["classification"] == DAUGHTER_MISSING_1
    assert row["failure_mode"] == "DETECTION_LIMITED"
    assert row["daughter1_matched"] is True
    assert row["daughter2_matched"] is False


def test_audit_classifies_present_nodes_with_no_predicted_association() -> None:
    pred = new_graph()
    lineage_graph(pred, link_second_daughter=False)

    _, audit = audit_of(pred)
    row = only_gt_row(audit)

    assert row["classification"] == NODES_PRESENT_EDGES_MISSING
    assert row["failure_mode"] == "LINKING_LIMITED"
    assert row["parent_matched"] and row["daughter1_matched"] and row["daughter2_matched"]
    assert row["candidate_found"] is False
    assert row["raw_parent_out_degree"] == 1
    assert row["raw_parent_has_both_daughters"] is False


def test_audit_emits_one_row_per_predicted_fork_with_geometry() -> None:
    pred = new_graph()
    nodes = lineage_graph(pred)

    _, audit = audit_of(pred)

    assert len(audit.pred_rows) == 1
    row = audit.pred_rows[0]

    assert row["parent_node"] == nodes["divider"]
    assert row["daughter_count"] == 2
    assert row["counted_as_tp"] is True
    assert row["counted_as_fp"] is False
    assert row["passes_geometry"] is True
    assert row["daughter_separation_um"] == pytest.approx(8.0)
    assert row["division_angle_deg"] == pytest.approx(180.0)
    assert row["origin_g2_accepted"] is True
    assert row["origin_includes_recovered_edge"] is False


def test_audit_is_deterministic_across_repeated_runs() -> None:
    classifications = []

    for _ in range(3):
        pred = new_graph()
        lineage_graph(pred, p2=0.5)
        _, audit = audit_of(pred)
        classifications.append(only_gt_row(audit)["classification"])

    assert set(classifications) == {CANDIDATE_REJECT_P2}


# ============================================================
# Association-gap kinds
# ============================================================


@pytest.mark.parametrize(
    ("raw_out_degree", "daughters_proposed", "expected"),
    [
        (2, 2, "both_proposed"),
        (3, 2, "both_proposed"),
        (0, 0, "no_outgoing_edge"),
        (1, 1, "one_of_two_proposed"),
        (2, 1, "one_of_two_proposed"),
        (1, 0, "neither_proposed"),
        (2, 0, "neither_proposed"),
    ],
)
def test_association_gap_kind_names_the_missing_association(raw_out_degree, daughters_proposed, expected) -> None:
    assert _association_gap_kind(raw_out_degree, daughters_proposed) == expected
    assert expected in ASSOCIATION_GAP_KINDS


def test_audit_records_the_association_gap_for_a_missing_second_edge() -> None:
    pred = new_graph()
    lineage_graph(pred, link_second_daughter=False)

    _, audit = audit_of(pred)
    row = only_gt_row(audit)

    assert row["classification"] == NODES_PRESENT_EDGES_MISSING
    assert row["raw_parent_daughters_proposed"] == 1
    assert row["association_gap_kind"] == "one_of_two_proposed"


# ============================================================
# False-positive structural classification
# ============================================================


def test_fp_classification_treats_an_annotated_parent_as_gt_contradiction() -> None:
    # GT tracks the parent but only one daughter branch: GT states this cell
    # does not divide, so the predicted fork contradicts GT rather than filling
    # an annotation gap.
    gt = new_graph()
    lineage_graph(gt, link_second_daughter=False)

    pred = new_graph()
    lineage_graph(pred)
    staged = staged_from(pred)

    row = audit_dataset(staged, gt, prefix="synthetic").pred_rows[0]

    assert row["gt_parent_out_degree"] == 1
    assert row["gt_contradicts_division"] is True
    assert row["gt_evidence_absent"] is False
    assert row["sparse_gt_plausible"] is False
    assert row["passes_geometry"] is True
    assert row["fp_structure_class"] == FP_STRUCTURE_CLASSES[2]


def test_fp_classification_calls_an_untracked_parent_plausibly_unlabeled() -> None:
    # GT never tracks the dividing cell, so nothing in GT contradicts the fork.
    gt = new_graph()
    add_node(gt, t=0, y=FAR_AWAY)
    add_node(gt, t=1, y=FAR_AWAY)
    add_edge(gt, 0, 1, prob=0.99)

    pred = new_graph()
    lineage_graph(pred)
    staged = staged_from(pred)

    row = audit_dataset(staged, gt, prefix="synthetic").pred_rows[0]

    assert row["parent_matches_gt"] is False
    assert row["gt_evidence_absent"] is True
    assert row["sparse_gt_plausible"] is True
    assert row["fp_structure_class"] == FP_STRUCTURE_CLASSES[1]
    assert row["counted_as_fp"] is False


def test_fp_classification_flags_an_out_degree_three_fork_as_implausible() -> None:
    pred = new_graph()
    nodes = lineage_graph(pred)
    third = add_node(pred, t=2, y=0.0, x=4.0)
    add_edge(pred, nodes["divider"], third, prob=0.9)

    staged = staged_from(pred)
    row = audit_dataset(staged, gt_lineage(), prefix="synthetic").pred_rows[0]

    assert row["daughter_count"] == 3
    assert row["fp_structure_class"] == FP_STRUCTURE_CLASSES[0]


def test_summarize_pred_fork_scope_partitions_every_fork() -> None:
    pred = pd.DataFrame(
        [
            {
                "counted_as_tp": True,
                "counted_as_fp": False,
                "evaluated_by_metric": True,
                "parent_matches_gt": True,
                "gt_evidence_absent": False,
                "gt_contradicts_division": False,
                "passes_geometry": True,
            },
            {
                "counted_as_tp": False,
                "counted_as_fp": True,
                "evaluated_by_metric": True,
                "parent_matches_gt": True,
                "gt_evidence_absent": False,
                "gt_contradicts_division": True,
                "passes_geometry": True,
            },
            {
                "counted_as_tp": False,
                "counted_as_fp": False,
                "evaluated_by_metric": False,
                "parent_matches_gt": False,
                "gt_evidence_absent": True,
                "gt_contradicts_division": False,
                "passes_geometry": True,
            },
        ]
    )

    scope = summarize_pred_fork_scope(pred).set_index("scope")

    assert scope.loc["division_tp", "n"] == 1
    assert scope.loc["division_fp", "n"] == 1
    assert scope.loc["division_fp", "gt_contradicts_division"] == 1
    assert scope.loc["not_evaluated", "n"] == 1
    assert scope.loc["not_evaluated", "gt_evidence_absent"] == 1
    assert scope["n"].sum() == len(pred)
