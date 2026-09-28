"""Focused tests for src/biohub_cell_tracking/association_audit.py.

Small deterministic in-memory graphs only. No checkpoint loads, no model
inference, no dataset reads. The Fold0 audit lives in
scripts/audit_stage11_association.py.
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

from biohub_cell_tracking.association_audit import (
    ASSOCIATION_DATAFLOW,
    BLOCKER_ASSOCIATION_ONLY,
    BLOCKER_G2_GEOMETRY,
    CASE_STRUCTURES,
    DAUGHTER_CLAIMED_BY_COMPETITOR,
    DAUGHTER_UNCLAIMED_PARENT_AT_CAP,
    DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP,
    EDGE_PRESENT,
    EDGE_STATUS_MEANING,
    EDGE_STATUSES,
    FROZEN_PREDICT_CONFIG,
    ROOT_CAUSE_HYPOTHESES,
    EdgeRecord,
    build_case_row,
    build_case_table,
    case_structure,
    classify_missing_edge,
    competing_incoming_edges,
    competing_outgoing_edges,
    daughter_incoming_edges,
    graph_view,
    gt_fork_geometry,
    parent_outgoing_edges,
    restored_fork_geometry,
    summarize_cases,
)

UNIT_SCALE = (1.0, 1.0, 1.0)


# ============================================================
# Graph fixtures
# ============================================================


def new_graph(*, with_edge_attrs: bool = True):
    """Empty graph shaped like a prediction GEFF (or a GT GEFF without attrs)."""
    graph = td.graph.IndexedRXGraph()

    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)

    if with_edge_attrs:
        for key in ("edge_prob", "edge_dist"):
            graph.add_edge_attr_key(key, pl.Float64, 0.0)

    return graph


def add_node(graph, t: int, z: float = 0.0, y: float = 0.0, x: float = 0.0) -> int:
    return graph.add_node({"t": int(t), "z": float(z), "y": float(y), "x": float(x)})


def add_edge(graph, source: int, target: int, prob: float = 0.9, dist: float = 1.0) -> int:
    return graph.add_edge(source, target, {"edge_prob": float(prob), "edge_dist": float(dist)})


def snapshot(graph) -> tuple[set, set]:
    return (
        {int(n) for n in graph.node_ids()},
        {(int(s), int(t)) for s, t in graph.edge_list()},
    )


def division_prediction(
    *,
    link_second: bool = True,
    competitor_claims_second: bool = False,
    extra_children: int = 0,
):
    """Parent at the origin, two daughters at y = +/-4 in the next frame.

    Options reproduce each way the frozen predictor can lose the second edge.
    """
    graph = new_graph()

    parent = add_node(graph, t=0, y=0.0)
    daughter1 = add_node(graph, t=1, y=4.0)
    daughter2 = add_node(graph, t=1, y=-4.0)

    add_edge(graph, parent, daughter1, prob=0.90)

    if link_second:
        add_edge(graph, parent, daughter2, prob=0.85)

    if competitor_claims_second:
        competitor = add_node(graph, t=0, y=-9.0)
        add_edge(graph, competitor, daughter2, prob=0.72)

    for i in range(extra_children):
        other = add_node(graph, t=1, y=0.0, x=float(3 + i))
        add_edge(graph, parent, other, prob=0.95)

    return graph, parent, daughter1, daughter2


def view_of(graph, scale=UNIT_SCALE):
    return graph_view(graph, "synthetic", scale)


# ============================================================
# Frozen constants
# ============================================================


def test_frozen_predict_config_records_the_locked_association_policy() -> None:
    assert FROZEN_PREDICT_CONFIG["edge_activation"] == "softmax"
    assert FROZEN_PREDICT_CONFIG["softmax_dim"] == 0
    assert FROZEN_PREDICT_CONFIG["threshold"] == 0.5
    assert FROZEN_PREDICT_CONFIG["max_parents_per_node"] == 1
    assert FROZEN_PREDICT_CONFIG["max_children_per_node"] == 2
    assert FROZEN_PREDICT_CONFIG["use_ilp"] is False


def test_dataflow_documents_every_stage_with_a_source_anchor() -> None:
    stages = ASSOCIATION_DATAFLOW["stages"]

    names = [s["stage"] for s in stages]
    assert names[0] == "image window"
    assert names[-1] == "GEFF serialisation"

    for stage in stages:
        assert stage["source"]
        assert ":" in stage["source"]
        assert stage["detail"]

    answers = ASSOCIATION_DATAFLOW["answers"]
    assert len(answers) == 16
    assert "None" in answers["6_spatial_gating"]
    assert "None" in answers["7_top_k"]


def test_every_edge_status_has_a_documented_meaning() -> None:
    assert set(EDGE_STATUS_MEANING) == set(EDGE_STATUSES)

    for status, meaning in EDGE_STATUS_MEANING.items():
        assert isinstance(meaning["determinable_from_geff"], bool)
        assert meaning["implication"]
        cause = meaning["root_cause"]
        assert cause is None or cause in ROOT_CAUSE_HYPOTHESES, status


# ============================================================
# Edge extraction
# ============================================================


def test_parent_outgoing_edges_are_returned_highest_probability_first() -> None:
    graph, parent, daughter1, daughter2 = division_prediction()
    view = view_of(graph)

    edges = parent_outgoing_edges(view, parent)

    assert [e.target for e in edges] == [daughter1, daughter2]
    assert [e.edge_prob for e in edges] == pytest.approx([0.90, 0.85])
    assert all(isinstance(e, EdgeRecord) for e in edges)


def test_parent_outgoing_edges_is_empty_for_a_leaf() -> None:
    graph, _, daughter1, _ = division_prediction()

    assert parent_outgoing_edges(view_of(graph), daughter1) == ()


def test_daughter_incoming_edges_reports_every_claiming_source() -> None:
    graph, parent, _, daughter2 = division_prediction(link_second=False, competitor_claims_second=True)
    view = view_of(graph)

    edges = daughter_incoming_edges(view, daughter2)

    assert len(edges) == 1
    assert edges[0].source != parent
    assert edges[0].edge_prob == pytest.approx(0.72)


def test_daughter_incoming_edges_is_empty_when_no_source_claimed_the_column() -> None:
    graph, _, _, daughter2 = division_prediction(link_second=False)

    assert daughter_incoming_edges(view_of(graph), daughter2) == ()


def test_competing_edge_detection_excludes_the_required_endpoints() -> None:
    graph, parent, daughter1, daughter2 = division_prediction(
        link_second=False, competitor_claims_second=True, extra_children=1
    )
    view = view_of(graph)

    competing_out = competing_outgoing_edges(view, parent, {daughter1, daughter2})
    assert len(competing_out) == 1
    assert competing_out[0].target not in {daughter1, daughter2}

    competing_in = competing_incoming_edges(view, daughter2, parent)
    assert len(competing_in) == 1
    assert competing_in[0].source != parent

    assert competing_incoming_edges(view, daughter1, parent) == ()


def test_view_degrees_and_distances_use_physical_scale() -> None:
    graph, parent, daughter1, daughter2 = division_prediction()

    isotropic = view_of(graph)
    assert isotropic.out_degree(parent) == 2
    assert isotropic.in_degree(daughter1) == 1
    assert isotropic.distance_um(parent, daughter1) == pytest.approx(4.0)

    anisotropic = view_of(graph, scale=(1.0, 2.0, 1.0))
    assert anisotropic.distance_um(parent, daughter1) == pytest.approx(8.0)
    assert anisotropic.distance_um(daughter1, daughter2) == pytest.approx(16.0)


def test_graph_view_reads_a_graph_without_association_attributes() -> None:
    graph = new_graph(with_edge_attrs=False)
    parent = add_node(graph, t=0, y=0.0)
    child = add_node(graph, t=1, y=4.0)
    graph.add_edge(parent, child, {})

    view = view_of(graph)

    assert view.out_degree(parent) == 1
    assert np.isnan(view.out_edges[parent][0].edge_prob)


# ============================================================
# Classification
# ============================================================


def test_present_edge_is_classified_as_present() -> None:
    graph, parent, daughter1, daughter2 = division_prediction()
    view = view_of(graph)

    assert classify_missing_edge(view, parent, daughter1) == EDGE_PRESENT
    assert classify_missing_edge(view, parent, daughter2) == EDGE_PRESENT


def test_competitor_claim_is_classified_as_column_lost() -> None:
    graph, parent, _, daughter2 = division_prediction(link_second=False, competitor_claims_second=True)

    status = classify_missing_edge(view_of(graph), parent, daughter2)

    assert status == DAUGHTER_CLAIMED_BY_COMPETITOR
    assert EDGE_STATUS_MEANING[status]["root_cause"] == "MODEL_SCORE_LIMITED"
    assert EDGE_STATUS_MEANING[status]["determinable_from_geff"] is True


def test_unclaimed_daughter_below_the_child_cap_is_threshold_suppression() -> None:
    graph, parent, _, daughter2 = division_prediction(link_second=False)
    view = view_of(graph)

    assert view.out_degree(parent) < FROZEN_PREDICT_CONFIG["max_children_per_node"]

    status = classify_missing_edge(view, parent, daughter2)

    assert status == DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP
    assert EDGE_STATUS_MEANING[status]["root_cause"] == "EDGE_THRESHOLD_SUPPRESSION"
    assert EDGE_STATUS_MEANING[status]["determinable_from_geff"] is True


def test_unclaimed_daughter_at_the_child_cap_is_ambiguous() -> None:
    graph, parent, _, daughter2 = division_prediction(link_second=False, extra_children=1)
    view = view_of(graph)

    assert view.out_degree(parent) == FROZEN_PREDICT_CONFIG["max_children_per_node"]

    status = classify_missing_edge(view, parent, daughter2)

    assert status == DAUGHTER_UNCLAIMED_PARENT_AT_CAP
    assert EDGE_STATUS_MEANING[status]["determinable_from_geff"] is False


def test_competitor_claim_outranks_the_child_cap() -> None:
    graph, parent, _, daughter2 = division_prediction(
        link_second=False, competitor_claims_second=True, extra_children=1
    )

    assert classify_missing_edge(view_of(graph), parent, daughter2) == DAUGHTER_CLAIMED_BY_COMPETITOR


@pytest.mark.parametrize(
    ("status1", "status2", "expected"),
    [
        (EDGE_PRESENT, EDGE_PRESENT, "BOTH_EDGES_PRESENT"),
        (EDGE_PRESENT, DAUGHTER_CLAIMED_BY_COMPETITOR, "ONE_EDGE_PRESENT"),
        (DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP, EDGE_PRESENT, "ONE_EDGE_PRESENT"),
        (DAUGHTER_CLAIMED_BY_COMPETITOR, DAUGHTER_UNCLAIMED_PARENT_AT_CAP, "ZERO_REQUIRED_EDGES"),
        (
            DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP,
            DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP,
            "ZERO_REQUIRED_EDGES",
        ),
    ],
)
def test_case_structure_classification(status1, status2, expected) -> None:
    assert case_structure(status1, status2) == expected
    assert expected in CASE_STRUCTURES


# ============================================================
# Restored / GT fork geometry
# ============================================================


def test_restored_fork_geometry_accepts_a_clean_symmetric_division() -> None:
    graph, parent, daughter1, daughter2 = division_prediction(link_second=False)

    geometry = restored_fork_geometry(view_of(graph), parent, daughter1, daughter2)

    assert geometry["restored_daughter_separation_um"] == pytest.approx(8.0)
    assert geometry["restored_division_angle_deg"] == pytest.approx(180.0)
    assert geometry["restored_max_parent_daughter_um"] == pytest.approx(4.0)
    assert geometry["restored_distance_balance"] == pytest.approx(0.0)
    assert geometry["restored_passes_g2_geometry"] is True


def test_restored_fork_geometry_rejects_an_over_long_parent_daughter_step() -> None:
    graph = new_graph()
    parent = add_node(graph, t=0, y=0.0)
    daughter1 = add_node(graph, t=1, y=9.0)
    daughter2 = add_node(graph, t=1, y=-9.0)
    add_edge(graph, parent, daughter1, prob=0.9)

    geometry = restored_fork_geometry(view_of(graph), parent, daughter1, daughter2)

    assert geometry["restored_passes_distance"] is False
    assert geometry["restored_passes_g2_geometry"] is False


def test_restored_fork_geometry_is_symmetric_in_the_two_daughters() -> None:
    graph, parent, daughter1, daughter2 = division_prediction()
    view = view_of(graph)

    forward = restored_fork_geometry(view, parent, daughter1, daughter2)
    reversed_ = restored_fork_geometry(view, parent, daughter2, daughter1)

    assert forward == reversed_


def test_gt_fork_geometry_flags_a_division_the_frozen_policy_cannot_accept() -> None:
    gt = new_graph(with_edge_attrs=False)
    parent = add_node(gt, t=0, y=0.0)
    # Both daughters on the same side: a real division the 120 deg gate rejects.
    daughter1 = add_node(gt, t=1, y=4.0, x=4.0)
    daughter2 = add_node(gt, t=1, y=4.0, x=-4.0)

    geometry = gt_fork_geometry(view_of(gt), parent, daughter1, daughter2)

    assert geometry["gt_passes_separation"] is True
    assert geometry["gt_passes_angle"] is False
    assert geometry["gt_passes_g2_geometry"] is False


def test_gt_fork_geometry_returns_nan_for_absent_nodes() -> None:
    gt = new_graph(with_edge_attrs=False)
    parent = add_node(gt, t=0, y=0.0)

    geometry = gt_fork_geometry(view_of(gt), parent, 999, 1000)

    assert np.isnan(geometry["gt_max_parent_daughter_um"])
    assert geometry["gt_passes_g2_geometry"] is False


# ============================================================
# Case rows and tables
# ============================================================


def case_frame(dataset: str, parent: int, daughter1: int, daughter2: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "prefix": "6bba",
                "gt_parent": 1,
                "gt_daughter_1": 2,
                "gt_daughter_2": 3,
                "pred_parent": parent,
                "pred_daughter_1": daughter1,
                "pred_daughter_2": daughter2,
            }
        ]
    )


def test_case_row_records_both_edge_statuses_and_the_blocker() -> None:
    graph, parent, daughter1, daughter2 = division_prediction(link_second=False, competitor_claims_second=True)

    row = build_case_row(
        view_of(graph),
        dataset="synthetic",
        prefix="6bba",
        gt_parent=1,
        gt_daughter_1=2,
        gt_daughter_2=3,
        parent=parent,
        daughter1=daughter1,
        daughter2=daughter2,
    )

    assert row["edge_1_status"] == EDGE_PRESENT
    assert row["edge_2_status"] == DAUGHTER_CLAIMED_BY_COMPETITOR
    assert row["case_structure"] == "ONE_EDGE_PRESENT"
    assert row["edge_1_prob"] == pytest.approx(0.90)
    assert row["edge_2_prob"] is None
    assert row["matching_suppression_indicated"] is True
    assert row["degree_cap_possible"] is False
    assert row["root_cause_determinable_from_geff"] is True
    assert row["root_causes"] == "MODEL_SCORE_LIMITED"
    assert row["recovery_blocker"] == BLOCKER_ASSOCIATION_ONLY
    assert row["daughter_1_frame_gap"] == 1
    assert row["competitor_2_to_daughter_um"] == pytest.approx(5.0)


def test_case_row_marks_a_geometry_blocked_division() -> None:
    graph = new_graph()
    parent = add_node(graph, t=0, y=0.0)
    daughter1 = add_node(graph, t=1, y=9.0)
    daughter2 = add_node(graph, t=1, y=-9.0)
    add_edge(graph, parent, daughter1, prob=0.9)

    row = build_case_row(
        view_of(graph),
        dataset="synthetic",
        prefix="6bba",
        gt_parent=1,
        gt_daughter_1=2,
        gt_daughter_2=3,
        parent=parent,
        daughter1=daughter1,
        daughter2=daughter2,
    )

    assert row["recovery_blocker"] == BLOCKER_G2_GEOMETRY
    assert row["restored_passes_g2_geometry"] is False


def test_case_table_is_deterministic_and_input_ordered() -> None:
    graph, parent, daughter1, daughter2 = division_prediction(link_second=False)
    views = {"synthetic": view_of(graph)}
    cases = case_frame("synthetic", parent, daughter1, daughter2)

    tables = [build_case_table(cases, views) for _ in range(3)]

    for table in tables[1:]:
        pd.testing.assert_frame_equal(tables[0], table)

    assert list(tables[0]["dataset"]) == ["synthetic"]
    assert tables[0].loc[0, "edge_2_status"] == DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP


def test_case_table_attaches_gt_geometry_when_a_gt_view_is_supplied() -> None:
    graph, parent, daughter1, daughter2 = division_prediction(link_second=False)

    gt = new_graph(with_edge_attrs=False)
    add_node(gt, t=0, y=0.0)
    gt_parent = add_node(gt, t=0, y=0.0)
    gt_d1 = add_node(gt, t=1, y=4.0)
    gt_d2 = add_node(gt, t=1, y=-4.0)

    cases = case_frame("synthetic", parent, daughter1, daughter2)
    cases.loc[0, ["gt_parent", "gt_daughter_1", "gt_daughter_2"]] = [gt_parent, gt_d1, gt_d2]

    table = build_case_table(cases, {"synthetic": view_of(graph)}, gt_views={"synthetic": view_of(gt)})

    assert bool(table.loc[0, "gt_passes_g2_geometry"]) is True
    assert bool(table.loc[0, "association_only_and_gt_g2_compatible"]) is True


def test_building_the_case_table_does_not_mutate_the_graph() -> None:
    graph, parent, daughter1, daughter2 = division_prediction(link_second=False, competitor_claims_second=True)

    before = snapshot(graph)

    view = view_of(graph)
    build_case_table(case_frame("synthetic", parent, daughter1, daughter2), {"synthetic": view})

    assert snapshot(graph) == before
    assert "edge_prob" in graph.edge_attr_keys()


def test_summarize_cases_counts_statuses_structures_and_blockers() -> None:
    graph_a, parent_a, d1_a, d2_a = division_prediction(link_second=False, competitor_claims_second=True)
    graph_b, parent_b, d1_b, d2_b = division_prediction(link_second=False)

    cases = pd.concat(
        [case_frame("a", parent_a, d1_a, d2_a), case_frame("b", parent_b, d1_b, d2_b)],
        ignore_index=True,
    )
    views = {"a": view_of(graph_a), "b": view_of(graph_b)}

    summary = summarize_cases(build_case_table(cases, views))

    assert summary["n_cases"] == 2
    assert summary["edge_status_counts"][EDGE_PRESENT] == 2
    assert summary["edge_status_counts"][DAUGHTER_CLAIMED_BY_COMPETITOR] == 1
    assert summary["edge_status_counts"][DAUGHTER_UNCLAIMED_PARENT_BELOW_CAP] == 1
    assert summary["missing_edges"] == 2
    assert summary["case_structure_counts"]["ONE_EDGE_PRESENT"] == 2
    assert summary["cases_determinable_from_geff"] == 2
    assert summary["cases_requiring_raw_scores"] == 0
    assert summary["root_cause_counts"]["MODEL_SCORE_LIMITED"] == 1
    assert summary["root_cause_counts"]["EDGE_THRESHOLD_SUPPRESSION"] == 1
    assert summary["recovery_blocker_counts"][BLOCKER_ASSOCIATION_ONLY] == 2
    assert summary["prefix_counts"] == {"6bba": 2}


def test_summarize_cases_flags_the_ambiguous_child_cap_case() -> None:
    graph, parent, daughter1, daughter2 = division_prediction(link_second=False, extra_children=1)
    cases = case_frame("a", parent, daughter1, daughter2)

    summary = summarize_cases(build_case_table(cases, {"a": view_of(graph)}))

    assert summary["cases_requiring_raw_scores"] == 1
    assert summary["cases_with_possible_degree_cap"] == 1
