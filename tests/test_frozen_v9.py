"""Focused tests for src/biohub_cell_tracking/frozen_v9.py.

Small deterministic in-memory graphs only. No dataset loads, no training, no
inference, no GT evaluation. The full-dataset reproduction lives in
scripts/verify_frozen_v9.py.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import tracksdata as td

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub_cell_tracking import frozen_v9
from biohub_cell_tracking.frozen_v9 import (
    FROZEN_V9_CONFIG,
    FrozenV9Config,
    FrozenV9Evaluation,
    FrozenV9Policies,
    add_available_repairs,
    add_two_edge_bridges,
    apply_frozen_component_cleanup,
    apply_frozen_g2,
    apply_frozen_v9,
    build_mutual_policy,
    build_strict_two_edge_policy,
    load_graph,
    summarize_by_prefix,
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


def node_count(graph) -> int:
    return len(graph.node_attrs(unpack=True).to_pandas())


def edge_pairs(graph) -> set[tuple[int, int]]:
    return {(int(s), int(t)) for s, t in graph.edge_list()}


# ============================================================
# FrozenV9Config
# ============================================================


def test_config_is_frozen() -> None:
    config = FrozenV9Config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.division_p2_min = 0.5


def test_config_defaults_match_frozen_v9_policy() -> None:
    config = FROZEN_V9_CONFIG

    # Detection
    assert config.det_threshold == 0.995
    assert config.high_threshold == 0.9975

    # Confidence pruning
    assert config.confidence_component_limit == 9

    # Division (G2)
    assert config.division_p2_min == 0.80
    assert config.division_daughter_sep_min_um == 6.0
    assert config.division_angle_min_deg == 120.0
    assert config.division_max_parent_daughter_um == 8.0
    assert config.division_distance_balance_max == 0.50

    # Recovery geometry, shared by single-edge and strict two-edge
    assert config.recovery_residual_max_um == 4.0
    assert config.recovery_max_step_um == 8.0
    assert config.recovery_step_balance_max == 0.50
    assert config.recovery_turn_angle_max_deg == 60.0
    assert config.recovery_ambiguity_margin_min_um == 1.0

    # Strict two-edge uniqueness
    assert config.two_edge_required_pair_count == 1

    # Cleanup
    assert config.cleanup_two_node_max_um == 3.5

    # Structural / metric
    assert config.model_downsample_zyx == (1.0, 4.0, 4.0)
    assert config.metric_max_distance == 7.0


def test_module_constant_is_default_config() -> None:
    assert FROZEN_V9_CONFIG == FrozenV9Config()


# ============================================================
# load_graph normalization
# ============================================================


def test_load_graph_unwraps_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        td.graph.IndexedRXGraph,
        "from_geff",
        staticmethod(lambda path: (sentinel, {"metadata": True})),
    )

    assert load_graph(Path("whatever.geff")) is sentinel


def test_load_graph_passes_through_plain_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        td.graph.IndexedRXGraph,
        "from_geff",
        staticmethod(lambda path: sentinel),
    )

    assert load_graph(Path("whatever.geff")) is sentinel


# ============================================================
# G2 division policy
# ============================================================


def _fork_graph(sep: float, p2: float) -> tuple[object, int, int, int]:
    """Parent at origin with two daughters separated symmetrically along y."""
    graph = new_graph()
    parent = add_node(graph, 0, 0.0, 0.0, 0.0)
    child_a = add_node(graph, 1, 0.0, sep / 2.0, 0.0)
    child_b = add_node(graph, 1, 0.0, -sep / 2.0, 0.0)
    add_edge(graph, parent, child_a, prob=0.99)
    add_edge(graph, parent, child_b, prob=p2)
    return graph, parent, child_a, child_b


def test_g2_keeps_second_daughter_when_geometry_passes() -> None:
    graph, parent, child_a, child_b = _fork_graph(sep=6.0, p2=0.9)

    kept = apply_frozen_g2(graph, UNIT_SCALE)

    assert kept == 1
    assert edge_pairs(graph) == {(parent, child_a), (parent, child_b)}


def test_g2_drops_second_daughter_below_p2_threshold() -> None:
    graph, parent, child_a, _child_b = _fork_graph(sep=6.0, p2=0.6)

    kept = apply_frozen_g2(graph, UNIT_SCALE)

    assert kept == 0
    assert edge_pairs(graph) == {(parent, child_a)}


def test_g2_drops_second_daughter_when_separation_too_small() -> None:
    graph, parent, child_a, _child_b = _fork_graph(sep=2.0, p2=0.99)

    kept = apply_frozen_g2(graph, UNIT_SCALE)

    assert kept == 0
    assert edge_pairs(graph) == {(parent, child_a)}


def test_g2_never_adds_nodes() -> None:
    graph, *_ = _fork_graph(sep=2.0, p2=0.99)
    before = node_count(graph)

    apply_frozen_g2(graph, UNIT_SCALE)

    assert node_count(graph) == before


# ============================================================
# Single-edge recovery
# ============================================================


def _single_policy_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["dataset", "prefix", "source_id", "target_id"])


def test_single_repair_adds_edge_between_free_endpoints() -> None:
    graph = new_graph()
    a = add_node(graph, 0)
    b = add_node(graph, 1, y=1.0)

    stats = add_available_repairs(
        graph,
        _single_policy_frame(
            [
                {"dataset": "d1", "prefix": "d", "source_id": a, "target_id": b},
            ]
        ),
    )

    assert stats["added"] == 1
    assert edge_pairs(graph) == {(a, b)}


def test_single_repair_rejects_degree_conflict() -> None:
    graph = new_graph()
    a = add_node(graph, 0)
    b = add_node(graph, 1, y=1.0)
    c = add_node(graph, 1, y=2.0)
    add_edge(graph, a, c)  # a already has an outgoing edge

    stats = add_available_repairs(
        graph,
        _single_policy_frame(
            [
                {"dataset": "d1", "prefix": "d", "source_id": a, "target_id": b},
            ]
        ),
    )

    assert stats["added"] == 0
    assert stats["degree_conflict"] == 1
    assert edge_pairs(graph) == {(a, c)}


def test_single_repair_rejects_missing_endpoint() -> None:
    graph = new_graph()
    a = add_node(graph, 0)

    stats = add_available_repairs(
        graph,
        _single_policy_frame(
            [
                {"dataset": "d1", "prefix": "d", "source_id": a, "target_id": 999},
            ]
        ),
    )

    assert stats["added"] == 0
    assert stats["missing_endpoint"] == 1


def test_single_repair_skips_already_present_edge() -> None:
    graph = new_graph()
    a = add_node(graph, 0)
    b = add_node(graph, 1, y=1.0)
    add_edge(graph, a, b)

    stats = add_available_repairs(
        graph,
        _single_policy_frame(
            [
                {"dataset": "d1", "prefix": "d", "source_id": a, "target_id": b},
            ]
        ),
    )

    assert stats["added"] == 0
    assert stats["already_present"] == 1


def test_single_repair_order_first_row_wins_on_shared_source() -> None:
    """Two candidates share a source; the first consumes it, the second conflicts."""
    graph = new_graph()
    a = add_node(graph, 0)
    b1 = add_node(graph, 1, y=1.0)
    b2 = add_node(graph, 1, y=2.0)

    stats = add_available_repairs(
        graph,
        _single_policy_frame(
            [
                {"dataset": "d1", "prefix": "d", "source_id": a, "target_id": b1},
                {"dataset": "d1", "prefix": "d", "source_id": a, "target_id": b2},
            ]
        ),
    )

    assert stats["added"] == 1
    assert stats["degree_conflict"] == 1
    assert edge_pairs(graph) == {(a, b1)}


def test_single_repair_adds_no_nodes() -> None:
    graph = new_graph()
    a = add_node(graph, 0)
    b = add_node(graph, 1, y=1.0)
    before = node_count(graph)

    add_available_repairs(
        graph,
        _single_policy_frame(
            [
                {"dataset": "d1", "prefix": "d", "source_id": a, "target_id": b},
            ]
        ),
    )

    assert node_count(graph) == before


# ============================================================
# Strict two-edge recovery
# ============================================================


def _two_edge_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["dataset", "prefix", "a_id", "b_id", "c_id"])


def _abc_graph():
    graph = new_graph()
    a = add_node(graph, 0, y=0.0)
    b = add_node(graph, 1, y=1.0)
    c = add_node(graph, 2, y=2.0)
    return graph, a, b, c


def test_two_edge_bridge_adds_both_edges() -> None:
    graph, a, b, c = _abc_graph()

    stats = add_two_edge_bridges(
        graph,
        _two_edge_frame(
            [
                {"dataset": "d1", "prefix": "d", "a_id": a, "b_id": b, "c_id": c},
            ]
        ),
    )

    assert stats["added_bridges"] == 1
    assert stats["edges_added"] == 2
    assert edge_pairs(graph) == {(a, b), (b, c)}


def test_two_edge_bridge_rejects_existing_edge_conflict() -> None:
    graph, a, b, c = _abc_graph()
    add_edge(graph, a, b)

    stats = add_two_edge_bridges(
        graph,
        _two_edge_frame(
            [
                {"dataset": "d1", "prefix": "d", "a_id": a, "b_id": b, "c_id": c},
            ]
        ),
    )

    assert stats["added_bridges"] == 0
    assert stats["existing_edge_conflict"] == 1
    assert edge_pairs(graph) == {(a, b)}


def test_two_edge_bridge_rejects_degree_conflict() -> None:
    """B is no longer an isolated middle: it already has a predecessor."""
    graph, a, b, c = _abc_graph()
    other = add_node(graph, 0, y=5.0)
    add_edge(graph, other, b)

    stats = add_two_edge_bridges(
        graph,
        _two_edge_frame(
            [
                {"dataset": "d1", "prefix": "d", "a_id": a, "b_id": b, "c_id": c},
            ]
        ),
    )

    assert stats["added_bridges"] == 0
    assert stats["degree_conflict"] == 1
    assert edge_pairs(graph) == {(other, b)}


def test_two_edge_bridge_rejects_missing_endpoint() -> None:
    graph, a, b, _c = _abc_graph()

    stats = add_two_edge_bridges(
        graph,
        _two_edge_frame(
            [
                {"dataset": "d1", "prefix": "d", "a_id": a, "b_id": b, "c_id": 999},
            ]
        ),
    )

    assert stats["added_bridges"] == 0
    assert stats["missing_endpoint"] == 1


def test_two_edge_bridge_adds_no_nodes() -> None:
    graph, a, b, c = _abc_graph()
    before = node_count(graph)

    add_two_edge_bridges(
        graph,
        _two_edge_frame(
            [
                {"dataset": "d1", "prefix": "d", "a_id": a, "b_id": b, "c_id": c},
            ]
        ),
    )

    assert node_count(graph) == before


# ============================================================
# Component cleanup
# ============================================================


def test_cleanup_removes_singleton_component() -> None:
    graph = new_graph()
    lonely = add_node(graph, 0)
    a = add_node(graph, 0, y=10.0)
    b = add_node(graph, 1, y=10.5)
    c = add_node(graph, 2, y=11.0)
    add_edge(graph, a, b)
    add_edge(graph, b, c)

    stats = apply_frozen_component_cleanup(graph, UNIT_SCALE)

    assert stats["singleton_components"] == 1
    assert stats["nodes_removed"] == 1
    assert not graph.has_node(lonely)
    assert graph.has_node(a) and graph.has_node(b) and graph.has_node(c)


def test_cleanup_removes_two_node_component_above_threshold() -> None:
    graph = new_graph()
    a = add_node(graph, 0, y=0.0)
    b = add_node(graph, 1, y=4.0)  # 4.0 um > 3.5 um
    add_edge(graph, a, b)

    stats = apply_frozen_component_cleanup(graph, UNIT_SCALE)

    assert stats["long_two_node_components"] == 1
    assert stats["nodes_removed"] == 2
    assert not graph.has_node(a)
    assert not graph.has_node(b)


def test_cleanup_preserves_two_node_component_at_threshold() -> None:
    graph = new_graph()
    a = add_node(graph, 0, y=0.0)
    b = add_node(graph, 1, y=3.5)  # exactly 3.5 um, not > 3.5
    add_edge(graph, a, b)

    stats = apply_frozen_component_cleanup(graph, UNIT_SCALE)

    assert stats["long_two_node_components"] == 0
    assert stats["nodes_removed"] == 0
    assert graph.has_node(a) and graph.has_node(b)


def test_cleanup_preserves_short_two_node_component() -> None:
    graph = new_graph()
    a = add_node(graph, 0, y=0.0)
    b = add_node(graph, 1, y=1.0)
    add_edge(graph, a, b)

    stats = apply_frozen_component_cleanup(graph, UNIT_SCALE)

    assert stats["nodes_removed"] == 0
    assert graph.has_node(a) and graph.has_node(b)


def test_cleanup_respects_anisotropic_scale() -> None:
    """A 1-voxel z step is 5 um under z-scale 5.0 and must be removed."""
    graph = new_graph()
    a = add_node(graph, 0, z=0.0)
    b = add_node(graph, 1, z=1.0)
    add_edge(graph, a, b)

    stats = apply_frozen_component_cleanup(graph, (5.0, 1.0, 1.0))

    assert stats["long_two_node_components"] == 1
    assert stats["nodes_removed"] == 2


def test_cleanup_adds_no_nodes() -> None:
    graph = new_graph()
    a = add_node(graph, 0, y=0.0)
    b = add_node(graph, 1, y=1.0)
    add_edge(graph, a, b)

    apply_frozen_component_cleanup(graph, UNIT_SCALE)

    assert node_count(graph) == 2


# ============================================================
# Policy builders
# ============================================================


def _gap_candidate(dataset: str, a_id: int, b_id: int, c_id: int, kind: str, **over) -> dict:
    row = {
        "dataset": dataset,
        "prefix": dataset.split("_")[0],
        "type": kind,
        "a_id": a_id,
        "b_id": b_id,
        "c_id": c_id,
        "prediction_residual_um": 1.0,
        "ambiguity_margin_um": 2.0,
        "max_step_um": 3.0,
        "step_balance": 0.1,
        "turn_angle_deg": 10.0,
    }
    row.update(over)
    return row


def test_mutual_policy_requires_both_directions() -> None:
    """An edge passes only when both LEFT_MISSING and RIGHT_MISSING agree."""
    # Both rows must resolve to the same edge (1, 2):
    #   LEFT_MISSING(a, b, c)  -> edge (a, b)
    #   RIGHT_MISSING(a, b, c) -> edge (b, c)
    mutual = pd.DataFrame(
        [
            _gap_candidate("d1", 1, 2, 3, "LEFT_MISSING"),
            _gap_candidate("d1", 0, 1, 2, "RIGHT_MISSING"),
        ]
    )
    one_sided = pd.DataFrame([_gap_candidate("d1", 4, 5, 6, "LEFT_MISSING")])

    assert len(build_mutual_policy(mutual)) == 1
    assert len(build_mutual_policy(one_sided)) == 0


def test_mutual_policy_rejects_residual_beyond_frozen_radius() -> None:
    candidates = pd.DataFrame(
        [
            _gap_candidate("d1", 1, 2, 3, "LEFT_MISSING", prediction_residual_um=5.0),
            _gap_candidate("d1", 0, 1, 2, "RIGHT_MISSING", prediction_residual_um=5.0),
        ]
    )

    assert len(build_mutual_policy(candidates)) == 0


def test_mutual_policy_rejects_low_ambiguity_margin() -> None:
    candidates = pd.DataFrame(
        [
            _gap_candidate("d1", 1, 2, 3, "LEFT_MISSING", ambiguity_margin_um=0.5),
            _gap_candidate("d1", 0, 1, 2, "RIGHT_MISSING", ambiguity_margin_um=0.5),
        ]
    )

    assert len(build_mutual_policy(candidates)) == 0


def _bridge_candidate(dataset: str, a_id: int, b_id: int, c_id: int, **over) -> dict:
    row = {
        "dataset": dataset,
        "prefix": dataset.split("_")[0],
        "a_id": a_id,
        "b_id": b_id,
        "c_id": c_id,
        "cv_residual_um": 1.0,
        "max_step_um": 3.0,
        "step_balance": 0.1,
        "turn_angle_deg": 10.0,
        "ambiguity_margin_um": 2.0,
        "pair_count": 1,
    }
    row.update(over)
    return row


def test_strict_two_edge_policy_accepts_clean_unique_bridge() -> None:
    candidates = pd.DataFrame([_bridge_candidate("d1", 1, 2, 3)])

    assert len(build_strict_two_edge_policy(candidates)) == 1


def test_strict_two_edge_policy_rejects_endpoint_conflict() -> None:
    """The same A feeding two different middles is not conflict free."""
    candidates = pd.DataFrame(
        [
            _bridge_candidate("d1", 1, 2, 3),
            _bridge_candidate("d1", 1, 4, 5),
        ]
    )

    assert len(build_strict_two_edge_policy(candidates)) == 0


def test_strict_two_edge_policy_rejects_non_unique_pair_count() -> None:
    candidates = pd.DataFrame([_bridge_candidate("d1", 1, 2, 3, pair_count=2)])

    assert len(build_strict_two_edge_policy(candidates)) == 0


def test_strict_two_edge_policy_rejects_relaxed_geometry() -> None:
    """Rejected experimental variants (r=5, angle 90, balance 0.75) must not pass."""
    assert (
        len(
            build_strict_two_edge_policy(
                pd.DataFrame(
                    [
                        _bridge_candidate("d1", 1, 2, 3, cv_residual_um=5.0),
                    ]
                )
            )
        )
        == 0
    )

    assert (
        len(
            build_strict_two_edge_policy(
                pd.DataFrame(
                    [
                        _bridge_candidate("d1", 1, 2, 3, turn_angle_deg=90.0),
                    ]
                )
            )
        )
        == 0
    )

    assert (
        len(
            build_strict_two_edge_policy(
                pd.DataFrame(
                    [
                        _bridge_candidate("d1", 1, 2, 3, step_balance=0.75),
                    ]
                )
            )
        )
        == 0
    )


def test_strict_two_edge_policy_scopes_conflicts_per_dataset() -> None:
    """The same node ids in different datasets must not collide."""
    candidates = pd.DataFrame(
        [
            _bridge_candidate("d1", 1, 2, 3),
            _bridge_candidate("d2", 1, 2, 3),
        ]
    )

    assert len(build_strict_two_edge_policy(candidates)) == 2


# ============================================================
# Model-specific policy filtering
# ============================================================


def _policies_for_two_datasets() -> FrozenV9Policies:
    single = _single_policy_frame(
        [
            {"dataset": "d1", "prefix": "d", "source_id": 0, "target_id": 1},
            {"dataset": "d2", "prefix": "d", "source_id": 0, "target_id": 1},
        ]
    )
    two = _two_edge_frame(
        [
            {"dataset": "d2", "prefix": "d", "a_id": 0, "b_id": 1, "c_id": 2},
        ]
    )
    return FrozenV9Policies(
        gap_candidates=pd.DataFrame(),
        single_policy=single,
        isolated_candidates=pd.DataFrame(),
        isolated_scope=pd.DataFrame(
            [
                {"dataset": "d1", "isolated_nodes": 3, "isolated_with_local_pair": 1},
                {"dataset": "d2", "isolated_nodes": 4, "isolated_with_local_pair": 2},
            ]
        ),
        strict_two_policy=two,
    )


def test_apply_frozen_v9_uses_only_the_requested_dataset_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, _a, _b, _c = _abc_graph()

    monkeypatch.setattr(
        frozen_v9,
        "prepare_precleanup_graph",
        lambda name, p995, p9975, train, config: (graph, UNIT_SCALE),
    )

    policies = _policies_for_two_datasets()

    _graph, _scale, stats = apply_frozen_v9("d1", Path("p995"), Path("p9975"), policies, Path("train"))

    # d1 has one single-edge row and no two-edge rows; d2's rows must be ignored.
    assert stats.single_added == 1
    assert stats.two_requested == 0
    assert stats.two_added == 0
    assert stats.dataset == "d1"


def test_apply_frozen_v9_reports_associations_and_never_adds_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = new_graph()
    a = add_node(graph, 0, y=0.0)
    b = add_node(graph, 1, y=1.0)
    c = add_node(graph, 2, y=2.0)
    d = add_node(graph, 0, y=20.0)
    e = add_node(graph, 1, y=20.5)
    before = node_count(graph)

    monkeypatch.setattr(
        frozen_v9,
        "prepare_precleanup_graph",
        lambda name, p995, p9975, train, config: (graph, UNIT_SCALE),
    )

    policies = FrozenV9Policies(
        gap_candidates=pd.DataFrame(),
        single_policy=_single_policy_frame(
            [
                {"dataset": "d1", "prefix": "d", "source_id": d, "target_id": e},
            ]
        ),
        isolated_candidates=pd.DataFrame(),
        isolated_scope=pd.DataFrame(),
        strict_two_policy=_two_edge_frame(
            [
                {"dataset": "d1", "prefix": "d", "a_id": a, "b_id": b, "c_id": c},
            ]
        ),
    )

    _graph, _scale, stats = apply_frozen_v9("d1", Path("p995"), Path("p9975"), policies, Path("train"))

    assert stats.single_added == 1
    assert stats.two_added == 1
    assert stats.two_edges_added == 2
    assert stats.associations_added == 2  # two-edge stage only, 2 per bridge
    assert stats.total_edges_added == 3
    # No synthetic nodes are ever inserted by the pipeline.
    assert node_count(graph) == before


def test_policies_counts_are_reported_from_the_generated_tables() -> None:
    policies = _policies_for_two_datasets()

    counts = policies.counts()

    assert counts["single_policy"] == 2
    assert counts["strict_two_policy"] == 1
    assert counts["isolated_nodes"] == 7
    assert counts["isolated_with_local_pair"] == 3


# ============================================================
# Prefix summary
# ============================================================


def _evaluation_for_prefix_tests() -> FrozenV9Evaluation:
    records = [
        {
            "edge_tp": 10,
            "edge_fp": 1,
            "edge_fn": 1,
            "division_tp": 1,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 100,
            "total_node_ratio": 1.0,
            "edge_jaccard": 0.8,
            "adj_edge_jaccard": 0.8,
            "node_recall": 0.9,
        },
        {
            "edge_tp": 20,
            "edge_fp": 2,
            "edge_fn": 2,
            "division_tp": 0,
            "division_fp": 1,
            "division_fn": 1,
            "num_pred_nodes": 200,
            "total_node_ratio": 1.0,
            "edge_jaccard": 0.7,
            "adj_edge_jaccard": 0.7,
            "node_recall": 0.8,
        },
    ]
    dataset_df = pd.DataFrame(
        [
            {"dataset": "44b6_aaa", "prefix": "44b6"},
            {"dataset": "6bba_bbb", "prefix": "6bba"},
        ]
    )
    return FrozenV9Evaluation(
        summary={},
        records=records,
        dataset_df=dataset_df,
        interventions={},
        policies=_policies_for_two_datasets(),
    )


def test_summarize_by_prefix_groups_existing_records() -> None:
    evaluation = _evaluation_for_prefix_tests()

    summary = summarize_by_prefix(evaluation)

    assert list(summary["prefix"]) == ["44b6", "6bba"]
    assert list(summary["n"]) == [1, 1]
    assert "score" in summary.columns


def test_summarize_by_prefix_does_not_alter_policies_or_records() -> None:
    evaluation = _evaluation_for_prefix_tests()

    single_before = evaluation.policies.single_policy.copy()
    two_before = evaluation.policies.strict_two_policy.copy()
    records_before = [dict(record) for record in evaluation.records]

    summarize_by_prefix(evaluation)

    pd.testing.assert_frame_equal(evaluation.policies.single_policy, single_before)
    pd.testing.assert_frame_equal(evaluation.policies.strict_two_policy, two_before)
    assert evaluation.records == records_before


def test_summarize_by_prefix_honors_explicit_prefix_map() -> None:
    evaluation = _evaluation_for_prefix_tests()

    summary = summarize_by_prefix(
        evaluation,
        prefix_map={"44b6_aaa": "groupA", "6bba_bbb": "groupA"},
    )

    assert list(summary["prefix"]) == ["groupA"]
    assert list(summary["n"]) == [2]
