"""Focused tests for src/biohub_cell_tracking/division_geometry.py.

Small deterministic in-memory frames and graphs only. No dataset reads, no
model inference. The full census lives in scripts/audit_stage11_geometry.py.
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

from biohub_cell_tracking.division_audit import fork_geometry
from biohub_cell_tracking.division_geometry import (
    GEOMETRY_VARIABLES,
    QUANTILES,
    GeometryGate,
    enumerate_geometry_grid,
    evaluate_geometry_gate,
    geometry_pass_rates,
    grid_gt_recall,
    quantile_summary,
)
from biohub_cell_tracking.frozen_v9 import FROZEN_V9_CONFIG, apply_frozen_g2

UNIT_SCALE = (1.0, 1.0, 1.0)
ANISOTROPIC_SCALE = (1.625, 0.40625, 0.40625)


# ============================================================
# Geometry measurement
# ============================================================


def test_opposed_daughters_measure_a_180_degree_division() -> None:
    geometry = fork_geometry((0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, -4.0, 0.0), UNIT_SCALE)

    assert geometry.division_angle_deg == pytest.approx(180.0)
    assert geometry.daughter_separation_um == pytest.approx(8.0)
    assert geometry.max_parent_daughter_um == pytest.approx(4.0)
    assert geometry.distance_balance == pytest.approx(0.0)


def test_perpendicular_daughters_measure_a_90_degree_division() -> None:
    geometry = fork_geometry((0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0), UNIT_SCALE)

    assert geometry.division_angle_deg == pytest.approx(90.0)
    # Right isosceles triangle: separation is the hypotenuse.
    assert geometry.daughter_separation_um == pytest.approx(5.0 * np.sqrt(2.0))
    assert geometry.max_parent_daughter_um == pytest.approx(5.0)


def test_daughter_separation_is_the_daughter_to_daughter_distance() -> None:
    # Daughters on the same side: separation is independent of the parent.
    geometry = fork_geometry((0.0, 0.0, 0.0), (0.0, 10.0, 3.0), (0.0, 10.0, -3.0), UNIT_SCALE)

    assert geometry.daughter_separation_um == pytest.approx(6.0)


def test_geometry_respects_anisotropic_voxel_scale() -> None:
    # One voxel along z is 4x one voxel along y under the competition scale.
    along_z = fork_geometry((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), ANISOTROPIC_SCALE)
    along_y = fork_geometry((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), ANISOTROPIC_SCALE)

    assert along_z.daughter_separation_um == pytest.approx(2 * 1.625)
    assert along_y.daughter_separation_um == pytest.approx(2 * 0.40625)
    assert along_z.daughter_separation_um == pytest.approx(4.0 * along_y.daughter_separation_um)


def test_max_parent_daughter_takes_the_longer_arm() -> None:
    geometry = fork_geometry((0.0, 0.0, 0.0), (0.0, 7.0, 0.0), (0.0, -2.0, 0.0), UNIT_SCALE)

    assert geometry.d1_um == pytest.approx(7.0)
    assert geometry.d2_um == pytest.approx(2.0)
    assert geometry.max_parent_daughter_um == pytest.approx(7.0)


def test_distance_balance_is_normalised_arm_asymmetry() -> None:
    symmetric = fork_geometry((0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, -4.0, 0.0), UNIT_SCALE)
    lopsided = fork_geometry((0.0, 0.0, 0.0), (0.0, 7.0, 0.0), (0.0, -2.0, 0.0), UNIT_SCALE)

    assert symmetric.distance_balance == pytest.approx(0.0)
    assert lopsided.distance_balance == pytest.approx(5.0 / 9.0)


# ============================================================
# GeometryGate
# ============================================================


def test_frozen_gate_matches_the_locked_g2_thresholds() -> None:
    gate = GeometryGate.frozen()

    assert gate.daughter_sep_min_um == 6.0
    assert gate.angle_min_deg == 120.0
    assert gate.max_parent_daughter_um == 8.0
    assert gate.distance_balance_max == 0.50
    assert gate.p2_min == 0.80
    assert gate.is_frozen_geometry


def test_gate_label_is_stable_and_distinguishes_gates() -> None:
    frozen = GeometryGate.frozen()
    relaxed = GeometryGate(0.0, 60.0, 12.0, 1.0)

    assert frozen.label == GeometryGate.frozen().label
    assert frozen.label != relaxed.label
    assert not relaxed.is_frozen_geometry


def test_gate_to_config_changes_only_the_division_thresholds() -> None:
    gate = GeometryGate(2.0, 60.0, 12.0, 0.75, p2_min=0.85)
    config = gate.to_config()

    assert config.division_daughter_sep_min_um == 2.0
    assert config.division_angle_min_deg == 60.0
    assert config.division_max_parent_daughter_um == 12.0
    assert config.division_distance_balance_max == 0.75
    assert config.division_p2_min == 0.85

    # Everything else must remain the frozen pipeline.
    for field in config.__dataclass_fields__:
        if field.startswith("division_"):
            continue
        assert getattr(config, field) == getattr(FROZEN_V9_CONFIG, field), field


def test_frozen_gate_round_trips_through_to_config() -> None:
    assert GeometryGate.frozen().to_config() == FROZEN_V9_CONFIG


# ============================================================
# Gate evaluation reproduces frozen G2
# ============================================================


def new_graph():
    graph = td.graph.IndexedRXGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    for key in ("edge_prob", "edge_dist"):
        graph.add_edge_attr_key(key, pl.Float64, 0.0)
    return graph


def fork_graph(*, d1: float, d2: float, angle_deg: float, p2: float):
    """Parent forking with fully controlled geometry, as apply_frozen_g2 sees it."""
    theta = np.radians(angle_deg)
    graph = new_graph()
    parent = graph.add_node({"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    child1 = graph.add_node({"t": 1, "z": 0.0, "y": d1, "x": 0.0})
    child2 = graph.add_node({"t": 1, "z": 0.0, "y": d2 * np.cos(theta), "x": d2 * np.sin(theta)})
    graph.add_edge(parent, child1, {"edge_prob": 0.99, "edge_dist": 1.0})
    graph.add_edge(parent, child2, {"edge_prob": float(p2), "edge_dist": 1.0})
    return graph, parent, child1, child2


CASES = [
    # (d1, d2, angle, p2, expected accept under the frozen gate)
    (4.0, 4.0, 180.0, 0.95, True),
    (4.0, 4.0, 180.0, 0.50, False),  # p2 too low
    (2.9, 2.9, 180.0, 0.95, False),  # separation 5.8 < 6
    (6.0, 6.0, 90.0, 0.95, False),  # angle 90 < 120
    (9.0, 9.0, 180.0, 0.95, False),  # max distance 9 > 8
    (7.0, 2.0, 180.0, 0.95, False),  # balance 5/9 > 0.5
]


@pytest.mark.parametrize(("d1", "d2", "angle", "p2", "expected"), CASES)
def test_gate_evaluation_reproduces_apply_frozen_g2(d1, d2, angle, p2, expected) -> None:
    graph, parent, child1, child2 = fork_graph(d1=d1, d2=d2, angle_deg=angle, p2=p2)

    nodes = graph.node_attrs(unpack=True).to_pandas().set_index("node_id")
    measured = fork_geometry(
        (nodes.loc[parent, "z"], nodes.loc[parent, "y"], nodes.loc[parent, "x"]),
        (nodes.loc[child1, "z"], nodes.loc[child1, "y"], nodes.loc[child1, "x"]),
        (nodes.loc[child2, "z"], nodes.loc[child2, "y"], nodes.loc[child2, "x"]),
        UNIT_SCALE,
    )

    frame = pd.DataFrame(
        [
            {
                "daughter_separation_um": measured.daughter_separation_um,
                "division_angle_deg": measured.division_angle_deg,
                "max_parent_daughter_um": measured.max_parent_daughter_um,
                "distance_balance": measured.distance_balance,
                "p2": p2,
            }
        ]
    )

    gate_says = bool(evaluate_geometry_gate(frame, GeometryGate.frozen()).loc[0, "passes_all"])

    apply_frozen_g2(graph, np.asarray(UNIT_SCALE, dtype=float))
    g2_kept = (parent, child2) in {(int(s), int(t)) for s, t in graph.edge_list()}

    assert gate_says == g2_kept
    assert gate_says == expected


def test_gate_treats_a_non_finite_angle_as_a_failure() -> None:
    frame = pd.DataFrame(
        [
            {
                "daughter_separation_um": 8.0,
                "division_angle_deg": float("nan"),
                "max_parent_daughter_um": 4.0,
                "distance_balance": 0.0,
            }
        ]
    )

    flags = evaluate_geometry_gate(frame, GeometryGate.frozen())

    assert bool(flags.loc[0, "passes_angle"]) is False
    assert bool(flags.loc[0, "passes_geometry"]) is False


def test_gate_without_p2_column_reports_geometry_only() -> None:
    frame = pd.DataFrame(
        [
            {
                "daughter_separation_um": 8.0,
                "division_angle_deg": 180.0,
                "max_parent_daughter_um": 4.0,
                "distance_balance": 0.0,
            }
        ]
    )

    flags = evaluate_geometry_gate(frame, GeometryGate.frozen())

    assert "passes_p2" not in flags.columns
    assert bool(flags.loc[0, "passes_all"]) is True


def test_evaluate_geometry_gate_does_not_mutate_its_input() -> None:
    frame = pd.DataFrame(
        [
            {
                "daughter_separation_um": 8.0,
                "division_angle_deg": 180.0,
                "max_parent_daughter_um": 4.0,
                "distance_balance": 0.0,
                "p2": 0.9,
            }
        ]
    )
    before = frame.copy(deep=True)

    evaluate_geometry_gate(frame, GeometryGate.frozen())

    pd.testing.assert_frame_equal(frame, before)


# ============================================================
# Census frames
# ============================================================


def census_frame() -> pd.DataFrame:
    """Six synthetic GT divisions spanning both folds and both prefixes."""
    return pd.DataFrame(
        [
            # fold, prefix, sep, angle, d1, d2, maxd, balance
            (0, "44b6", 8.0, 180.0, 4.0, 4.0, 4.0, 0.00),
            (0, "44b6", 4.0, 180.0, 2.0, 2.0, 2.0, 0.00),
            (0, "6bba", 9.0, 90.0, 6.4, 6.4, 6.4, 0.00),
            (1, "6bba", 8.0, 180.0, 4.0, 4.0, 4.0, 0.00),
            (1, "6bba", 12.0, 150.0, 9.0, 9.0, 9.0, 0.00),
            (1, "44b6", 9.0, 180.0, 7.0, 2.0, 7.0, 5.0 / 9.0),
        ],
        columns=[
            "fold",
            "prefix",
            "daughter_separation_um",
            "division_angle_deg",
            "parent_daughter_1_um",
            "parent_daughter_2_um",
            "max_parent_daughter_um",
            "distance_balance",
        ],
    )


def test_quantile_summary_covers_every_geometry_variable() -> None:
    summary = quantile_summary(census_frame())

    assert set(summary["variable"]) == set(GEOMETRY_VARIABLES)
    for q in QUANTILES:
        assert f"q{round(q * 100):02d}" in summary.columns

    sep = summary[summary["variable"] == "daughter_separation_um"].iloc[0]
    assert sep["n"] == 6
    assert sep["min"] == pytest.approx(4.0)
    assert sep["max"] == pytest.approx(12.0)
    assert sep["q50"] == pytest.approx(8.5)


def test_quantile_summary_aggregates_by_fold() -> None:
    summary = quantile_summary(census_frame(), group_cols=["fold"])

    assert set(summary["fold"]) == {0, 1}
    fold0 = summary[(summary["fold"] == 0) & (summary["variable"] == "daughter_separation_um")]
    assert int(fold0.iloc[0]["n"]) == 3
    assert fold0.iloc[0]["q50"] == pytest.approx(8.0)


def test_quantile_summary_aggregates_by_prefix() -> None:
    summary = quantile_summary(census_frame(), group_cols=["prefix"])

    counts = {
        row["prefix"]: int(row["n"]) for _, row in summary[summary["variable"] == "division_angle_deg"].iterrows()
    }
    assert counts == {"44b6": 3, "6bba": 3}


def test_pass_rates_report_each_criterion_globally() -> None:
    rates = geometry_pass_rates(census_frame(), GeometryGate.frozen())

    assert int(rates.loc[0, "n"]) == 6
    # Exactly one row fails each criterion: rows 1, 2, 4 and 5 respectively.
    assert int(rates.loc[0, "passes_separation_n"]) == 5
    assert int(rates.loc[0, "passes_angle_n"]) == 5
    assert int(rates.loc[0, "passes_distance_n"]) == 5
    assert int(rates.loc[0, "passes_balance_n"]) == 5
    assert int(rates.loc[0, "passes_geometry_n"]) == 2
    assert rates.loc[0, "passes_geometry"] == pytest.approx(2 / 6)


def test_pass_rates_group_by_fold_and_prefix() -> None:
    by_fold = geometry_pass_rates(census_frame(), GeometryGate.frozen(), group_cols=["fold"])
    by_prefix = geometry_pass_rates(census_frame(), GeometryGate.frozen(), group_cols=["prefix"])

    fold_pass = dict(zip(by_fold["fold"], by_fold["passes_geometry_n"], strict=True))
    assert fold_pass == {0: 1, 1: 1}

    prefix_n = dict(zip(by_prefix["prefix"], by_prefix["n"], strict=True))
    assert prefix_n == {"44b6": 3, "6bba": 3}


# ============================================================
# Threshold grid
# ============================================================


def test_grid_enumeration_is_deterministic_and_complete() -> None:
    kwargs = {
        "separations": [0.0, 6.0],
        "angles": [0.0, 120.0],
        "distances": [8.0, 12.0],
        "balances": [0.5, 1.0],
    }

    first = enumerate_geometry_grid(**kwargs)
    second = enumerate_geometry_grid(**kwargs)

    assert len(first) == 2 * 2 * 2 * 2
    assert [g.label for g in first] == [g.label for g in second]
    assert len({g.label for g in first}) == len(first)
    assert first[0].geometry_key == (0.0, 0.0, 8.0, 0.5)


def test_grid_contains_the_frozen_gate_when_its_levels_are_present() -> None:
    grid = enumerate_geometry_grid([0, 2, 4, 6], [0, 60, 90, 120], [8, 10, 12, 14], [0.50, 0.75, 1.00])

    assert len(grid) == 192
    assert sum(g.is_frozen_geometry for g in grid) == 1


def test_grid_p2_levels_multiply_the_configurations() -> None:
    grid = enumerate_geometry_grid([6.0], [120.0], [8.0], [0.5], p2_values=[0.75, 0.80, 0.85])

    assert [g.p2_min for g in grid] == [0.75, 0.80, 0.85]
    assert len({g.label for g in grid}) == 3


def test_grid_gt_recall_reports_overall_fold_and_prefix_recall() -> None:
    census = census_frame()
    gates = (GeometryGate.frozen(), GeometryGate(0.0, 0.0, 14.0, 1.0))

    recall = grid_gt_recall(census, gates).set_index("gate")

    frozen = recall.loc[GeometryGate.frozen().label]
    assert int(frozen["gt_pass"]) == 2
    assert frozen["gt_recall"] == pytest.approx(2 / 6)
    assert frozen["gt_recall_fold0"] == pytest.approx(1 / 3)
    assert frozen["gt_recall_fold1"] == pytest.approx(1 / 3)
    assert bool(frozen["is_frozen_geometry"]) is True

    permissive = recall.loc[GeometryGate(0.0, 0.0, 14.0, 1.0).label]
    assert permissive["gt_recall"] == pytest.approx(1.0)
    assert permissive["gt_recall_fold_min"] == pytest.approx(1.0)
    assert permissive["gt_recall_fold_spread"] == pytest.approx(0.0)
    assert permissive["gt_recall_44b6"] == pytest.approx(1.0)
    assert permissive["gt_recall_6bba"] == pytest.approx(1.0)


def test_grid_gt_recall_ignores_p2_for_ground_truth() -> None:
    census = census_frame()
    strict_p2 = GeometryGate(0.0, 0.0, 14.0, 1.0, p2_min=0.99)
    loose_p2 = GeometryGate(0.0, 0.0, 14.0, 1.0, p2_min=0.10)

    recall = grid_gt_recall(census, (strict_p2, loose_p2))

    assert recall["gt_recall"].nunique() == 1
