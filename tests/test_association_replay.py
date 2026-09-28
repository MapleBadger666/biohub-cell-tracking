"""Focused tests for src/biohub_cell_tracking/association_replay.py.

Tiny hand-built score matrices only. No checkpoint is loaded and no model runs;
the targeted replay lives in scripts/replay_stage11_association.py.
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

from biohub_cell_tracking.association_replay import (
    CANDIDATE_RULES,
    COMPETITOR_MARGIN_MAX,
    DEGREE_CONFLICT,
    HIGH_RANK_COMPETITOR_SUPPRESSED,
    HIGH_RANK_THRESHOLD_SUPPRESSED,
    LOW_RANK_MODEL_FAILURE,
    MISSING_EDGE_CLASSES,
    OTHER,
    CandidateRule,
    FramePair,
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


def softmax_columns(raw: np.ndarray) -> np.ndarray:
    """The predictor's activation: softmax over dim=0, i.e. down each column."""
    raw = np.asarray(raw, dtype=float)
    shifted = raw - raw.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)


# 3 sources x 2 targets. Column 0 is dominated by source 0; column 1 is a near
# tie between sources 1 and 2.
RAW = np.array(
    [
        [4.0, 0.0],
        [1.0, 2.0],
        [0.0, 2.2],
    ]
)
PROBS = softmax_columns(RAW)


# ============================================================
# Frame pairs
# ============================================================


def test_frame_pair_exposes_the_following_frame() -> None:
    pair = FramePair("6bba_x", 48)

    assert pair.t_next == 49
    assert pair.key == ("6bba_x", 48)


def test_duplicate_frame_pairs_are_collapsed_deterministically() -> None:
    pairs = [
        FramePair("b", 5),
        FramePair("a", 9),
        FramePair("b", 5),
        FramePair("a", 2),
        FramePair("a", 9),
    ]

    deduped = dedupe_frame_pairs(pairs)

    assert deduped == (FramePair("a", 2), FramePair("a", 9), FramePair("b", 5))
    assert dedupe_frame_pairs(pairs) == deduped
    assert dedupe_frame_pairs(list(reversed(pairs))) == deduped


def test_dedupe_of_an_empty_request_list_is_empty() -> None:
    assert dedupe_frame_pairs([]) == ()


# ============================================================
# Column statistics
# ============================================================


def test_column_ranks_follow_the_column_softmax() -> None:
    winner = column_stats(PROBS, RAW, source_index=0, target_index=0)
    runner_up = column_stats(PROBS, RAW, source_index=1, target_index=0)
    last = column_stats(PROBS, RAW, source_index=2, target_index=0)

    assert (winner.column_rank, runner_up.column_rank, last.column_rank) == (1, 2, 3)
    assert winner.is_column_winner
    assert winner.column_top1_source == 0
    assert winner.column_top2_source == 1
    assert winner.column_size == 3
    assert winner.column_softmax_prob == pytest.approx(PROBS[0, 0])
    assert winner.raw_logit == pytest.approx(4.0)


def test_column_margin_is_measured_against_the_column_winner() -> None:
    stats = column_stats(PROBS, RAW, source_index=1, target_index=1)

    assert stats.column_rank == 2
    assert stats.column_top1_source == 2
    assert stats.column_margin_to_top1 == pytest.approx(PROBS[2, 1] - PROBS[1, 1])
    # Near tie: the margin is small even though the correct source lost.
    assert stats.column_margin_to_top1 < 0.10


def test_column_winner_has_zero_margin() -> None:
    stats = column_stats(PROBS, RAW, source_index=2, target_index=1)

    assert stats.column_rank == 1
    assert stats.column_margin_to_top1 == pytest.approx(0.0)


def test_column_ranks_break_ties_by_ascending_source_index() -> None:
    tied_raw = np.array([[1.0], [1.0], [1.0]])
    tied_probs = softmax_columns(tied_raw)

    ranks = [column_stats(tied_probs, tied_raw, i, 0).column_rank for i in range(3)]

    assert ranks == [1, 2, 3]
    assert column_stats(tied_probs, tied_raw, 0, 0).column_top1_source == 0


def test_single_source_column_has_no_runner_up() -> None:
    raw = np.array([[2.0, 3.0]])
    stats = column_stats(softmax_columns(raw), raw, 0, 0)

    assert stats.column_rank == 1
    assert stats.column_top2_source is None
    assert np.isnan(stats.column_top2_prob)


# ============================================================
# Row statistics
# ============================================================


def test_row_ranks_follow_the_raw_logits() -> None:
    stats = row_stats(RAW, source_index=1, target_index=1)

    assert stats.row_rank_by_raw_logit == 1
    assert stats.parent_top1_target == 1
    assert stats.parent_top2_target == 0
    assert stats.parent_top3_target is None
    assert stats.row_size == 2


def test_row_rank_order_can_invert_the_probability_order() -> None:
    # Column 1 carries a dominant competitor, so source 1's larger logit there
    # still yields a smaller column-softmax probability. Ranking a row by
    # probability would therefore report the wrong best target.
    raw = np.array([[0.0, 5.0], [1.0, 2.0]])
    probs = softmax_columns(raw)

    assert raw[1, 0] < raw[1, 1]
    assert probs[1, 0] > probs[1, 1]

    assert row_stats(raw, 1, 1).row_rank_by_raw_logit == 1
    assert row_stats(raw, 1, 0).row_rank_by_raw_logit == 2


def test_row_rank_orders_every_target_of_a_source() -> None:
    raw = np.array([[0.5, 3.0, 1.0, 2.0]])

    ranks = [row_stats(raw, 0, j).row_rank_by_raw_logit for j in range(4)]

    assert ranks == [4, 1, 3, 2]
    assert row_stats(raw, 0, 0).parent_top3_target == 2


def test_row_ranks_break_ties_by_ascending_target_index() -> None:
    raw = np.array([[2.0, 2.0, 2.0]])

    ranks = [row_stats(raw, 0, j).row_rank_by_raw_logit for j in range(3)]

    assert ranks == [1, 2, 3]


# ============================================================
# Missing-edge classification
# ============================================================


def classify(**overrides) -> str:
    kwargs = {
        "column_rank": 1,
        "column_softmax_prob": 0.30,
        "column_margin_to_top1": 0.0,
        "parent_at_child_cap": False,
    }
    kwargs.update(overrides)
    return classify_missing_edge(**kwargs)


def test_rank_one_below_threshold_is_threshold_suppressed() -> None:
    assert classify(column_rank=1, column_softmax_prob=0.49) == HIGH_RANK_THRESHOLD_SUPPRESSED


def test_rank_two_with_a_small_margin_is_competitor_suppressed() -> None:
    assert (
        classify(column_rank=2, column_softmax_prob=0.40, column_margin_to_top1=0.05) == HIGH_RANK_COMPETITOR_SUPPRESSED
    )


def test_rank_two_with_a_large_margin_is_a_model_failure() -> None:
    assert classify(column_rank=2, column_softmax_prob=0.05, column_margin_to_top1=0.80) == LOW_RANK_MODEL_FAILURE


def test_the_competitor_margin_boundary_is_inclusive() -> None:
    at_boundary = classify(column_rank=2, column_softmax_prob=0.3, column_margin_to_top1=COMPETITOR_MARGIN_MAX)
    past_boundary = classify(
        column_rank=2,
        column_softmax_prob=0.3,
        column_margin_to_top1=COMPETITOR_MARGIN_MAX + 1e-9,
    )

    assert at_boundary == HIGH_RANK_COMPETITOR_SUPPRESSED
    assert past_boundary == LOW_RANK_MODEL_FAILURE


@pytest.mark.parametrize("rank", [3, 4, 17])
def test_low_rank_is_a_model_failure(rank) -> None:
    assert classify(column_rank=rank, column_margin_to_top1=0.01) == LOW_RANK_MODEL_FAILURE


def test_a_viable_winning_edge_blocked_by_the_child_cap_is_a_degree_conflict() -> None:
    assert classify(column_rank=1, column_softmax_prob=0.90, parent_at_child_cap=True) == DEGREE_CONFLICT


def test_a_viable_winning_edge_with_no_cap_pressure_is_unexplained() -> None:
    assert classify(column_rank=1, column_softmax_prob=0.90, parent_at_child_cap=False) == OTHER


def test_classification_is_deterministic_and_within_the_vocabulary() -> None:
    results = {classify(column_rank=2, column_margin_to_top1=0.05) for _ in range(20)}

    assert results == {HIGH_RANK_COMPETITOR_SUPPRESSED}
    assert results <= set(MISSING_EDGE_CLASSES)


# ============================================================
# Candidate rules
# ============================================================


def rule_named(name: str) -> CandidateRule:
    return next(rule for rule in CANDIDATE_RULES if rule.name == name)


def test_rule_set_is_predeclared_and_marks_division_only_rules() -> None:
    names = [rule.name for rule in CANDIDATE_RULES]

    assert names[0] == "A_current_col_gt_0.50"
    assert len(names) == len(set(names))

    for rule in CANDIDATE_RULES:
        assert rule.division_only == (rule.kind in {"column_topk", "row_topk"})


def test_current_rule_reproduces_the_frozen_threshold_exactly() -> None:
    mask = rule_candidate_edges(rule_named("A_current_col_gt_0.50"), PROBS, RAW)

    assert np.array_equal(mask, PROBS > 0.50)
    # Column-softmax means at most one source per column can clear 0.5.
    assert mask.sum(axis=0).max() <= 1


def test_lower_column_thresholds_are_strictly_more_permissive() -> None:
    counts = [
        int(rule_candidate_edges(rule_named(name), PROBS, RAW).sum())
        for name in ("A_current_col_gt_0.50", "B_col_gt_0.40", "B_col_gt_0.30", "B_col_gt_0.20")
    ]

    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_column_top2_visibility_exposes_two_sources_per_target() -> None:
    mask = rule_candidate_edges(rule_named("C_col_rank_le_2"), PROBS, RAW)

    assert list(mask.sum(axis=0)) == [2, 2]
    # The near-tie runner-up in column 1 becomes visible; the current rule hides it.
    assert mask[1, 1]
    assert not rule_candidate_edges(rule_named("A_current_col_gt_0.50"), PROBS, RAW)[1, 1]


def test_parent_top_k_visibility_exposes_k_targets_per_source() -> None:
    raw = np.array([[0.5, 3.0, 1.0, 2.0], [4.0, 0.1, 0.2, 0.3]])
    probs = softmax_columns(raw)

    top2 = rule_candidate_edges(rule_named("D_row_rank_le_2"), probs, raw)
    top3 = rule_candidate_edges(rule_named("D_row_rank_le_3"), probs, raw)

    assert list(top2.sum(axis=1)) == [2, 2]
    assert list(top3.sum(axis=1)) == [3, 3]
    assert top2[0, 1] and top2[0, 3]
    assert not top2[0, 2]
    assert top3[0, 2]


def test_row_top_k_is_capped_by_the_number_of_targets() -> None:
    raw = np.array([[1.0, 2.0]])
    probs = softmax_columns(raw)

    mask = rule_candidate_edges(rule_named("D_row_rank_le_3"), probs, raw)

    assert int(mask.sum()) == 2


def test_edge_visible_agrees_with_the_matrix_mask() -> None:
    for rule in CANDIDATE_RULES:
        mask = rule_candidate_edges(rule, PROBS, RAW)
        for i in range(PROBS.shape[0]):
            for j in range(PROBS.shape[1]):
                expected = edge_visible(
                    rule,
                    column_rank=column_stats(PROBS, RAW, i, j).column_rank,
                    column_prob=float(PROBS[i, j]),
                    row_rank=row_stats(RAW, i, j).row_rank_by_raw_logit,
                )
                assert bool(mask[i, j]) == expected, (rule.name, i, j)


def test_edge_visible_rejects_an_unknown_rule_kind() -> None:
    with pytest.raises(ValueError, match="unknown rule kind"):
        edge_visible(CandidateRule("bogus", "nonsense", 1, False), column_rank=1, column_prob=1.0, row_rank=1)


def test_fork_counts_report_edges_and_multi_child_sources() -> None:
    mask = np.array([[True, True, False], [True, False, False], [False, False, False]])

    counts = rule_fork_counts(mask)

    assert counts == {
        "candidate_edges": 3,
        "candidate_forks": 1,
        "sources_with_any": 2,
        "max_out_candidates": 2,
    }


def test_rule_masks_do_not_mutate_the_score_matrices() -> None:
    probs_before = PROBS.copy()
    raw_before = RAW.copy()

    for rule in CANDIDATE_RULES:
        rule_candidate_edges(rule, PROBS, RAW)

    assert np.array_equal(PROBS, probs_before)
    assert np.array_equal(RAW, raw_before)


# ============================================================
# Fork evidence
# ============================================================


def edge_record(prob: float, logit: float, col_rank: int, row_rank: int, margin: float) -> dict:
    return {
        "column_softmax_prob": prob,
        "raw_logit": logit,
        "column_rank": col_rank,
        "row_rank_by_raw_logit": row_rank,
        "column_margin_to_top1": margin,
    }


def test_fork_evidence_summarises_both_daughter_edges() -> None:
    evidence = fork_evidence(
        edge_record(0.90, 3.0, 1, 1, 0.0),
        edge_record(0.40, 1.0, 2, 2, 0.15),
    )

    assert evidence["min_prob"] == pytest.approx(0.40)
    assert evidence["max_prob"] == pytest.approx(0.90)
    assert evidence["prob_product"] == pytest.approx(0.36)
    assert evidence["prob_gap"] == pytest.approx(0.50)
    assert evidence["sum_raw_logit"] == pytest.approx(4.0)
    assert evidence["min_raw_logit"] == pytest.approx(1.0)
    assert evidence["max_column_rank"] == 2
    assert evidence["min_column_rank"] == 1
    assert evidence["both_column_rank_le_2"] is True
    assert evidence["both_row_rank_le_2"] is True


def test_fork_evidence_flags_a_low_ranked_daughter() -> None:
    evidence = fork_evidence(
        edge_record(0.90, 3.0, 1, 1, 0.0),
        edge_record(0.02, -2.0, 7, 9, 0.80),
    )

    assert evidence["both_column_rank_le_2"] is False
    assert evidence["both_row_rank_le_3"] is False
    assert evidence["max_competitor_margin"] == pytest.approx(0.80)


def test_fork_evidence_is_symmetric_in_its_two_edges() -> None:
    a = edge_record(0.90, 3.0, 1, 1, 0.0)
    b = edge_record(0.40, 1.0, 2, 2, 0.15)

    assert fork_evidence(a, b) == fork_evidence(b, a)


# ============================================================
# Node mapping (fail-closed)
# ============================================================


def saved_frame(coords: list[tuple[int, int, int, int]], node_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": node_ids,
            "t": [c[0] for c in coords],
            "z": [c[1] for c in coords],
            "y": [c[2] for c in coords],
            "x": [c[3] for c in coords],
        }
    )


def test_node_mapping_accepts_an_exact_coordinate_match() -> None:
    replay = np.array([[7, 1, 10, 20], [7, 2, 30, 40]], dtype=np.int16)
    saved = saved_frame([(7, 1, 40, 80), (7, 2, 120, 160)], [11, 12])

    mapping = build_node_mapping("d", 7, replay, saved, (1.0, 4.0, 4.0))

    assert mapping.is_valid
    assert mapping.index_of(12) == 1
    assert mapping.index_of(999) is None


def test_node_mapping_fails_closed_on_a_coordinate_disagreement() -> None:
    replay = np.array([[7, 1, 10, 20]], dtype=np.int16)
    saved = saved_frame([(7, 1, 44, 80)], [11])

    mapping = build_node_mapping("d", 7, replay, saved, (1.0, 4.0, 4.0))

    assert mapping.coordinates_match is False
    assert mapping.is_valid is False


def test_node_mapping_fails_closed_on_a_count_disagreement() -> None:
    replay = np.array([[7, 1, 10, 20], [7, 1, 11, 21]], dtype=np.int16)
    saved = saved_frame([(7, 1, 40, 80)], [11])

    mapping = build_node_mapping("d", 7, replay, saved, (1.0, 4.0, 4.0))

    assert mapping.is_valid is False


# ============================================================
# No graph mutation
# ============================================================


def test_replay_helpers_never_touch_a_graph() -> None:
    graph = td.graph.IndexedRXGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    for key in ("edge_prob", "edge_dist"):
        graph.add_edge_attr_key(key, pl.Float64, 0.0)

    parent = graph.add_node({"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    child = graph.add_node({"t": 1, "z": 0.0, "y": 4.0, "x": 0.0})
    graph.add_edge(parent, child, {"edge_prob": 0.9, "edge_dist": 1.0})

    before = (
        {int(n) for n in graph.node_ids()},
        {(int(s), int(t)) for s, t in graph.edge_list()},
    )

    frame = graph.node_attrs(unpack=True).to_pandas()
    build_node_mapping("d", 1, np.array([[1, 0, 4, 0]], dtype=np.int16), frame[frame.t == 1], (1.0, 1.0, 1.0))
    for rule in CANDIDATE_RULES:
        rule_candidate_edges(rule, PROBS, RAW)

    after = (
        {int(n) for n in graph.node_ids()},
        {(int(s), int(t)) for s, t in graph.edge_list()},
    )
    assert after == before
